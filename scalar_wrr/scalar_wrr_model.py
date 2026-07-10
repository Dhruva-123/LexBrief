import os
import re
import json
import numpy as np
import torch
import nltk
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Ensure NLTK packages are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 1. Device and Model Paths
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Resolve absolute path to models directory relative to project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, 'models', 'legal-pegasus')
MAX_INPUT_TOKENS = 1024

# Lazy load model and tokenizer
_tokenizer = None
_model = None

def get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)
        _model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    return _model, _tokenizer

# 2. Heuristic Classifiers and Constants
RHETORICAL_ROLES = {
    'facts': 0, 'issue': 1, 'arguments': 2,
    'statute': 3, 'precedent': 4, 'ratio': 5, 'ruling': 6
}

ROLE_CUES = {
    'facts': ['the petitioner', 'the appellant', 'the respondent', 'the plaintiff',
              'on', 'dated', 'year', 'property', 'marriage', 'accident'],
    'issue': ['question', 'issue', 'whether', 'determine', 'decide'],
    'arguments': ['contended', 'submitted', 'argued', 'counsel', 'learned advocate',
                  'it was urged', 'it is contended'],
    'statute': ['section', 'act,', 'article', 'rule', 'schedule', 'clause', 'proviso'],
    'precedent': ['v.', 'vs.', 'air ', 'scc ', 'scr ', r'(\d{4})', 'reported in', 'relied upon'],
    'ratio': ['held', 'we hold', 'it is settled', 'the ratio', 'principle', 'law is'],
    'ruling': ['appeal is allowed', 'appeal is dismissed', 'petition is allowed',
               'set aside', 'accordingly', 'result', 'we direct', 'the judgment']
}

def pegasus_summarize(text, max_new_tokens=256, min_new_tokens=40):
    """Single-pass Pegasus summarization with truncation."""
    if not text.strip():
        return ''
    model_obj, tokenizer_obj = get_model_and_tokenizer()
    inputs = tokenizer_obj(
        text, max_length=MAX_INPUT_TOKENS, truncation=True,
        return_tensors='pt', padding='longest'
    ).to(DEVICE)
    with torch.no_grad():
        ids = model_obj.generate(
            **inputs, num_beams=4, max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens, length_penalty=2.0,
            early_stopping=True, no_repeat_ngram_size=3
        )
    return tokenizer_obj.decode(ids[0], skip_special_tokens=True)

def chunk_text(text, max_tokens=1000, overlap_tokens=50):
    """Split text into overlapping token-based chunks."""
    tokens = text.split()  # word-level approximation
    chunks, i = [], 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(' '.join(chunk))
        i += max_tokens - overlap_tokens
    return chunks

def system1_naive(judgment_text, max_new_tokens_chunk=150, max_new_tokens_final=256):
    """
    Naive recursive summarization:
    1. Chunk into ~1000-token pieces
    2. Summarize each chunk
    3. Concatenate chunk summaries and summarize again (recursive)
    """
    chunks = chunk_text(judgment_text, max_tokens=1000, overlap_tokens=50)

    # Level 1: summarize each chunk
    chunk_summaries = [pegasus_summarize(c, max_new_tokens=max_new_tokens_chunk) for c in chunks]

    # Level 2: concatenate and summarize again
    combined = ' '.join(chunk_summaries)

    # If still too long, do another level
    if len(combined.split()) > 800:
        level2_chunks = chunk_text(combined, max_tokens=1000, overlap_tokens=50)
        level2_summaries = [pegasus_summarize(c, max_new_tokens=max_new_tokens_chunk) for c in level2_chunks]
        combined = ' '.join(level2_summaries)

    final = pegasus_summarize(combined, max_new_tokens=max_new_tokens_final)
    return final

def classify_sentence_heuristic(sentence):
    """Rule-based rhetorical role classification."""
    s_lower = sentence.lower()
    scores = {role: 0 for role in RHETORICAL_ROLES}
    for role, cues in ROLE_CUES.items():
        for cue in cues:
            if re.search(cue, s_lower):
                scores[role] += 1
    return max(scores, key=scores.get)

def system2_rhetorical(judgment_text):
    """
    Rhetorical chunking pipeline:
    1. Sentence-split
    2. Assign role to each sentence
    3. Group sentences by role into semantic chunks
    4. Summarize each role-group
    5. Concatenate in logical order, final summarization
    """
    import nltk
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(judgment_text)

    # Step 2: assign roles
    role_groups = {role: [] for role in RHETORICAL_ROLES}
    for sent in sentences:
        role = classify_sentence_heuristic(sent)
        role_groups[role].append(sent)

    # Step 3: Summarize each non-empty group
    LOGICAL_ORDER = ['facts', 'issue', 'statute', 'precedent', 'arguments', 'ratio', 'ruling']
    role_summaries = {}
    for role in LOGICAL_ORDER:
        group_text = ' '.join(role_groups[role])
        if len(group_text.split()) > 30:
            role_summaries[role] = pegasus_summarize(group_text, max_new_tokens=100)

    # Step 4: Concatenate and final pass
    combined = ' '.join(role_summaries.get(r, '') for r in LOGICAL_ORDER if r in role_summaries)
    final = pegasus_summarize(combined, max_new_tokens=256)
    return final, role_summaries

def compute_role_weights(train_data, sample_size=200):
    """
    Compute importance weight for each role by measuring its correlation
    with high ROUGE scores in training data.

    Approximation: measure how much each role contributes to the reference summary
    by computing n-gram overlap between role-group text and gold summary.
    """
    from collections import defaultdict
    from rouge_score import rouge_scorer

    role_rouge = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    for item in train_data[:sample_size]:
        _, role_summaries = system2_rhetorical(item['judgment'])
        ref = item['summary']
        for role, text in role_summaries.items():
            if text:
                s = scorer.score(ref, text)
                role_rouge[role].append(s['rouge2'].fmeasure)

    weights = {role: np.mean(scores) if scores else 0.0 for role, scores in role_rouge.items()}
    # Normalize to sum to 1
    total = sum(weights.values()) or 1
    weights = {k: v/total for k, v in weights.items()}
    print("Role importance weights:", {k: f"{v:.3f}" for k, v in weights.items()})
    return weights

def system3_weighted(judgment_text, role_weights):
    """
    System 3: Inject role importance scores as input tags.
    High-weight roles get more emphasis via explicit tagging.
    """
    import nltk
    sentences = nltk.sent_tokenize(judgment_text)

    # Assign and tag sentences with their role + weight
    role_groups = {role: [] for role in RHETORICAL_ROLES}
    for sent in sentences:
        role = classify_sentence_heuristic(sent)
        weight = role_weights.get(role, 0.1)
        # Tag: [ROLE:weight] sentence
        tagged = f"[{role.upper()}:{weight:.2f}] {sent}"
        role_groups[role].append(tagged)

    LOGICAL_ORDER = ['facts', 'issue', 'statute', 'precedent', 'arguments', 'ratio', 'ruling']

    # Summarize each group with tags injected (role-aware fine-tuning reads these)
    role_summaries = {}
    for role in LOGICAL_ORDER:
        group_text = ' '.join(role_groups[role])
        if len(group_text.split()) > 30:
            role_summaries[role] = pegasus_summarize(group_text, max_new_tokens=100)

    combined = ' '.join(role_summaries.get(r, '') for r in LOGICAL_ORDER if r in role_summaries)
    final = pegasus_summarize(combined, max_new_tokens=256)
    return final
