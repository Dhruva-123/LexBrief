import re
import json
import time
import nltk
import ollama

from .config import (
    MAX_TOKENS,
    BUDGET_FACTS,
    BUDGET_ANALYSIS,
    BUDGET_CONCLUSION,
    OVERLAP_SENTENCES
)

# Ensure NLTK packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================================================
# Lazy Loading for Hugging Face Pegasus
# =============================================================================
pegasus_tokenizer = None
pegasus_model = None

def get_pegasus_model():
    """
    Lazy load Legal-Pegasus tokenizer and model on CPU or GPU.
    """
    global pegasus_tokenizer, pegasus_model
    if pegasus_model is None:
        print("[SAC Engine] Loading nsi319/legal-pegasus locally...")
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        import torch
        
        model_id = 'nsi319/legal-pegasus'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_id)
        pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_id).to(device)
        print(f"[SAC Engine] Legal-Pegasus loaded successfully on device: {device}")
        
    return pegasus_tokenizer, pegasus_model


# =============================================================================
# Rhetorical Boundary Heuristics & Trigger Patterns
# =============================================================================
CONCLUSION_TRIGGERS = [
    r"the appeal is (?:accordingly\s+)?(?:allowed|dismissed)",
    r"the petition is (?:accordingly\s+)?disposed of",
    r"for the (?:above|reasons|aforesaid)",
    r"in the result",
    r"we are of the considered view",
    r"we,?\s+therefore,?\s+hold that",
]

ARGUMENT_TRIGGERS = [
    r"learned counsel for the (?:petitioner|appellant|respondent)",
    r"it was contended (?:by|that)",
    r"per contra",
    r"the issue for consideration is",
]

def segment_document_heuristic(sentences):
    """
    SAC-H: Heuristic rhetorical segmentation using regex boundary triggers.
    """
    n = len(sentences)
    if n == 0:
        return [], [], []

    # Step 1: Conclusion Identification (final 20%)
    conclusion_start_idx = None
    for i in range(max(0, int(n * 0.80)), n):
        for pat in CONCLUSION_TRIGGERS:
            if re.search(pat, sentences[i], re.IGNORECASE):
                conclusion_start_idx = i
                break
        if conclusion_start_idx is not None:
            break
    if conclusion_start_idx is None:
        conclusion_start_idx = max(0, int(n * 0.85))

    # Step 2: Arguments & Analysis Identification (before conclusion)
    arg_start_idx = None
    for i in range(0, conclusion_start_idx):
        for pat in ARGUMENT_TRIGGERS:
            if re.search(pat, sentences[i], re.IGNORECASE):
                arg_start_idx = i
                break
        if arg_start_idx is not None:
            break
    if arg_start_idx is None:
        arg_start_idx = max(0, int(n * 0.30))

    # Step 3: Section Delineation
    return (
        sentences[:arg_start_idx],
        sentences[arg_start_idx:conclusion_start_idx],
        sentences[conclusion_start_idx:]
    )


def segment_document_llm(sentences, model_name="qwen3:8b"):
    """
    SAC-LLM: Zero-shot boundary detection using local Qwen.
    """
    full_text = " ".join(sentences)
    
    prompt = f"""Analyze the following legal judgment. Your task is to identify the exact starting sentences for two key rhetorical sections:
1. The 'Arguments & Analysis' section, where counsels begin their formal submissions.
2. The 'Conclusion' section, where the final verdict is delivered.

Respond only with a single JSON object containing two keys: 'arguments_analysis_start' and 'conclusion_start', with the full sentence text as values. Do not output any thinking or markdown code blocks, just raw JSON.

DOCUMENT: {full_text}"""

    # Pegasus is an encoder-decoder summarization model, so it cannot perform JSON segmentation tasks.
    # Fallback to local Qwen via Ollama.
    run_model = model_name
    if "pegasus" in model_name.lower():
        run_model = "qwen3:8b"

    for attempt in range(3):
        try:
            # Query local Ollama model
            response = ollama.chat(
                model=run_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1}
            )
            response_text = response['message']['content'].strip()
            
            # Clean formatting if any exists
            response_text = re.sub(r'```json|```', '', response_text).strip()
            parsed = json.loads(response_text)
            
            arg_start_sent = parsed.get("arguments_analysis_start", "")
            conc_start_sent = parsed.get("conclusion_start", "")

            arg_start_idx = None
            conc_start_idx = None

            # Substring matching to find boundaries
            for i, sent in enumerate(sentences):
                if arg_start_idx is None and arg_start_sent[:50] in sent:
                    arg_start_idx = i
                if conc_start_idx is None and conc_start_sent[:50] in sent:
                    conc_start_idx = i

            if arg_start_idx is None or conc_start_idx is None:
                raise ValueError("Could not match boundary sentences")
            if not (0 < arg_start_idx < conc_start_idx < len(sentences)):
                raise ValueError("Invalid boundary ordering")

            return (
                sentences[:arg_start_idx],
                sentences[arg_start_idx:conc_start_idx],
                sentences[conc_start_idx:]
            )
        except Exception as e:
            print(f"[SAC Engine] SAC-LLM Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(1)
            else:
                print("[SAC Engine] All attempts failed. Falling back to heuristic segmentation (SAC-H).")
                return segment_document_heuristic(sentences)


# =============================================================================
# Sliding Window Chunker & Token Count Heuristics
# =============================================================================
def count_tokens(sentences):
    """
    Approximates tokens in a list of sentences based on a 1.3 word-to-token ratio.
    """
    text = " ".join(sentences)
    return int(len(text.split()) * 1.3)


def chunk_with_overlap(sentences, max_tokens=MAX_TOKENS, overlap=OVERLAP_SENTENCES):
    """
    SAC-H+: Intra-section chunking with sliding-window overlap context.
    """
    if not sentences:
        return []

    chunks = []
    current_sents = []
    prev_chunk_tail = []

    for sent in sentences:
        candidate = prev_chunk_tail + current_sents + [sent]
        if count_tokens(candidate) <= max_tokens:
            current_sents.append(sent)
        else:
            if current_sents:
                full_chunk = prev_chunk_tail + current_sents
                chunks.append(full_chunk)
                prev_chunk_tail = full_chunk[-overlap:]
                current_sents = [sent]
            else:
                current_sents.append(sent)

    if current_sents:
        chunks.append(prev_chunk_tail + current_sents)

    return chunks


# =============================================================================
# Core Summarizers
# =============================================================================
def summarize_chunk(text, target_words, model_name="qwen3:8b"):
    """
    Summarize a single text chunk using local Qwen via Ollama or local Legal-Pegasus.
    """
    if "pegasus" in model_name.lower():
        try:
            import torch
            tokenizer, model = get_pegasus_model()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            inputs = tokenizer(
                text, max_length=1024, truncation=True,
                padding='longest', return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=target_words,
                    min_new_tokens=max(20, target_words // 4),
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            return tokenizer.decode(ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[SAC Engine] Legal-Pegasus summarization error: {e}")
            return f"[Summarization Error: {e}]"

    # Default to local Ollama (e.g. Qwen)
    prompt = f"""You are summarizing an Indian Supreme Court judgment for a legal news publication.
Write in third person. Include judge names, case names, and specific statutes if mentioned.
Be factual and concise. Approximately {target_words} words. Our focus is on getting the highest ROUGE scores.
Respond directly with the summary text. Do not include any introductory remarks, preambles, explanations, or thinking blocks (like <think>...</think>). Start your response immediately with the summary text.

Text:
{text}

Summary:"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,
                "num_predict": max(512, int(target_words * 4))  # Ensure headroom to prevent truncation
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[SAC Engine] Ollama summarization error: {e}")
        return f"[Summarization Error: {e}]"


def summarize_section_chunks(chunks, budget_words, model_name="qwen3:8b"):
    """
    Summarizes a list of sentence chunks under a specific word budget.
    """
    if not chunks:
        return "", []

    # Target limit per chunk (minimum 30 words to keep summaries meaningful)
    words_per_chunk = max(30, budget_words // len(chunks))
    summaries = []
    
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        summary = summarize_chunk(chunk_text, words_per_chunk, model_name=model_name)
        summaries.append(summary)

    concatenated = " ".join(summaries)
    return concatenated, summaries


# =============================================================================
# Main Entry Point
# =============================================================================
def generate_legal_summary(document_text, model_name="qwen3:8b", use_llm_segmentation=False):
    """
    Stand-alone pipeline function to generate abstractive summaries for long legal documents
    according to Structure-Aware Chunking (SAC) rules.
    
    Parameters
    ----------
    document_text : str
        Raw text of the legal judgment/document.
    model_name : str, optional
        Name of the local model to use for summarization:
        - "nsi319/legal-pegasus (Hugging Face)" or "pegasus" for Hugging Face Legal-Pegasus.
        - "qwen3:8b" or any other local Ollama models.
    use_llm_segmentation : bool, optional
        - True: Uses SAC-LLM (LLM-based rhetorical boundary detection via Qwen/Ollama).
        - False (default): Uses SAC-H (lexical regex trigger-based heuristic segmentation).
        
    Returns
    -------
    dict
        A dictionary containing:
          - "final_summary" (str): Combined summary of all sections.
          - "section_summaries" (dict): Individual summaries for Facts, Analysis, and Conclusion.
          - "segmented_text" (dict): Extracted raw text split by section.
          - "sentence_counts" (dict): Number of sentences per section.
          - "chunk_counts" (dict): Number of 1024-token chunks per section.
    """
    # 1. Tokenize document into sentences
    sentences = nltk.sent_tokenize(document_text)

    # 2. Rhetorical Segmentation (SAC-H or SAC-LLM)
    if use_llm_segmentation:
        facts_sents, analysis_sents, conclusion_sents = segment_document_llm(sentences, model_name=model_name)
    else:
        facts_sents, analysis_sents, conclusion_sents = segment_document_heuristic(sentences)

    # 3. Intra-section chunking with overlap (SAC-H+ sliding window)
    facts_chunks = chunk_with_overlap(facts_sents, max_tokens=MAX_TOKENS, overlap=OVERLAP_SENTENCES)
    analysis_chunks = chunk_with_overlap(analysis_sents, max_tokens=MAX_TOKENS, overlap=OVERLAP_SENTENCES)
    conclusion_chunks = chunk_with_overlap(conclusion_sents, max_tokens=MAX_TOKENS, overlap=OVERLAP_SENTENCES)

    # 4. Summarize per section under PBA budget
    facts_summary, facts_chunk_summaries = summarize_section_chunks(facts_chunks, BUDGET_FACTS, model_name)
    analysis_summary, analysis_chunk_summaries = summarize_section_chunks(analysis_chunks, BUDGET_ANALYSIS, model_name)
    conclusion_summary, conclusion_chunk_summaries = summarize_section_chunks(conclusion_chunks, BUDGET_CONCLUSION, model_name)

    # 5. Concatenate
    final_summary = f"{facts_summary} {analysis_summary} {conclusion_summary}".strip()

    return {
        "final_summary": final_summary,
        "section_summaries": {
            "facts": facts_summary,
            "analysis": analysis_summary,
            "conclusion": conclusion_summary
        },
        "segmented_text": {
            "facts": " ".join(facts_sents),
            "analysis": " ".join(analysis_sents),
            "conclusion": " ".join(conclusion_sents)
        },
        "sentence_counts": {
            "facts": len(facts_sents),
            "analysis": len(analysis_sents),
            "conclusion": len(conclusion_sents)
        },
        "chunk_counts": {
            "facts": len(facts_chunks),
            "analysis": len(analysis_chunks),
            "conclusion": len(conclusion_chunks)
        }
    }
