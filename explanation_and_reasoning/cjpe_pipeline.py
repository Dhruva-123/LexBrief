import torch
import numpy as np
import itertools
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from utils.sentence_splitter import split_sentences
import os

_MODEL = None
_TOKENIZER = None

def get_model_and_tokenizer():
    """
    Retrieves and caches the pretrained XLNet model and tokenizer for CJPE (Court Judgment Prediction and Explanation).

    Returns:
        tuple: (model, tokenizer) where model is XLNetForSequenceClassification and tokenizer is XLNetTokenizer.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "CJPE_XLNet"))
        _TOKENIZER = XLNetTokenizer.from_pretrained(model_path)
        _MODEL = XLNetForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(device)
    return _MODEL, _TOKENIZER

def pad_sequences(sequences, maxlen, value=0, padding="pre"):
    """
    Pads or truncates a list of sequences to a uniform length.

    Args:
        sequences (list[list]): Input list of token or index sequences.
        maxlen (int): Target length.
        value (int, optional): Padding token/value. Defaults to 0.
        padding (str, optional): Padding position ('pre' or 'post'). Defaults to 'pre'.

    Returns:
        list[list]: List of padded/truncated sequences.
    """
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if padding == "pre":
                new_seq = seq[-maxlen:]
            else:
                new_seq = seq[:maxlen]
        else:
            num_pads = maxlen - len(seq)
            if padding == "pre":
                new_seq = [value] * num_pads + list(seq)
            else:
                new_seq = list(seq) + [value] * num_pads
        padded.append(new_seq)
    return padded

def xlnet_tokenize(sents, tokenizer):
    """
    Tokenizes a list of sentence strings using the XLNet tokenizer.

    Args:
        sents (list[str]): List of sentence strings.
        tokenizer (XLNetTokenizer): Pre-loaded tokenizer.

    Returns:
        list[list[str]]: Nested list containing tokenized representations of each sentence.
    """
    tok_sents = []
    for sen in sents:
        tok_sents.append(tokenizer.tokenize(sen))
    return tok_sents

def sentence_marker(tokenized_sents):
    """
    Generates a sentence boundary identifier index/marker mask matching the token list shape.

    It labels the start token of the N-th sentence with integer N, and all subsequent tokens
    within the same sentence with 0.

    Args:
        tokenized_sents (list[list[str]]): List of tokenized sentences.

    Returns:
        list[list[int]]: Nested lists of boundary markers.
    """
    marker_array = []
    sent_num = 1
    for tokenized_sentence in tokenized_sents:
        sentence_marker = []
        for i in range(len(tokenized_sentence)):
            if i == 0:
                sentence_marker.append(sent_num)
            else:
                sentence_marker.append(0)
        sent_num += 1
        marker_array.append(sentence_marker)
    return marker_array

def chunked_tokens_maker(all_toks, markers):
    """
    Splits the flattened list of tokens and markers into overlapping chunks for input into XLNet.

    Args:
        all_toks (list[str]): Flat list of tokens.
        markers (list[int]): Flat list of boundary markers.

    Returns:
        tuple[list[list[str]], list[list[int]]]: Tuple of chunked tokens and markers.
    """
    splitted_toks = []
    splitted_markers = []
    l = 0
    r = 510
    while l < len(all_toks):
        splitted_toks.append(all_toks[l:min(r, len(all_toks))])
        splitted_markers.append(markers[l:min(r, len(markers))])
        l += 410
        r += 410
    return splitted_toks, splitted_markers

def calculate_num_of_sents(chunk_marker_list):
    """
    Counts the number of unique sentences present in a chunk using its marker list.

    Args:
        chunk_marker_list (list[int]): The list of markers for a given chunk.

    Returns:
        int: Count of unique sentences.
    """
    ct = 0
    for i in range(len(chunk_marker_list)):
        if chunk_marker_list[i] != 0:
            ct += 1
    return ct - 1

def get_global_sentence_idx(global_token_idx, markers):
    """
    Scans backward from global_token_idx in markers list to find the active sentence number.
    
    Args:
        global_token_idx (int): The current global token position in the markers list.
        markers (list[int]): Flat markers array containing 1-based sentence indices.
        
    Returns:
        int: 0-based global sentence index.
    """
    for idx in range(global_token_idx, -1, -1):
        if markers[idx] != 0:
            return markers[idx] - 1
    return 0

def sentence_tokens_maker(marks):
    """
    Extracts token index ranges (start, end) for each sentence in a chunk.

    Args:
        marks (list[int]): Local chunk markers with custom delimiters.

    Returns:
        list[tuple[int, int]]: List of (start_index, end_index) bounds.
    """
    pair_of_ids = []
    st = -1000
    ed = -1000
    for i, mark in enumerate(marks):
        if mark == -777:
            st = i
        if mark != -777 and mark != 777 and mark != 0:
            ed = i - 1
            pair_of_ids.append((st, ed))
            st = i
        if mark == 777:
            ed = i
            pair_of_ids.append((st, ed))
    return pair_of_ids

def att_masking(input_ids):
    """
    Creates attention masks (1 for real tokens, 0 for padding tokens) for a list of padded sequences.

    Args:
        input_ids (list[list[int]]): Encoded token index lists.

    Returns:
        list[list[int]]: Binary attention mask lists.
    """
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks

def get_output_for_one_vec(input_id, att_mask, model, device):
    """
    Obtains prediction logits for a single sequence vector from XLNet.

    Args:
        input_id (list[int]): Single list of token IDs.
        att_mask (list[int]): Attention mask list.
        model (XLNetForSequenceClassification): The classification model.
        device (torch.device): CUDA or CPU device.

    Returns:
        Tensor: Output logits tensor.
    """
    input_ids = torch.tensor(input_id)
    att_masks = torch.tensor(att_mask)
    input_ids = input_ids.unsqueeze(0)
    att_masks = att_masks.unsqueeze(0)
    model.eval()
    input_ids = input_ids.to(device)
    att_masks = att_masks.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks)
        logits = outputs.logits
    return logits

def get_XLNet_output_logits(encoded_sents, tokenizer, model, device):
    """
    Wraps sequence padding and attention masking to retrieve logits for an encoded sentence segment.

    Args:
        encoded_sents (list[int]): Encoded token IDs.
        tokenizer (XLNetTokenizer): Pre-loaded tokenizer.
        model (XLNetForSequenceClassification): The classification model.
        device (torch.device): Target hardware device.

    Returns:
        Tensor: Output classification logits.
    """
    e_sents = [encoded_sents]
    e_sents = pad_sequences(e_sents, maxlen=512, value=0, padding="pre")
    att_masks = att_masking(e_sents)
    return get_output_for_one_vec(e_sents[0], att_masks[0], model, device)

def generate_cjpe_explanation(text, sentence_metadata=None):
    """
    Runs Court Judgment Prediction and Explanation (CJPE) via a hierarchical occlusion method.

    It tokenizes the text, processes overlapping chunks, obtains the predicted judgment outcome,
    occludes (pads out) each sentence in turn to calculate its contribution score to the model's
    prediction, and outputs ranked explanation passages.

    Args:
        text (str): Raw text of the legal judgment.
        sentence_metadata (list[dict], optional): Sentence metadata mapping tokens back to UI structures.

    Returns:
        dict: Predictions (label, confidence), ranked explanation passages, raw outputs, and method details.
    """
    model, tokenizer = get_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if sentence_metadata:
        sents = [meta["sentence"] for meta in sentence_metadata]
    else:
        sents = split_sentences(text)
        
    if not sents:
        return {
            "prediction": {"label": "Unknown", "confidence": 0.0},
            "explanation_passages": [],
            "raw_explanation_text": "",
            "raw_explanation_passages": [],
            "method": "CJPE Hierarchical Occlusion"
        }
        
    xlnet_tokenized_sents = xlnet_tokenize(sents, tokenizer)
    marked_tokenized_sents = sentence_marker(xlnet_tokenized_sents)
    
    xlnet_tokens = list(itertools.chain.from_iterable(xlnet_tokenized_sents))
    markers = list(itertools.chain.from_iterable(marked_tokenized_sents))
    
    if len(xlnet_tokens) > 10000:
        xlnet_tokens = xlnet_tokens[-10000:]
        markers = markers[-10000:]
        
    chunked_xlnet_tokens, chunked_markers = chunked_tokens_maker(xlnet_tokens, markers)
    
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD = tokenizer.pad_token
    
    chunk_logits_list = []
    for chunk_number in range(len(chunked_xlnet_tokens)):
        chunk_toks = chunked_xlnet_tokens[chunk_number] + [SEP] + [CLS]
        encoded_sents = tokenizer.convert_tokens_to_ids(chunk_toks)
        logits = get_XLNet_output_logits(encoded_sents, tokenizer, model, device)
        chunk_logits_list.append(logits[0].cpu().numpy())
        
    # Voting Ensemble prediction
    chunk_preds = [int(np.argmax(logits)) for logits in chunk_logits_list]
    vote_val = sum(chunk_preds) / len(chunk_preds)
    predicted_label = 1 if vote_val > 0.5 else 0
    confidence = vote_val if predicted_label == 1 else (1.0 - vote_val)
    
    label_text = "Appeal Allowed" if predicted_label == 1 else "Appeal Dismissed"
    
    # Store chunk-by-chunk raw selections to build cjpe_raw_output
    selected_raw_list = []
    
    for chunk_number in range(len(chunked_xlnet_tokens)):
        logits_val = chunk_logits_list[chunk_number]
        # Fallback logit thresholding rule for positive chunk selection
        chunk_score = logits_val[predicted_label] - logits_val[1 - predicted_label]
        if chunk_score <= 0:
            continue
            
        chunk_marks = list(chunked_markers[chunk_number])
        
        if chunk_number == 0:
            chunk_marks[0] = -777
            chunk_marks[-1] = 777
        else:
            if len(chunk_marks) < 101:
                continue
            chunk_marks[100] = -777
            chunk_marks[-1] = 777
            
        ct_sent = calculate_num_of_sents(chunk_marks)
        top_k = int(0.4 * ct_sent)
        
        pair_of_ids = sentence_tokens_maker(chunk_marks)
        # Sentinel index range mapping (-1000 mapped to 0)
        for i in range(len(pair_of_ids)):
            if pair_of_ids[i][0] == -1000:
                pair_of_ids[i] = (0, pair_of_ids[i][1])
                
        chunk_toks = chunked_xlnet_tokens[chunk_number] + [SEP] + [CLS]
        original_logits = get_XLNet_output_logits(tokenizer.convert_tokens_to_ids(chunk_toks), tokenizer, model, device)
        original_score = float(original_logits[0][predicted_label])
        
        dict_sent_to_score = {}
        
        for i in range(len(pair_of_ids)):
            st_idx, ed_idx = pair_of_ids[i]
            normalizing_length = ed_idx - st_idx + 1
            if normalizing_length == 0:
                continue
                
            left = chunked_xlnet_tokens[chunk_number][:st_idx]
            right = chunked_xlnet_tokens[chunk_number][ed_idx+1:]
            pad_sentence = [PAD] * normalizing_length
            
            final_tok_sequence = left + pad_sentence + right + [SEP] + [CLS]
            encoded_sents = tokenizer.convert_tokens_to_ids(final_tok_sequence)
            logits = get_XLNet_output_logits(encoded_sents, tokenizer, model, device)
            score_for_predicted_label = float(logits[0][predicted_label])
            
            sent_score = original_score - score_for_predicted_label
            sent_score_norm = sent_score / normalizing_length
            
            # Map start token to the correct global sentence index
            global_start_idx = chunk_number * 410 + st_idx
            sent_idx = get_global_sentence_idx(global_start_idx, markers)
            
            sentence_in_words = tokenizer.convert_tokens_to_string(chunked_xlnet_tokens[chunk_number][st_idx:ed_idx+1])
            dict_sent_to_score[sentence_in_words] = (sent_score_norm, sent_idx, chunked_xlnet_tokens[chunk_number][st_idx:ed_idx+1])
            
        sort_scores = sorted(dict_sent_to_score.items(), key=lambda x: x[1][0], reverse=True)
        for text_str, (score_val, sent_idx, toks) in sort_scores[:top_k]:
            selected_raw_list.append((text_str, (score_val, sent_idx, toks)))
            
    # Build cjpe_raw_output (exact original duplicate occurrences and order)
    raw_explanation_passages = []
    raw_explanation_text_parts = []
    for rank_idx, (text_str, (score_val, sent_idx, toks)) in enumerate(selected_raw_list, start=1):
        raw_explanation_text_parts.append(text_str)
        raw_explanation_passages.append({
            "text": text_str,
            "importance_score": float(score_val),
            "rank": rank_idx,
            "sentence_index": sent_idx
        })
    raw_explanation_text = "".join(raw_explanation_text_parts)
    
    # Build application_display_output (deduplicated by sentence_index, ranked globally by score)
    dedup_display = {}
    for text_str, (score_val, sent_idx, toks) in selected_raw_list:
        if sent_idx not in dedup_display or score_val > dedup_display[sent_idx][0]:
            dedup_display[sent_idx] = (score_val, text_str, toks)
            
    sorted_display = sorted(dedup_display.items(), key=lambda x: x[1][0], reverse=True)
    
    explanation_passages = []
    for rank, (sent_idx, (score, text_str, toks)) in enumerate(sorted_display, start=1):
        passage_text = text_str
        page_num = None
        sent_ids = []
        if sentence_metadata and 0 <= sent_idx < len(sentence_metadata):
            meta = sentence_metadata[sent_idx]
            page_num = meta.get("page_number")
            sent_ids = [meta.get("sentence_id")]
            passage_text = meta.get("sentence")
            
        explanation_passages.append({
            "text": passage_text,
            "importance_score": float(score),
            "rank": rank,
            "page_number": page_num,
            "sentence_ids": sent_ids,
            "sentence_index": sent_idx
        })
        
    return {
        "prediction": {
            "label": label_text,
            "confidence": confidence
        },
        "explanation_passages": explanation_passages,
        "raw_explanation_text": raw_explanation_text,
        "raw_explanation_passages": raw_explanation_passages,
        "method": "CJPE Hierarchical Occlusion"
    }
