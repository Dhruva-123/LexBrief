import sys
import os
import torch
import json
import string
import streamlit as st

# Setup sys.path to resolve imports within rhetorical_role_module
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "rhetorical_role_module")
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Explicit verified configuration parameters for the target checkpoint model_state4.tar
PRETRAINED = False
EMB_DIM = 200
WORD_EMB_DIM = 100
HIDDEN_DIM = 200
VOCAB_SIZE = 164108
N_TAGS = 10

@st.cache_resource
def load_rhetorical_resources():
    """
    Loads and caches the Hierarchical BiLSTM-CRF model, word vocabulary mapping,
    and label index mappings using Streamlit's cache_resource.
    Performs strict validation of the checkpoint's architecture and weights.
    """
    # Import inside the function to ensure sys.path is already updated
    from models.hierarchical_bilstm_crf.model import Hier_LSTM_CRF_Classifier
    from models.hierarchical_bilstm_crf.checkpoint_loader import load_or_download_checkpoint
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Download/load model checkpoints
    ckpt_paths = load_or_download_checkpoint()
    
    with open(ckpt_paths["tag2idx"], "r") as fp:
        tag2idx = json.load(fp)
        
    with open(ckpt_paths["word2idx"], "r") as fp:
        word2idx = json.load(fp)
        
    # Verify vocab sizes
    if len(tag2idx) != N_TAGS:
        raise ValueError(f"Tag vocabulary size mismatch: expected {N_TAGS}, got {len(tag2idx)}")
    if len(word2idx) != VOCAB_SIZE:
        raise ValueError(f"Word vocabulary size mismatch: expected {VOCAB_SIZE}, got {len(word2idx)}")
        
    # Load weights checkpoint
    checkpoint = torch.load(ckpt_paths["model_state"], map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    # STRICT CHECKPOINT VALIDATION: Check compatibility of keys and tensor shapes
    expected_shapes = {
        "sent_encoder.emb.weight": (VOCAB_SIZE, WORD_EMB_DIM),
        "sent_encoder.lstm.weight_ih_l0": (400, WORD_EMB_DIM),
        "emitter.lstm.weight_ih_l0": (400, EMB_DIM),
        "crf.transitions": (N_TAGS, N_TAGS)
    }
    
    for key, expected_shape in expected_shapes.items():
        if key not in state_dict:
            raise ValueError(
                f"Model compatibility error: Required parameter key '{key}' is missing from the checkpoint state_dict. "
                f"Expected pretrained = {PRETRAINED} configuration."
            )
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Model shape mismatch for parameter '{key}': Expected {expected_shape}, got {actual_shape} from checkpoint."
            )
            
    # Instantiate model with explicit validated arguments
    model = Hier_LSTM_CRF_Classifier(
        n_tags=N_TAGS,
        sent_emb_dim=EMB_DIM,
        sos_tag_idx=tag2idx["<start>"],
        eos_tag_idx=tag2idx["<end>"],
        pad_tag_idx=tag2idx["<pad>"],
        vocab_size=VOCAB_SIZE,
        word_emb_dim=WORD_EMB_DIM,
        pretrained=PRETRAINED,
        device=device
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    idx_to_tag = {v: k for k, v in tag2idx.items()}
    
    return model, word2idx, idx_to_tag, device

def classify_judgment_rhetorical_roles(sentences, sentence_metadata=None):
    """
    Classifies a list of judgment sentences using the real Hierarchical BiLSTM-CRF model.
    Checks that the prediction count matches the input sentence count exactly.
    Returns structured records matching Phase 3 requirements.
    """
    if not sentences:
        return []
        
    model, word2idx, idx_to_tag, device = load_rhetorical_resources()
    
    # Track sentence mapping and empty/skipped sentences
    x_doc = []
    active_indices = []
    
    for idx, sent_text in enumerate(sentences):
        # Match original non-pretrained cleaning/tokenization exactly
        cleaned = sent_text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = cleaned.split()
        if words:
            word_ids = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in words]
            x_doc.append(word_ids)
            active_indices.append(idx)
            
    if not x_doc:
        pred_tag_ids = [None] * len(sentences)
        prediction_sources = ["not_inferred_empty_model_input"] * len(sentences)
    else:
        x = [x_doc]
        
        # Run inference
        with torch.no_grad():
            pred_tag_ids_batch = model(x)
            
        active_preds = pred_tag_ids_batch[0]
        
        # Validate count match between model inputs and predictions
        if len(active_preds) != len(x_doc):
            raise ValueError(
                f"Model input sentences count ({len(x_doc)}) does not match decoded predictions count ({len(active_preds)})."
            )
            
        # Reconstruct prediction sequence
        pred_tag_ids = [None] * len(sentences)
        prediction_sources = [None] * len(sentences)
        active_idx = 0
        for idx in range(len(sentences)):
            if idx in active_indices:
                pred_tag_ids[idx] = active_preds[active_idx]
                active_idx += 1
            else:
                pred_tag_ids[idx] = None
                prediction_sources[idx] = "not_inferred_empty_model_input"
                
    # Validate count match
    if len(pred_tag_ids) != len(sentences):
        raise ValueError(
            f"Model prediction count ({len(pred_tag_ids)}) does not match input sentence count ({len(sentences)})."
        )
        
    # Import OFFICIAL_TO_UI to convert model tags to the UI-expected display strings
    from utils.document_state import OFFICIAL_TO_UI
    
    records = []
    for idx, (original_sent, tag_id) in enumerate(zip(sentences, pred_tag_ids)):
        source = prediction_sources[idx]
        if tag_id is not None:
            role = idx_to_tag.get(tag_id, "None")
            ui_role = OFFICIAL_TO_UI.get(role, "NONE")
        else:
            role = None
            ui_role = None
        
        # Match with original sentence metadata if available
        if sentence_metadata and idx < len(sentence_metadata):
            meta = sentence_metadata[idx]
            sentence_id = meta.get("sentence_id")
            page_number = meta.get("page_number")
            page_sentence_index = meta.get("page_sentence_index")
            document_sentence_index = meta.get("document_sentence_index")
        else:
            sentence_id = f"s{idx}"
            page_number = None
            page_sentence_index = None
            document_sentence_index = idx
            
        records.append({
            "sentence_id": sentence_id,
            "sentence": original_sent,
            "role": role,
            "ui_role": ui_role,
            "role_id": tag_id,
            "page_number": page_number,
            "page_sentence_index": page_sentence_index,
            "document_sentence_index": document_sentence_index,
            "prediction_source": source
        })
        
    return records
