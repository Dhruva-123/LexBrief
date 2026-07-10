import os
import json
import torch
import string
from backend.schemas import DocumentAnalysis, SentenceAnalysis
from backend.pdf.extractor import extract_pdf_pages, is_scanned_pdf
from backend.pdf.sentence_mapper import map_sentences_to_pdf
from backend.preprocessing.cleaner import clean_text
from backend.preprocessing.sentence_splitter import split_sentences
from backend.rhetorical.labels import TAG_TO_IDX, IDX_TO_TAG, MODEL_TO_USER
from models.hierarchical_bilstm_crf.model import Hier_LSTM_CRF_Classifier
from models.hierarchical_bilstm_crf.checkpoint_loader import load_or_download_checkpoint
from models.embeddings.sent2vec_encoder import Sent2VecEncoder

# Explicit verified configuration parameters for the target checkpoint model_state4.tar
PRETRAINED = False
EMB_DIM = 200
WORD_EMB_DIM = 100
HIDDEN_DIM = 200
VOCAB_SIZE = 164108
N_TAGS = 10

# Global caches to avoid redundant loads
_MODEL = None
_WORD2IDX = None

def get_device() -> str:
    """Detects available hardware device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_resources() -> tuple:
    """
    Loads checkpoints and initializes the model and vocab mappings.
    Performs strict validation of the checkpoint's architecture and weights.
    """
    global _MODEL, _WORD2IDX
    
    device = get_device()
    
    if _MODEL is None:
        ckpt_paths = load_or_download_checkpoint()
        
        # Load index mappings
        with open(ckpt_paths["tag2idx"], "r") as fp:
            tag2idx = json.load(fp)
            
        with open(ckpt_paths["word2idx"], "r") as fp:
            _WORD2IDX = json.load(fp)
            
        # Verify tag and vocab size compatibility with configured parameters
        if len(tag2idx) != N_TAGS:
            raise ValueError(f"Tag vocabulary size mismatch: Config expected {N_TAGS}, got {len(tag2idx)} from tag2idx.json")
        if len(_WORD2IDX) != VOCAB_SIZE:
            raise ValueError(f"Word vocabulary size mismatch: Config expected {VOCAB_SIZE}, got {len(_WORD2IDX)} from word2idx.json")
            
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
        _MODEL = Hier_LSTM_CRF_Classifier(
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
        
        _MODEL.load_state_dict(state_dict)
        _MODEL.eval()
        print("Loaded model with strict compatibility verification successfully.")
        
    return _MODEL, _WORD2IDX

def analyze_text(text: str, force_mock: bool = False) -> DocumentAnalysis:
    """
    Analyzes raw text: splits it into sentences and predicts a rhetorical role for each.
    Coordinates and page numbers are returned as None.
    """
    if force_mock:
        raise ValueError("Mock/synthetic embeddings and mock inference mode have been completely removed.")
        
    if not text or not text.strip():
        return DocumentAnalysis(sentences=[])
        
    try:
        model, word2idx = load_resources()
        
        # Split text into sentences
        raw_sentences = split_sentences(text)
        if not raw_sentences:
            return DocumentAnalysis(sentences=[])
            
        # Match original non-pretrained cleaning, tokenization, and sentence alignment tracking
        x_doc = []
        active_indices = []
        
        for idx, sent_text in enumerate(raw_sentences):
            # Clean sentence text (lowercase + translate punctuation)
            cleaned = sent_text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            words = cleaned.split()
            if words:
                word_ids = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in words]
                x_doc.append(word_ids)
                active_indices.append(idx)
                
        if not x_doc:
            pred_tag_ids = [None] * len(raw_sentences)
            prediction_sources = ["not_inferred_empty_model_input"] * len(raw_sentences)
        else:
            x = [x_doc]
            
            # Run BiLSTM-CRF inference
            with torch.no_grad():
                pred_tag_ids_batch = model(x)
                
            active_preds = pred_tag_ids_batch[0]
            
            # Validate model-input sentence count equals decoded predictions count
            if len(active_preds) != len(x_doc):
                raise ValueError(
                    f"Model input sentences count ({len(x_doc)}) does not match decoded predictions count ({len(active_preds)})."
                )
                
            # Align predictions back to original sentence list
            pred_tag_ids = [None] * len(raw_sentences)
            prediction_sources = [None] * len(raw_sentences)
            active_idx = 0
            for idx in range(len(raw_sentences)):
                if idx in active_indices:
                    pred_tag_ids[idx] = active_preds[active_idx]
                    active_idx += 1
                else:
                    pred_tag_ids[idx] = None
                    prediction_sources[idx] = "not_inferred_empty_model_input"
                    
        # Final validation checks
        if len(pred_tag_ids) != len(raw_sentences):
            raise ValueError(
                f"Prediction count ({len(pred_tag_ids)}) does not match input sentence count ({len(raw_sentences)})."
            )
            
        sentences_analysis = []
        for idx, sent_text in enumerate(raw_sentences):
            tag_id = pred_tag_ids[idx]
            source = prediction_sources[idx]
            if tag_id is not None:
                model_tag = IDX_TO_TAG.get(tag_id, "None")
                user_role = MODEL_TO_USER.get(model_tag, "None")
            else:
                user_role = None
            
            sentences_analysis.append(SentenceAnalysis(
                sentence_id=idx,
                text=sent_text,
                role=user_role,
                role_id=tag_id,
                confidence=None,
                page_number=None,
                bbox=None,
                rects=None,
                prediction_source=source
            ))
            
        return DocumentAnalysis(sentences=sentences_analysis)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DocumentAnalysis(sentences=[], error_message=str(e))

def analyze_pdf(pdf_path: str, force_mock: bool = False) -> DocumentAnalysis:
    """
    Analyzes a legal judgment PDF file:
    1. Extracts page text and character coordinate positions.
    2. Validates if the document is scanned.
    3. Splits text into sentences.
    4. Predicts rhetorical roles.
    5. Maps sentence boundaries to page numbers and bboxes.
    """
    if force_mock:
        raise ValueError("Mock/synthetic embeddings and mock inference mode have been completely removed.")
        
    if not os.path.exists(pdf_path):
        return DocumentAnalysis(sentences=[], error_message=f"PDF file not found at {pdf_path}")
        
    try:
        pages_data = extract_pdf_pages(pdf_path)
        if is_scanned_pdf(pages_data):
            return DocumentAnalysis(sentences=[], is_scanned=True, error_message="Document appears to be scanned. OCR is not yet enabled for this module.")
            
        model, word2idx = load_resources()
        
        # Concatenate text from all pages
        full_text = " ".join(page["text"] for page in pages_data)
        
        # Split into sentences
        raw_sentences = split_sentences(full_text)
        if not raw_sentences:
            return DocumentAnalysis(sentences=[])
            
        # Match original non-pretrained cleaning, tokenization, and sentence alignment tracking
        x_doc = []
        active_indices = []
        
        for idx, sent_text in enumerate(raw_sentences):
            cleaned = sent_text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            words = cleaned.split()
            if words:
                word_ids = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in words]
                x_doc.append(word_ids)
                active_indices.append(idx)
                
        if not x_doc:
            pred_tag_ids = [None] * len(raw_sentences)
            prediction_sources = ["not_inferred_empty_model_input"] * len(raw_sentences)
        else:
            x = [x_doc]
            
            # Run BiLSTM-CRF inference
            with torch.no_grad():
                pred_tag_ids_batch = model(x)
                
            active_preds = pred_tag_ids_batch[0]
            
            # Validate input size matches model outputs
            if len(active_preds) != len(x_doc):
                raise ValueError(
                    f"Model input sentences count ({len(x_doc)}) does not match decoded predictions count ({len(active_preds)})."
                )
                
            # Align predictions back to original list
            pred_tag_ids = [None] * len(raw_sentences)
            prediction_sources = [None] * len(raw_sentences)
            active_idx = 0
            for idx in range(len(raw_sentences)):
                if idx in active_indices:
                    pred_tag_ids[idx] = active_preds[active_idx]
                    active_idx += 1
                else:
                    pred_tag_ids[idx] = None
                    prediction_sources[idx] = "not_inferred_empty_model_input"
                    
        # Final validation checks
        if len(pred_tag_ids) != len(raw_sentences):
            raise ValueError(
                f"Prediction count ({len(pred_tag_ids)}) does not match input sentence count ({len(raw_sentences)})."
            )
            
        # Align clean sentences back to word positions on PDF pages to locate coordinates
        mapped_sentences = map_sentences_to_pdf(raw_sentences, pages_data)
        
        sentences_analysis = []
        for idx, mapped in enumerate(mapped_sentences):
            tag_id = pred_tag_ids[idx]
            source = prediction_sources[idx]
            if tag_id is not None:
                model_tag = IDX_TO_TAG.get(tag_id, "None")
                user_role = MODEL_TO_USER.get(model_tag, "None")
            else:
                user_role = None
            
            sentences_analysis.append(SentenceAnalysis(
                sentence_id=idx,
                text=mapped["text"],
                role=user_role,
                role_id=tag_id,
                confidence=None,
                page_number=mapped["page_number"],
                bbox=mapped["bbox"],
                rects=mapped["rects"],
                prediction_source=source
            ))
            
        return DocumentAnalysis(sentences=sentences_analysis)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DocumentAnalysis(sentences=[], error_message=str(e))
