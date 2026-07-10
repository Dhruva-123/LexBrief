import re

# Initialize a blank SpaCy English pipeline with only the sentencizer component
# This is lightweight, fast, and does not require downloading heavy model weights (like en_core_web_sm)
try:
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
except Exception as e:
    print(f"Warning: Failed to initialize SpaCy sentencizer: {e}. Fallback regex splitter will be used.")
    nlp = None

def split_sentences(text: str) -> list:
    """
    Splits text into sentences using SpaCy's sentencizer.
    If SpaCy is unavailable, falls back to a regex-based sentence boundary detector.
    """
    if not text or not text.strip():
        return []

    # Replace multiple spaces/newlines with a single space to clean input
    normalized_text = re.sub(r'\s+', ' ', text).strip()
    
    if nlp is not None:
        doc = nlp(normalized_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    else:
        # Regex fallback: split by periods/questions/exclamations followed by capital letters
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(normalized_text)
        return [s.strip() for s in sentences if s.strip()]
