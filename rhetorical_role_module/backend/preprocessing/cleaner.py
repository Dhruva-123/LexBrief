import string

def clean_text(text: str) -> str:
    """
    Standard text cleaning matching the original LegalSeg preprocessing:
    1. Lowercase the text.
    2. Translate all punctuation symbols to spaces.
    3. Collapse multiple spaces and strip.
    """
    if not text:
        return ""
    # Replace punctuation with spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned = text.lower().translate(translator)
    # Strip and split to normalize spaces
    return " ".join(cleaned.split())
