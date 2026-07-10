import re

def split_sentences(text):
    """
    Cleans raw text and splits it into individual sentences.

    It normalizes whitespace by replacing all consecutive whitespace characters (including newlines)
    with a single space, splits the text using lookbehind assertions for punctuation characters (., !, ?),
    and filters out empty strings.

    Args:
        text (str): The raw input text.

    Returns:
        list[str]: A list of cleaned sentence strings.
    """
    if not text:
        return []
    
    # Replace all newlines and multiple spaces with a single space
    clean_text = re.sub(r'\s+', ' ', text)
    
    # Split by period, question mark, or exclamation mark followed by a space
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    sentences = []
    for sentence in raw_sentences:
        cleaned = sentence.strip()
        if cleaned:
            sentences.append(cleaned)
            
    return sentences
