def filter_sentences_by_role(sentences: list, role: str) -> list:
    """
    Filters a list of SentenceAnalysis objects by their rhetorical role.
    
    If role == "All", returns the entire list.
    Otherwise, returns only those sentences matching the given role.
    """
    if not role or role == "All":
        return sentences
    return [s for s in sentences if s.role == role]
