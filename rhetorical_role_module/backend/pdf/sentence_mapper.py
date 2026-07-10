import re

def clean_word(w: str) -> str:
    """Helper to clean a word for matching by removing non-alphanumeric characters and lowercasing."""
    return "".join(c for c in w.lower() if c.isalnum())

def map_sentences_to_pdf(sentences: list, pages_data: list) -> list:
    """
    Maps a list of segmented sentences to their page numbers and coordinates in the PDF.
    Returns a list of dicts: [
        {
            "sentence_id": int,
            "text": str,
            "page_number": int or None,
            "bbox": [x0, y0, x1, y1] or None,
            "rects": list of [x0, y0, x1, y1] (for multi-line rendering) or None
        }
    ]
    """
    # Flatten all pdf words into a single list with page and bbox metadata
    pdf_words = []
    for page in pages_data:
        p_num = page["page_number"]
        for word in page["words"]:
            pdf_words.append({
                "text": word["text"],
                "clean": clean_word(word["text"]),
                "page": p_num,
                "bbox": word["bbox"]
            })
            
    mapped_sentences = []
    pdf_word_idx = 0
    total_pdf_words = len(pdf_words)
    
    for sent_idx, sent_text in enumerate(sentences):
        # Tokenize sentence and clean words
        s_words = sent_text.split()
        s_words_clean = [clean_word(w) for w in s_words if clean_word(w)]
        
        if not s_words_clean:
            # Empty sentence or punctuation-only sentence
            mapped_sentences.append({
                "sentence_id": sent_idx,
                "text": sent_text,
                "page_number": None,
                "bbox": None,
                "rects": None
            })
            continue
            
        # Try to find a match in the pdf_words list
        best_start = -1
        best_end = -1
        best_match_count = 0
        
        # Search window starts from current cursor
        search_limit = min(pdf_word_idx + 300, total_pdf_words)
        
        first_word = s_words_clean[0]
        
        # Scan forward to find potential match for the first word
        for i in range(pdf_word_idx, search_limit):
            if pdf_words[i]["clean"] == first_word or (len(first_word) > 3 and first_word in pdf_words[i]["clean"]):
                # Found candidate starting index. Verify match density
                match_count = 0
                check_len = min(len(s_words_clean), total_pdf_words - i)
                
                for j in range(check_len):
                    if pdf_words[i + j]["clean"] == s_words_clean[j] or (len(s_words_clean[j]) > 3 and s_words_clean[j] in pdf_words[i + j]["clean"]):
                        match_count += 1
                        
                if match_count > best_match_count and match_count >= (len(s_words_clean) * 0.4):
                    best_match_count = match_count
                    best_start = i
                    best_end = i + check_len
                    
                # If we get a very high match rate, we stop searching
                if match_count >= (len(s_words_clean) * 0.8):
                    break
                    
        if best_start != -1:
            # We found a match! Extract word bounding boxes
            matched_words = pdf_words[best_start:best_end]
            
            # Group by page (in case the sentence spans page boundaries)
            # Usually a sentence is on a single page, but we'll use the page of the majority of words
            pages_involved = [w["page"] for w in matched_words]
            major_page = max(set(pages_involved), key=pages_involved.count)
            
            # Filter words on this major page
            page_words = [w for w in matched_words if w["page"] == major_page]
            
            if page_words:
                # Compute overall bbox: [min_x0, min_y0, max_x1, max_y1]
                x0 = min(w["bbox"][0] for w in page_words)
                y0 = min(w["bbox"][1] for w in page_words)
                x1 = max(w["bbox"][2] for w in page_words)
                y1 = max(w["bbox"][3] for w in page_words)
                
                # Multi-line rectangle extraction (grouping word bounding boxes by lines)
                # Words on the same line have similar y-coordinates (within 3pt tolerance)
                rects = []
                page_words_sorted = sorted(page_words, key=lambda w: w["bbox"][1])
                
                current_line_words = []
                for w in page_words_sorted:
                    if not current_line_words:
                        current_line_words.append(w)
                    else:
                        # If y-centers align within tolerance, they are on the same line
                        prev_y_center = (current_line_words[-1]["bbox"][1] + current_line_words[-1]["bbox"][3]) / 2.0
                        curr_y_center = (w["bbox"][1] + w["bbox"][3]) / 2.0
                        if abs(curr_y_center - prev_y_center) < 5.0:
                            current_line_words.append(w)
                        else:
                            # Save current line rect
                            lx0 = min(lw["bbox"][0] for lw in current_line_words)
                            ly0 = min(lw["bbox"][1] for lw in current_line_words)
                            lx1 = max(lw["bbox"][2] for lw in current_line_words)
                            ly1 = max(lw["bbox"][3] for lw in current_line_words)
                            rects.append([lx0, ly0, lx1, ly1])
                            current_line_words = [w]
                if current_line_words:
                    lx0 = min(lw["bbox"][0] for lw in current_line_words)
                    ly0 = min(lw["bbox"][1] for lw in current_line_words)
                    lx1 = max(lw["bbox"][2] for lw in current_line_words)
                    ly1 = max(lw["bbox"][3] for lw in current_line_words)
                    rects.append([lx0, ly0, lx1, ly1])
                
                mapped_sentences.append({
                    "sentence_id": sent_idx,
                    "text": sent_text,
                    "page_number": major_page,
                    "bbox": [x0, y0, x1, y1],
                    "rects": rects
                })
                # Move the PDF word cursor forward to the end of the match
                pdf_word_idx = best_end
                continue
                
        # Fallback if no match was found
        mapped_sentences.append({
            "sentence_id": sent_idx,
            "text": sent_text,
            "page_number": None,
            "bbox": None,
            "rects": None
        })
        
    return mapped_sentences
