import fitz  # PyMuPDF

def extract_pdf_pages(pdf_path: str) -> list:
    """
    Extracts text and word-level coordinate information from a PDF file using PyMuPDF.
    Returns a list of dicts: [
        {
            "page_number": int,
            "text": str,
            "words": [
                {"text": str, "bbox": (x0, y0, x1, y1)}
            ]
        }
    ]
    """
    pages_data = []
    
    # Open document using PyMuPDF
    doc = fitz.open(pdf_path)
    
    for page_idx, page in enumerate(doc):
        # Extract word list: list of tuples (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        raw_words = page.get_text("words")
        
        words_data = []
        page_text_parts = []
        
        for w in raw_words:
            x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
            words_data.append({
                "text": word_text,
                "bbox": (x0, y0, x1, y1)
            })
            page_text_parts.append(word_text)
            
        page_text = " ".join(page_text_parts)
        
        pages_data.append({
            "page_number": page_idx + 1,
            "text": page_text,
            "words": words_data
        })
        
    doc.close()
    return pages_data

def is_scanned_pdf(pages_data: list, min_chars_per_page: int = 5) -> bool:
    """
    Detects if the PDF is scanned (requires OCR) by checking if the total extracted character count 
    is below a threshold.
    """
    if not pages_data:
        return True
    
    total_chars = sum(len(page["text"]) for page in pages_data)
    avg_chars = total_chars / len(pages_data)
    
    return avg_chars < min_chars_per_page
