import fitz  # PyMuPDF
from backend.rhetorical.labels import get_role_highlight_color

def hex_to_rgb(hex_str: str) -> tuple:
    """Converts a hex color string (e.g. '#2563EB') to an RGB tuple (r, g, b) between 0.0 and 1.0."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def highlight_pdf_by_role(input_pdf_path: str, output_pdf_path: str, sentences: list, active_role: str = "All") -> None:
    """
    Reads the input PDF, adds translucent highlight annotations for sentences matching 
    the active_role, and writes the annotated PDF to output_pdf_path.
    
    If active_role == "All", all analyzed sentences are highlighted using their respective role colors.
    If active_role is a specific role (e.g. "Facts"), only sentences matching that role are highlighted.
    """
    doc = fitz.open(input_pdf_path)
    
    for sent in sentences:
        # Check if the sentence has mapping metadata
        if not sent.page_number or not sent.rects:
            continue
            
        # Determine if we should highlight this sentence based on the active role filter
        if active_role != "All" and sent.role != active_role:
            continue
            
        # Get RGB color representation of the role's color
        color_hex = get_role_highlight_color(sent.role)
        color_rgb = hex_to_rgb(color_hex)
        
        # PyMuPDF uses 0-based page indices, while page_number is 1-based
        page_idx = sent.page_number - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue
            
        page = doc[page_idx]
        
        # Add highlight annotations for each line rectangle of the sentence
        for rect_coords in sent.rects:
            try:
                rect = fitz.Rect(rect_coords)
                # Verify rect is valid (non-empty area)
                if rect.is_empty:
                    continue
                annot = page.add_highlight_annot(rect)
                if annot:
                    annot.set_colors(stroke=color_rgb)
                    annot.update()
            except Exception as e:
                # Silently catch minor annotation rendering errors to avoid breaking the pipeline
                print(f"Annotation warning for sentence {sent.sentence_id}: {e}")
                
    doc.save(output_pdf_path)
    doc.close()
