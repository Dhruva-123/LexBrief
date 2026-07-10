import io
from pypdf import PdfReader
from utils.sentence_splitter import split_sentences


def extract_pages(pdf_bytes):
    """
    Parses a PDF from a byte stream and extracts text from each page.

    Args:
        pdf_bytes (bytes): Binary content of the uploaded PDF file.

    Returns:
        list[dict]: A list of dictionaries, each containing 'page_number' and extracted 'text'.
    """
    if not pdf_bytes:
        return []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append({
            "page_number": page_number,
            "text": page.extract_text() or "",
        })
    return pages


def build_sentence_records(pages):
    """
    Splits page texts into individual sentences and builds structured records with indices.

    Args:
        pages (list[dict]): A list of page dictionaries with extracted text.

    Returns:
        list[dict]: A list of structured sentence records containing global and page-specific indices.
    """
    records = []
    document_index = 0
    for page in pages:
        page_number = page["page_number"]
        for page_index, sentence in enumerate(split_sentences(page["text"])):
            records.append({
                "sentence_id": f"p{page_number}_s{page_index}",
                "sentence": sentence,
                "page_number": page_number,
                "page_sentence_index": page_index,
                "document_sentence_index": document_index,
            })
            document_index += 1
    return records


def extract_document(pdf_bytes):
    """
    Orchestrates the entire document parsing workflow from raw bytes to structured sentences.

    Args:
        pdf_bytes (bytes): Binary content of the uploaded PDF file.

    Returns:
        dict: A dictionary containing the pages list, flat sentence list, structured sentence records, and full extracted text.
    """
    pages = extract_pages(pdf_bytes)
    records = build_sentence_records(pages)
    extracted_text = "\n".join(page["text"] for page in pages if page["text"])
    return {
        "pages": pages,
        "sentences": [record["sentence"] for record in records],
        "sentence_records": records,
        "extracted_text": extracted_text,
    }
