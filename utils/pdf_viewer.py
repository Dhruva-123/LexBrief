import base64
import json
import os
import streamlit.components.v1 as components


def render_pdf_viewer(pdf_bytes, page_number, highlight_records, height=600):
    """
    Renders a custom HTML-based PDF viewer inside the Streamlit application.

    This function reads a template HTML file, serializes the PDF file content as Base64,
    packs the current active page number and sentences to highlight into a configuration dictionary,
    and replaces the configuration placeholder in the template. The resulting interactive HTML component
    is rendered via Streamlit.

    Args:
        pdf_bytes (bytes): Binary data of the PDF file.
        page_number (int): The current active page number to display first.
        highlight_records (list[dict]): List of sentence records that should be highlighted.
        height (int, optional): Height of the iframe in pixels. Defaults to 600.
    """
    viewer_path = os.path.join("assets", "pdf_viewer.html")
    with open(viewer_path, "r", encoding="utf-8") as handle:
        template = handle.read()

    highlights = [
        {"page": record["page_number"], "text": record["sentence"]}
        for record in highlight_records
    ]
    config = {
        "pdfBase64": base64.b64encode(pdf_bytes).decode("utf-8"),
        "page": page_number,
        "highlights": highlights,
    }
    html = template.replace("__CONFIG__", json.dumps(config))
    components.html(html, height=height, scrolling=True)
