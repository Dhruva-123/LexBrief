import hashlib
import streamlit as st


OFFICIAL_TO_UI = {
    "Facts": "FACTS",
    "Issue": "ISSUE",
    "Arguments of Petitioner": "ARGUMENTS OF PETITIONER",
    "Arguments of Respondent": "ARGUMENTS OF RESPONDENT",
    "Reasoning": "REASONING",
    "Decision": "DECISION",
    "None": "NONE",
}
UI_TO_OFFICIAL = {value: key for key, value in OFFICIAL_TO_UI.items()}

UI_ROLES = list(UI_TO_OFFICIAL.keys())

DOC_STATE_KEYS = [
    "document_id",
    "filename",
    "pdf_data",
    "pdf_uploaded",
    "pages",
    "extracted_text",
    "sentences",
    "sentence_records",
    "rhetorical_records",
    "selected_role",
    "selected_page",
    "selected_view",
    "selected_explanation_section",
    "summary_cache",
    "summary_error",
    "rhetorical_error",
    "highlight_align_errors",
    "explanation_result",
    "explanation_error",
    "explanation_document_id",
    "cjpe_document_id",
    "cjpe_prediction",
    "cjpe_explanation_passages",
    "cjpe_alignment_results",
    "cjpe_error",
]


def document_id_from_bytes(pdf_bytes):
    """
    Computes a unique SHA-256 hash representation of PDF bytes to use as a document ID.

    Args:
        pdf_bytes (bytes): The raw bytes of the PDF document.

    Returns:
        str: Hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(pdf_bytes).hexdigest()


def init_session_state():
    """
    Initializes standard UI-related variables in Streamlit's session state if they do not exist.

    This ensures that default states (e.g., page numbers, active selections, errors, predictions,
    caches) are safely set up before Streamlit renders the page, preventing unexpected AttributeError issues.
    """
    defaults = {
        "document_id": None,
        "filename": "",
        "pdf_data": None,
        "pdf_uploaded": False,
        "pages": [],
        "extracted_text": "",
        "sentences": [],
        "sentence_records": [],
        "rhetorical_records": [],
        "rhetorical_cache": {},
        "selected_role": None,
        "selected_page": 1,
        "selected_view": None,
        "selected_explanation_section": None,
        "summary_cache": {},
        "summary_error": "",
        "rhetorical_error": "",
        "highlight_align_errors": [],
        "explanation_result": None,
        "explanation_error": "",
        "explanation_document_id": None,
        "cjpe_document_id": None,
        "cjpe_prediction": None,
        "cjpe_explanation_passages": [],
        "cjpe_alignment_results": [],
        "cjpe_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_document_state():
    """
    Clears all document-specific Streamlit session state fields to prepare for a new document upload.
    """
    st.session_state.document_id = None
    st.session_state.filename = ""
    st.session_state.pdf_data = None
    st.session_state.pdf_uploaded = False
    st.session_state.pages = []
    st.session_state.extracted_text = ""
    st.session_state.sentences = []
    st.session_state.sentence_records = []
    st.session_state.rhetorical_records = []
    st.session_state.selected_role = None
    st.session_state.selected_page = 1
    st.session_state.selected_view = None
    st.session_state.selected_explanation_section = None
    st.session_state.summary_cache = {}
    st.session_state.summary_error = ""
    st.session_state.rhetorical_error = ""
    st.session_state.highlight_align_errors = []
    st.session_state.explanation_result = None
    st.session_state.explanation_error = ""
    st.session_state.explanation_document_id = None
    st.session_state.cjpe_document_id = None
    st.session_state.cjpe_prediction = None
    st.session_state.cjpe_explanation_passages = []
    st.session_state.cjpe_alignment_results = []
    st.session_state.cjpe_error = ""


def records_for_explanation_section(rhetorical_records, section_key):
    """
    Filters and retrieves sentence records corresponding to a given prediction explanation section.

    It maps explanation sections (like facts, reasoning, decision) to their expected rhetorical roles,
    filters the records, removes duplicates, and sorts them by their original order in the document.

    Args:
        rhetorical_records (list[dict]): List of classified sentence records.
        section_key (str): The configuration key name of the explanation section.

    Returns:
        list[dict]: A sorted list of unique sentence records belonging to that section.
    """
    from explanation_and_reasoning.utils import SECTION_ROLE_MAP
    ui_roles = SECTION_ROLE_MAP.get(section_key, [])
    records = []
    seen = set()
    for ui_role in ui_roles:
        for record in records_for_ui_role(rhetorical_records, ui_role):
            sentence_id = record.get("sentence_id")
            if sentence_id in seen:
                continue
            seen.add(sentence_id)
            records.append(record)
    return sorted(records, key=lambda item: item["document_sentence_index"])


def attach_predictions(sentence_records, predictions):
    """
    Merges model role classification predictions back into their corresponding sentence records.

    Args:
        sentence_records (list[dict]): Structured records for the document sentences.
        predictions (list[dict]): Rhetorical role classification outputs from the model.

    Returns:
        list[dict]: Merged records with 'role' and 'ui_role' keys included.

    Raises:
        ValueError: If predictions count doesn't match sentence records count.
    """
    if len(sentence_records) != len(predictions):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match sentence count ({len(sentence_records)})."
        )
    records = []
    for source, prediction in zip(sentence_records, predictions):
        records.append({
            **source,
            "role": prediction["role"],
            "ui_role": OFFICIAL_TO_UI.get(prediction["role"], "NONE"),
        })
    return records


def records_for_ui_role(rhetorical_records, ui_role):
    """
    Filters sentence records to return only those belonging to a specific UI role.

    Args:
        rhetorical_records (list[dict]): Merged sentence records.
        ui_role (str): UI representation role string (e.g. 'FACTS', 'REASONING').

    Returns:
        list[dict]: A filtered list containing only records matching the official role.
    """
    official = UI_TO_OFFICIAL.get(ui_role)
    if not official:
        return []
    return [
        record for record in rhetorical_records
        if record["role"] == official
    ]


def role_heading(ui_role):
    """
    Returns a human-readable title label for a given UI-mapped rhetorical role.

    Args:
        ui_role (str): The UI-mapped role string.

    Returns:
        str: Friendly display title (e.g. 'Facts', 'Arguments of Petitioner').
    """
    labels = {
        "FACTS": "Facts",
        "ISSUE": "Issue",
        "ARGUMENTS OF PETITIONER": "Arguments of Petitioner",
        "ARGUMENTS OF RESPONDENT": "Arguments of Respondent",
        "REASONING": "Reasoning",
        "DECISION": "Decision",
        "NONE": "None",
    }
    return labels.get(ui_role, "Analysis")


def format_role_content(records, ui_role):
    """
    Groups sentence records and formats them as HTML spans with hover metadata.

    Args:
        records (list[dict]): Filtered sentence records.
        ui_role (str): UI role for which content is formatted.

    Returns:
        str: A formatted HTML block of sentences.
    """
    if not records:
        return f"No sentences were classified as {role_heading(ui_role)} in this judgment."
    
    import html
    import re
    sorted_records = sorted(records, key=lambda item: item["document_sentence_index"])
    spans = []
    is_first = True
    for rec in sorted_records:
        txt = rec.get("sentence", "")
        # Clean page headers/footers (e.g. "Page 3 3", "Page 13 13 14", "Page 1")
        cleaned_txt = re.sub(r'\b[Pp]age\s+\d+(\s+\d+)*\b', '', txt)
        cleaned_txt = re.sub(r'\s+', ' ', cleaned_txt).strip()
        if not cleaned_txt:
            continue
        escaped_txt = html.escape(cleaned_txt)
        page_num = rec.get("page_number", 1)
        
        # Check if sentence starts with paragraph indicators: e.g. "1.", "2)", "(a)", "•"
        starts_paragraph = False
        if not is_first:
            if re.match(r'^\s*([\({\[\-\u2022\u25e6]*\d+[\)}\]\.\-]*|[\(\[\{]?[a-zA-Z][\)}\]\.\-]|[\(\[\{]?[ivxIVX]+[\)}\]\.\-|[\u2022\u25e6\u2013\u2014\-])', cleaned_txt):
                starts_paragraph = True
                
        span_html = f'<span class="hover-sentence" data-page="{page_num}" data-text="{escaped_txt}">{escaped_txt}</span>'
        
        if is_first:
            spans.append(span_html)
            is_first = False
        else:
            if starts_paragraph:
                spans.append(f'<br><br>{span_html}')
            else:
                spans.append(f'<br>{span_html}')
        
    return "".join(spans)


def summary_cache_key(document_id, model_name):
    """
    Generates a cache key combining document ID and model name for caching summary runs.

    Args:
        document_id (str): SHA-256 ID of the document.
        model_name (str): Identifier of the active summarization model.

    Returns:
        str: A composite cache key string.
    """
    return f"{document_id}:{model_name}"
