import html
import json
import os

import requests
import streamlit as st
import transformers.utils.logging as hf_logging

from explanation_and_reasoning.cjpe_pipeline import generate_cjpe_explanation
from tldr_uniandes.tldr_uniandes_model import (
    pipeline_a_tldr,
    pipeline_b_reward,
    pipeline_c_multiagent,
)
from rhetorical_integration import classify_judgment_rhetorical_roles
import utils.document_state as doc_state
from utils.document_state import (
    UI_ROLES,
    attach_predictions,
    clear_document_state,
    document_id_from_bytes,
    format_role_content,
    init_session_state,
    records_for_ui_role,
    role_heading,
    summary_cache_key,
)
from utils.pdf_extract import extract_document
from utils.pdf_viewer import render_pdf_viewer

hf_logging.disable_progress_bar()

st.set_page_config(page_title="Legal Judgement Summarizer", layout="wide")

# Initialize session state defaults directly to avoid caching/reload AttributeError issues
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
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as handle:
        st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)

MODELS = [
    "SCaLAR System 3 — Weighted Rhetorical Roles (WRR)",
    "Uniandes Reward-Driven Summary",
    "SAC-LLM — Qwen3 8B",
]

ACTIVE_MODELS = {
    "SCaLAR System 3 — Weighted Rhetorical Roles (WRR)",
    "Uniandes Reward-Driven Summary",
    "SAC-LLM — Qwen3 8B",
}

MODEL_INFO = {
    "SCaLAR System 3 — Weighted Rhetorical Roles (WRR)": "Semantic Chunking and Logical Alignment with Weighted Rhetorical Role tags injected to Pegasus.",
    "Uniandes Reward-Driven Summary": "Best single-agent reward-driven prompt from Uniandes legal summarizer research.",
    "SAC-LLM — Qwen3 8B": "Structure-Aware Chunking with LLM-based segmentation using local Qwen3 8B.",
}

EXPLANATION_SECTIONS = [
    ("facts_of_the_case", "Facts of the Case"),
    ("legal_issues", "Legal Issue(s) Presented"),
    ("applicable_law", "Applicable Law and Precedents"),
    ("analysis_reasoning", "Analysis / Reasoning"),
    ("predicted_conclusion", "Predicted Conclusion"),
]


def load_document(uploaded_file):
    """
    Callback function that reads the uploaded PDF bytes, parses pages, extracts text,
    splits it into sentences, and classifies the rhetorical roles of each sentence.

    It updates the Streamlit session state variables with the extraction results.

    Args:
        uploaded_file (UploadedFile): The streamlit file uploader object.
    """
    pdf_bytes = uploaded_file.getvalue()
    doc_id = document_id_from_bytes(pdf_bytes)
    if doc_id == st.session_state.document_id:
        return

    clear_document_state()
    st.session_state.document_id = doc_id
    st.session_state.filename = uploaded_file.name
    st.session_state.pdf_data = pdf_bytes
    st.session_state.pdf_uploaded = True

    try:
        extracted = extract_document(pdf_bytes)
    except Exception as error:
        st.session_state.rhetorical_error = f"PDF extraction failed: {error}"
        return

    st.session_state.pages = extracted["pages"]
    st.session_state.extracted_text = extracted["extracted_text"]
    st.session_state.sentences = extracted["sentences"]
    st.session_state.sentence_records = extracted["sentence_records"]

    if not st.session_state.sentences:
        st.session_state.rhetorical_error = "No text could be extracted from this PDF."
        return

    # Check if the document has cached predictions
    if doc_id in st.session_state.rhetorical_cache:
        st.session_state.rhetorical_records = st.session_state.rhetorical_cache[doc_id]


def generate_full_summary(model_name):
    """
    Orchestrates summary generation using the selected model, caching results in session state.

    Args:
        model_name (str): The name of the model selected by the user.
    """
    text = st.session_state.extracted_text.strip()
    if not text:
        st.session_state.summary_error = "No text extracted from PDF. Please use a text-based PDF."
        return
    st.session_state.summary_error = ""
    try:
        if model_name.startswith("Uniandes "):
            if model_name == "Uniandes TL;DR Baseline":
                result = pipeline_a_tldr(text)
            elif model_name == "Uniandes Reward-Driven Summary":
                result = pipeline_b_reward(text)
            elif model_name == "Uniandes Multi-Agent Summary":
                result = pipeline_c_multiagent(text)
            else:
                result = "Unsupported Uniandes pipeline"
        elif model_name.startswith("SAC-"):
            from sac_summarizer_module import generate_legal_summary
            use_llm = "SAC-LLM" in model_name
            backend = "pegasus" if "Legal-Pegasus" in model_name else "qwen3:8b"
            result = generate_legal_summary(
                document_text=text,
                model_name=backend,
                use_llm_segmentation=use_llm
            )
        elif model_name.startswith("SCaLAR "):
            from scalar_wrr.scalar_wrr_model import (
                system1_naive,
                system2_rhetorical,
                system3_weighted
            )
            if "System 1" in model_name:
                result = system1_naive(text)
            elif "System 2" in model_name:
                result, _ = system2_rhetorical(text)
            elif "System 3" in model_name:
                # Use fixed weights directly since they are not interactive
                weights = {
                    'facts': 9.48,
                    'issue': 0.60,
                    'arguments': (2.09 + 0.73) / 2.0,
                    'statute': 4.93,
                    'precedent': 4.93,
                    'ratio': 5.61,
                    'ruling': 4.77
                }
                result = system3_weighted(text, weights)
            else:
                result = "Unsupported SCaLAR system"
        else:
            result = f"Model {model_name} is currently unsupported."
        cache_key = summary_cache_key(st.session_state.document_id, model_name)
        st.session_state.summary_cache[cache_key] = result
    except requests.exceptions.ConnectionError:
        st.session_state.summary_error = "Local Ollama service is not running on port 11434."
    except Exception as error:
        st.session_state.summary_error = f"Summary generation failed: {error}"


def run_cjpe_pipeline_ui():
    """
    Executes the Court Judgment Prediction and Explanation (CJPE) analysis pipeline,
    updating session state fields with prediction and explanation results.
    """
    st.session_state.cjpe_error = ""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "CJPE_XLNet"))
    required_files = ["pytorch_model.bin", "config.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            st.session_state.cjpe_error = f"Missing required CJPE model artifact: {f}"
            return

    try:
        result = generate_cjpe_explanation(
            st.session_state.extracted_text,
            sentence_metadata=st.session_state.sentence_records
        )
        
        st.session_state.cjpe_prediction = result["prediction"]
        st.session_state.cjpe_explanation_passages = result["explanation_passages"]
        st.session_state.cjpe_document_id = st.session_state.document_id
        
        alignments = []
        for passage in result["explanation_passages"]:
            if passage.get("page_number") is not None:
                alignments.append({
                    "sentence": passage["text"],
                    "page_number": passage["page_number"],
                    "sentence_id": passage["sentence_ids"][0] if passage["sentence_ids"] else ""
                })
        st.session_state.cjpe_alignment_results = alignments
        
        for passage in result["explanation_passages"]:
            if passage.get("page_number") is not None:
                st.session_state.selected_page = passage["page_number"]
                st.session_state.selected_explanation_section = passage.get("sentence_index")
                break
        
    except Exception as error:
        st.session_state.cjpe_error = f"CJPE pipeline analysis failed: {error}"


def pdf_highlight_records(active_view, active_role):
    """
    Determines which sentence records should be highlighted in the PDF viewer based on active view and role.

    Args:
        active_view (str): The current view ("ROLE", "EXPLANATION", or "FULL SUMMARY").
        active_role (str): The selected UI rhetorical role.

    Returns:
        list[dict]: A list of sentence records to be highlighted in the PDF.
    """
    if active_view == "EXPLANATION":
        records = []
        for passage in st.session_state.get("cjpe_explanation_passages", []):
            if passage.get("page_number") is not None:
                sent_idx = passage.get("sentence_index")
                if 0 <= sent_idx < len(st.session_state.get("sentence_records", [])):
                    records.append(st.session_state.sentence_records[sent_idx])
        return records
    if active_view == "ROLE" and active_role:
        return records_for_ui_role(st.session_state.rhetorical_records, active_role)
    return []


def render_cjpe_explanation_panel():
    """
    Renders the UI panels for CJPE predictions and occlusion-based explanation passages.
    """
    if st.session_state.cjpe_error:
        st.error(st.session_state.cjpe_error)
        return

    prediction = st.session_state.cjpe_prediction
    passages = st.session_state.cjpe_explanation_passages

    if not prediction or st.session_state.cjpe_document_id != st.session_state.document_id:
        return

    st.markdown(
        f'<div class="prediction-box">Prediction: <strong>{html.escape(prediction["label"])}</strong> (Confidence: {prediction["confidence"]:.2%})</div>',
        unsafe_allow_html=True,
    )

    # Add a PDF download button for CJPE explanation report
    from utils.pdf_generator import generate_explanation_pdf
    formatted_passages = []
    for p in passages:
        role = "None"
        for rec in st.session_state.rhetorical_records:
            if rec.get("sentence") == p["text"]:
                role = rec.get("role") or "None"
                break
        formatted_passages.append({
            "page_number": p.get("page_number"),
            "role": role,
            "importance_score": p["importance_score"],
            "sentence": p["text"]
        })
    pdf_bytes = generate_explanation_pdf(
        st.session_state.document_id,
        st.session_state.filename,
        f"{prediction['label']} (Confidence: {prediction['confidence']:.2%})",
        formatted_passages
    )
    st.download_button(
        label="Download Explanation Report as PDF",
        data=pdf_bytes,
        file_name=f"cjpe_explanation_{st.session_state.document_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.subheader("Why the Model Reached This Prediction")
    st.markdown("<p style='font-size: 13px; color: #64748b; margin-top: -10px; margin-bottom: 15px;'>These passages most influenced the model\'s prediction according to the CJPE hierarchical occlusion method.</p>", unsafe_allow_html=True)

    if not passages:
        st.write("No influential passages identified.")
        return

    for idx, passage in enumerate(passages):
        rank = passage["rank"]
        page_num = passage.get("page_number")
        text = passage["text"]
        score = passage["importance_score"]
        
        rank_words = ["Most", "Second Most", "Third Most", "Fourth Most", "Fifth Most"]
        rank_str = rank_words[idx] if idx < len(rank_words) else f"{rank}th Most"
        title_str = f"**{rank_str} Influential Passage — Page {page_num if page_num is not None else 'N/A'}**"
        
        btn_key = f"cjpe_passage_{idx}_{passage.get('sentence_index')}"
        
        with st.container(border=True):
            st.markdown(title_str)
            st.write(text)
            st.markdown(f"<span style='color: #475569; font-size: 12px; font-weight: bold;'>Importance: {score:.4f}</span>", unsafe_allow_html=True)
            if page_num is not None:
                if st.button("🔍 Locate in PDF", key=btn_key, use_container_width=True):
                    st.session_state.selected_page = page_num
                    st.session_state.selected_explanation_section = passage.get("sentence_index")
                    st.rerun()


def format_summary_output(cached):
    if isinstance(cached, dict):
        text_parts = []
        if "final_summary" in cached:
            text_parts.append(f"COMBINED SUMMARY:\n{cached['final_summary']}")
        if "section_summaries" in cached:
            sec_sums = cached["section_summaries"]
            text_parts.append(
                f"SECTION SUMMARIES:\n"
                f"- Facts: {sec_sums.get('facts', '')}\n"
                f"- Analysis & Arguments: {sec_sums.get('analysis', '')}\n"
                f"- Conclusion: {sec_sums.get('conclusion', '')}"
            )
        if "sentence_counts" in cached and "chunk_counts" in cached:
            text_parts.append(
                f"STATISTICS:\n"
                f"- Facts: {cached['sentence_counts'].get('facts', 0)} sentences, {cached['chunk_counts'].get('facts', 0)} chunks\n"
                f"- Analysis & Arguments: {cached['sentence_counts'].get('analysis', 0)} sentences, {cached['chunk_counts'].get('analysis', 0)} chunks\n"
                f"- Conclusion: {cached['sentence_counts'].get('conclusion', 0)} sentences, {cached['chunk_counts'].get('conclusion', 0)} chunks"
            )
        return "\n\n".join(text_parts)
    return cached


def right_panel_content(has_document, active_view, active_role, picked):
    """
    Calculates the title and main content text for display in the right panel.

    Args:
        has_document (bool): True if a document is uploaded.
        active_view (str): The current active view configuration.
        active_role (str): The current active rhetorical role.
        picked (str): The chosen summary model.

    Returns:
        tuple[str, str]: A tuple of (panel heading, panel body content).
    """
    if not has_document:
        return "Analysis", "No PDF uploaded"

    # Handle the state after PDF upload but before rhetorical analysis
    if not st.session_state.rhetorical_records:
        if active_view == "FULL SUMMARY":
            if picked not in ACTIVE_MODELS:
                return "Generated Summary", MODEL_INFO[picked]
            if st.session_state.summary_error:
                return "Generated Summary", st.session_state.summary_error
            cached = st.session_state.summary_cache.get(
                summary_cache_key(st.session_state.document_id, picked),
                "",
            )
            if cached:
                return "Generated Summary", format_summary_output(cached)
            return "Generated Summary", "Click Generate Summary to create a summary using the selected model."
        else:
            return "Analysis", "Run rhetorical analysis or choose another available action."

    if active_view == "EXPLANATION":
        return "Explanation & Reasoning", ""

    if active_view == "FULL SUMMARY":
        if picked not in ACTIVE_MODELS:
            return "Generated Summary", MODEL_INFO[picked]
        if st.session_state.summary_error:
            return "Generated Summary", st.session_state.summary_error
        cached = st.session_state.summary_cache.get(
            summary_cache_key(st.session_state.document_id, picked),
            "",
        )
        if cached:
            return "Generated Summary", format_summary_output(cached)
        return "Generated Summary", "Click Generate Summary to create a summary using the selected model."

    if not active_role:
        return "Analysis", "Select a rhetorical role, full summary, or explanation & reasoning."

    role_records = records_for_ui_role(st.session_state.rhetorical_records, active_role)
    return role_heading(active_role), format_role_content(role_records, active_role)


init_session_state()

st.markdown("""
<div class="page-header">
    <h1>Legal Judgement Summarizer</h1>
</div>
""", unsafe_allow_html=True)

left, mid, right = st.columns([2.18, 0.47, 2.35])
has_document = bool(st.session_state.document_id and st.session_state.pdf_data)
active_view = st.session_state.selected_view
active_role = st.session_state.selected_role
buttons_disabled = not has_document or not st.session_state.rhetorical_records

with left:
    st.subheader("Document Viewer")
    if not has_document:
        st.markdown('<div class="empty-state">No PDF uploaded</div>', unsafe_allow_html=True)
    else:
        role_records = pdf_highlight_records(active_view, active_role)
        if role_records and active_view != "EXPLANATION":
            st.session_state.selected_page = role_records[0]["page_number"]
        render_pdf_viewer(
            st.session_state.pdf_data,
            st.session_state.selected_page,
            role_records,
            height=753,
        )

with mid:
    st.markdown('<div class="role-panel"></div>', unsafe_allow_html=True)
    st.subheader("Rhetorical Roles")
    if st.session_state.rhetorical_error:
        st.caption(st.session_state.rhetorical_error)

    if has_document and not st.session_state.rhetorical_records:
        if st.button("Run Rhetorical Analysis", type="primary", use_container_width=True):
            with st.spinner("Running rhetorical role analysis..."):
                try:
                    predictions = classify_judgment_rhetorical_roles(
                        st.session_state.sentences,
                        sentence_metadata=st.session_state.sentence_records
                    )
                    st.session_state.rhetorical_records = predictions
                    st.session_state.rhetorical_cache[st.session_state.document_id] = predictions
                    
                    # Auto-save to outputs folder on disk
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"rhetorical_roles_{st.session_state.document_id}.json")
                    with open(output_path, "w", encoding="utf-8") as out_f:
                        json.dump(predictions, out_f, indent=4)
                        
                    st.rerun()
                except Exception as error:
                    st.session_state.rhetorical_error = f"Rhetorical classification failed: {error}"
                    st.session_state.rhetorical_records = []
                    st.rerun()

    for name in UI_ROLES:
        kind = "primary" if active_role == name and active_view == "ROLE" else "secondary"
        if st.button(name, key=name, type=kind, use_container_width=True, disabled=buttons_disabled):
            st.session_state.selected_role = name
            st.session_state.selected_view = "ROLE"
            st.session_state.selected_explanation_section = None
            matches = records_for_ui_role(st.session_state.rhetorical_records, name)
            st.session_state.selected_page = matches[0]["page_number"] if matches else 1
            st.rerun()

    st.write("---")
    full_kind = "primary" if active_view == "FULL SUMMARY" else "secondary"
    if st.button("FULL SUMMARY", key="FULL SUMMARY", type=full_kind, use_container_width=True, disabled=not has_document):
        st.session_state.selected_view = "FULL SUMMARY"
        st.session_state.selected_role = None
        st.session_state.selected_explanation_section = None
        st.rerun()

    exp_kind = "primary" if active_view == "EXPLANATION" else "secondary"
    if st.button("EXPLANATION & REASONING", key="EXPLANATION", type=exp_kind, use_container_width=True, disabled=buttons_disabled):
        st.session_state.selected_view = "EXPLANATION"
        st.session_state.selected_role = None
        passages = st.session_state.get("cjpe_explanation_passages", [])
        for p in passages:
            if p.get("page_number") is not None:
                st.session_state.selected_page = p["page_number"]
                st.session_state.selected_explanation_section = p.get("sentence_index")
                break
        st.rerun()

    if st.session_state.rhetorical_records:
        st.write("---")
        
        # PDF Download Button
        from utils.pdf_generator import generate_analysis_pdf
        
        # Load predictions from the saved JSON file on disk
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        output_path = os.path.join(output_dir, f"rhetorical_roles_{st.session_state.document_id}.json")
        try:
            with open(output_path, "r", encoding="utf-8") as in_f:
                loaded_records = json.load(in_f)
        except Exception:
            # Fallback to session state if reading from disk fails
            loaded_records = st.session_state.rhetorical_records

        pdf_data = generate_analysis_pdf(loaded_records, st.session_state.document_id)
        st.download_button(
            label="Download Predictions PDF",
            data=pdf_data,
            file_name=f"rhetorical_roles_{st.session_state.document_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    dynamic_css = "<style>\n"
    for idx, name in enumerate(UI_ROLES):
        count = len(records_for_ui_role(st.session_state.rhetorical_records, name))
        child_idx = 5 + idx
        dynamic_css += (
            f'div[data-testid="column"]:has(.role-panel) '
            f'div.element-container:nth-child({child_idx}) button::after '
            f'{{ content: "{count}" !important; display: flex !important; }}\n'
        )
    for child_idx in range(12, 20):
        dynamic_css += (
            f'div[data-testid="column"]:has(.role-panel) '
            f'div.element-container:nth-child({child_idx}) button::after '
            f'{{ display: none !important; }}\n'
        )
    dynamic_css += "</style>"
    st.markdown(dynamic_css, unsafe_allow_html=True)

with right:
    if active_view == "EXPLANATION":
        if has_document and not buttons_disabled:
            if st.button("Generate Explanation & Reasoning", type="primary", use_container_width=True):
                with st.spinner("Running local CJPE explanation analysis..."):
                    run_cjpe_pipeline_ui()
                st.rerun()
        heading, content = right_panel_content(has_document, active_view, active_role, "")
        st.subheader(heading)
        if not has_document:
            st.info("No PDF uploaded. Upload a legal judgment to generate an explanation.")
        else:
            render_cjpe_explanation_panel()
    else:
        # Check if the summary exists for the currently selected model
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = MODELS[0]
            
        picked = st.session_state.selected_model
        cache_key = summary_cache_key(st.session_state.document_id, picked)
        has_summary = cache_key in st.session_state.summary_cache

        # The selectbox should only be visible in FULL SUMMARY view AND when the summary has not been generated yet
        show_selector = (active_view == "FULL SUMMARY") and (not has_summary)

        if show_selector:
            picked = st.selectbox("Select Model", MODELS, index=MODELS.index(st.session_state.selected_model), key="model_dropdown")
            st.session_state.selected_model = picked
            st.caption(MODEL_INFO[picked])
        elif active_view == "FULL SUMMARY" and has_summary:
            st.write(f"### Selected Model: {picked}")
            st.caption(MODEL_INFO[picked])



        if active_view == "FULL SUMMARY" and has_document and picked in ACTIVE_MODELS and not has_summary:
            if st.button("Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    generate_full_summary(picked)
                st.rerun()

        # Show Change Model and Download Summary PDF buttons if summary was generated and we are in FULL SUMMARY view
        if active_view == "FULL SUMMARY" and has_summary:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Change Model / Regenerate Summary", type="secondary", use_container_width=True):
                    if cache_key in st.session_state.summary_cache:
                        del st.session_state.summary_cache[cache_key]
                    st.rerun()
            with col2:
                from utils.pdf_generator import generate_summary_pdf
                summary_data = generate_summary_pdf(
                    st.session_state.document_id,
                    st.session_state.filename,
                    picked,
                    st.session_state.summary_cache[cache_key]
                )
                st.download_button(
                    label="Download Summary PDF",
                    data=summary_data,
                    file_name=f"summary_{st.session_state.document_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        if not has_document:
            st.info("Upload a PDF to begin.")
        else:
            heading, content = right_panel_content(has_document, active_view, active_role, picked)
            
            show_content = True
            if active_view == "FULL SUMMARY":
                show_content = has_summary
            elif active_view == "ROLE":
                show_content = len(st.session_state.rhetorical_records) > 0
                if not show_content:
                    st.info("Rhetorical Role Analysis has not been run yet. Please click 'Run Rhetorical Analysis' first.")

            if show_content:
                st.subheader(heading)
                if active_view == "ROLE":
                    safe_content = content
                else:
                    safe_content = html.escape(content)

                st.markdown(f"""
<div class="summary-container">
    <div class="summary-box" style="white-space:pre-wrap;">{safe_content}</div>
    <button class="copy-btn" type="button">Copy</button>
</div>
<script>
(function() {{
    var content = {json.dumps(content)};
    var scriptTag = document.currentScript;
    if (!scriptTag || !scriptTag.previousElementSibling) return;
    var container = scriptTag.previousElementSibling;
    var btn = container.querySelector(".copy-btn");
    if (btn) {{
        btn.addEventListener("click", function() {{
            var plainText = content;
            if ({json.dumps(active_view == "ROLE")}) {{
                var tempDiv = document.createElement("div");
                tempDiv.innerHTML = content;
                plainText = tempDiv.innerText || tempDiv.textContent || "";
            }}
            navigator.clipboard.writeText(plainText).then(function() {{
                btn.innerText = "Copied!";
                setTimeout(function() {{ btn.innerText = "Copy"; }}, 1500);
            }});
        }});
    }}
    var hoverSpans = container.querySelectorAll(".hover-sentence");
    hoverSpans.forEach(function(span) {{
        span.addEventListener("mouseenter", function() {{
            span.style.backgroundColor = "rgba(254, 240, 138, 0.7)";
            span.style.cursor = "pointer";
            var rawText = span.textContent || span.innerText || "";
            var page = parseInt(span.getAttribute("data-page"), 10);
            var iframes = [];
            try {{
                iframes = iframes.concat(Array.from(document.querySelectorAll("iframe")));
            }} catch(e) {{}}
            try {{
                iframes = iframes.concat(Array.from(window.parent.document.querySelectorAll("iframe")));
            }} catch(e) {{}}
            iframes = Array.from(new Set(iframes));
            iframes.forEach(function(iframe) {{
                try {{
                    iframe.contentWindow.postMessage({{
                        type: "HIGHLIGHT_SENTENCE",
                        text: rawText,
                        page: page
                    }}, "*");
                }} catch(e) {{}}
            }});
        }});
        span.addEventListener("mouseleave", function() {{
            span.style.backgroundColor = "transparent";
        }});
    }});
}})();
</script>
""", unsafe_allow_html=True)

    if has_document:
        st.subheader("Document Information")
        try:
            from rhetorical_integration import load_rhetorical_resources
            _, _, _, device = load_rhetorical_resources()
            device_str = "CUDA" if "cuda" in device.lower() else "CPU"
        except Exception:
            device_str = "CPU"

        st.markdown(f"""
<div style="border:1px solid #e2e8f0;border-radius:8px;padding:16px;background:white;">
    <div class="info-row"><span>File Name</span><span>{st.session_state.filename}</span></div>
    <div class="info-row"><span>Total Pages</span><span>{len(st.session_state.pages)}</span></div>
    <div class="info-row"><span>Total Sentences</span><span>{len(st.session_state.sentences)}</span></div>
    <div class="info-row"><span>Inference Device</span><span>{device_str}</span></div>
    <div class="info-row" style="border:none"><span>Document ID</span><span>{st.session_state.document_id[:12]}...</span></div>
</div>
""", unsafe_allow_html=True)

upload_label = "Upload again" if st.session_state.pdf_uploaded else "Upload"
with st.container(border=True):
    st.markdown('<div class="upload-panel"></div>', unsafe_allow_html=True)
    st.markdown("**Upload New Document**")
    uploaded = st.file_uploader(upload_label, type=["pdf"], help="200MB per file • PDF", key="pdf_uploader")
    if uploaded is not None:
        previous_id = st.session_state.document_id
        load_document(uploaded)
        if st.session_state.document_id != previous_id:
            st.rerun()
    if has_document and st.button("Remove PDF", key="remove_pdf"):
        clear_document_state()
        st.rerun()
