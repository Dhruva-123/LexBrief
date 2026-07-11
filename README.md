
# Legal Rhetorical Analyzer

An interactive Streamlit-based intelligence dashboard for segmenting, summarizing, and explaining court judgments. The system integrates state-of-the-art NLP models and architectures to assist legal professionals, researchers, and students in extracting key information, analyzing rhetorical structures, and understanding machine predictions.

---

## Key Capabilities

### 1. Document Ingestion & Interactive PDF Viewer
* Supports uploading standard legal judgment PDFs up to 200MB.
* Parses and extracts text, sentences, pages, and bounding box positions using custom utilities.
* Interactive, dual-pane layout: **Document Viewer** on the left, **Analysis Panels** on the right.
* Highlight mapping: selecting a rhetorical role or explanation section automatically focuses and highlights the corresponding text directly in the PDF viewer.

### 2. Sentence-level Rhetorical Role Classification
* Employs a custom-trained **Hierarchical BiLSTM-CRF** model (`L-NLProc/LegalSeg_Hier_BiLSTM-CRF`).
* Classifies each sentence into one of 10 legal roles, mapped to the following UI categories:
  * **Facts**: Background facts, parties, events leading to the dispute.
  * **Issue**: The specific legal questions framed for determination.
  * **Arguments**: Contentions and submissions made by counsels of either party.
  * **Statute**: Sections, rules, articles, or other statutory provisions.
  * **Precedent**: Case citations and references to past rulings.
  * **Ratio**: The core legal reasoning and binding principle (ratio decidendi).
  * **Ruling**: Final orders, directions, or results pronounced by the court.

### 3. Summarization Architectures
The system features three distinct legal summarization strategies:
* **SCaLAR System 3 (Weighted Rhetorical Roles - WRR)**:
  * Uses a local fine-tuned **Pegasus** model (`models/legal-pegasus`).
  * Features three configurations: Naive recursive chunking, Rhetorical-based chunking, and Weighted tagging (emphasizing specific roles like Facts or Statutes by injecting weighted tags like `[FACTS:9.48]`).
* **Uniandes Reward-Driven Summary**:
  * Leverages a local **Qwen3:8b** model via the Ollama API.
  * Features three pipelines: Baseline TL;DR, Reward-Driven (optimized prompt based on legal summarization criteria such as span fidelity and structure), and a Multi-Agent pipeline (10 extraction agents + 1 synthesis agent).
* **Structure-Aware Chunking (SAC)**:
  * Utilizes either Qwen3:8b or local Legal-Pegasus.
  * Employs boundary segmentation (regex heuristic SAC-H or zero-shot LLM SAC-LLM) to segment documents into *Facts*, *Arguments & Analysis*, and *Conclusion*.
  * Summarizes each segment using sliding windows with context overlap (SAC-H+) under a Proportional Budget Allocation (PBA) algorithm.

### 4. Court Judgment Prediction & Explanation (CJPE)
* Employs a local **XLNet** sequence classifier (`models/CJPE_XLNet`).
* Predicts the outcome of the judgment (e.g., `"Appeal Allowed"` or `"Appeal Dismissed"`) using a voting ensemble over chunks.
* Applies a **hierarchical occlusion method** (replacing sentence tokens with padding mask) to score and rank the influence of each sentence.
* Identifies and displays the most influential passages, allowing users to locate them in the PDF.

### 5. Report Generation & Downloads
* Generates downloadable, formatted PDF reports of:
  * Rhetorical Role predictions.
  * Generated summaries with model selection information.
  * Occlusion-based CJPE explanation findings.

---

## Project Structure

```
Legal_Rhetorical_Analyzer/
├── app.py                      # Main Streamlit dashboard entry point
├── rhetorical_integration.py    # Bridge for BiLSTM-CRF model loading & inference
├── requirements.txt            # Project python dependencies
├── assets/                     # Frontend styles & static assets
│   └── style.css               # Streamlit custom dark/premium styling
├── explanation_and_reasoning/   # CJPE (Court Judgment Prediction & Explanation)
│   ├── __init__.py
│   └── cjpe_pipeline.py        # XLNet inference & occlusion implementation
├── models/                     # Weight checkpoints for heavy models
│   ├── CJPE_XLNet/             # XLNet models and vocabulary files
│   ├── InLegalBERT/            # Embeddings/Classification weights
│   └── legal-pegasus/          # Pegasus tokenizers and weights
├── outputs/                    # Auto-saved run logs & predictions
├── rhetorical_role_module/     # Hierarchical BiLSTM-CRF model package
│   ├── backend/                # Services, schemas, and PDF processors
│   └── models/
│       └── hierarchical_bilstm_crf/
│           ├── checkpoint_loader.py # Hugging Face checkpoint download utility
│           └── model.py             # Custom BiLSTM-CRF PyTorch definition
├── sac_summarizer_module/      # Structure-Aware Chunking (SAC) pipeline
│   ├── config.py               # PBA budgets & max tokens configuration
│   └── engine.py               # SAC segmentation and summarization engine
├── scalar_wrr/                 # SCaLAR Pegasus WRR Pipeline
│   └── scalar_wrr_model.py     # Heuristics, recursive, and weighted tag systems
├── tldr_uniandes/              # Uniandes Qwen Prompting Pipeline
│   └── tldr_uniandes_model.py  # Single-agent reward & multi-agent synthesis
└── utils/                      # Helper utilities
    ├── document_state.py       # Global session states and mappings
    ├── pdf_extract.py          # PyPDF/Docx parser
    ├── pdf_generator.py        # ReportLab PDF report exporter
    ├── pdf_viewer.py           # Streamlit PDF rendering pipeline
    └── sentence_splitter.py    # Sentence boundary tokenizer
```

---

## Setup & Installation

### 1. Prerequisites
* Python 3.10 or 3.11 (recommended).
* CUDA-enabled GPU (optional but highly recommended for fast local inference).
* [Ollama](https://ollama.com/) (required for running Qwen models locally).

### 2. Environment Configuration
Clone the repository, create a Python virtual environment, and install dependencies:
```bash
# Clone the repository
cd Legal_Rhetorical_Analyzer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Weight Downloads
Make sure the following directory layouts exist under `models/` at the root of the project:

* **Legal-Pegasus**: Download/place your `legal-pegasus` weights in [models/legal-pegasus](file:///d:/WORK/internship/Legal_Rhetorical_Analyzer/Legal_Rhetorical_Analyzer/models/legal-pegasus).
* **CJPE XLNet**: Place tokenizer, config, and `pytorch_model.bin` in [models/CJPE_XLNet](file:///d:/WORK/internship/Legal_Rhetorical_Analyzer/Legal_Rhetorical_Analyzer/models/CJPE_XLNet).
* **Hierarchical BiLSTM-CRF**: The checkpoint files (`model_state4.tar`, `tag2idx.json`, `word2idx.json`) will be **automatically downloaded** from Hugging Face (`L-NLProc/LegalSeg_Hier_BiLSTM-CRF`) on the first run of the application or analysis by [checkpoint_loader.py](file:///d:/WORK/internship/Legal_Rhetorical_Analyzer/Legal_Rhetorical_Analyzer/rhetorical_role_module/models/hierarchical_bilstm_crf/checkpoint_loader.py).

### 4. Running the Ollama Local Service
If utilizing any of the Qwen-based modules (Uniandes Pipelines, SAC-LLM), ensure the Ollama server is running and pull the model:
```bash
# Pull the target Qwen 8B model
ollama pull qwen3:8b
```

---

## Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser.

### The User interface

<img width="1916" height="784" alt="Screenshot 2026-07-11 092237" src="https://github.com/user-attachments/assets/36420527-a758-4076-9172-b757c6ece4e8" />


<img width="1914" height="951" alt="Screenshot 2026-07-11 091949" src="https://github.com/user-attachments/assets/3b0cea2a-8e92-4884-a54d-882523ad6a07" />

### Step-by-Step Usage Guide

1. **Upload Document**: Drag & drop or browse for a court judgment PDF in the bottom-left uploader panel.
2. **Execute Rhetorical Analysis**: Click **Run Rhetorical Analysis** in the middle panel. This executes the BiLSTM-CRF model, tags each sentence, and caches/saves results under `outputs/`.
3. **Explore Rhetorical Roles**: Click on any of the role buttons (e.g., *Facts*, *Statute*, *Ratio*) to highlight matching sentences in the PDF viewer.
4. **Generate Summaries**:
   * Click the **FULL SUMMARY** button.
   * Pick a summarization model from the drop-down selector.
   * Click **Generate Summary** (uses local Pegasus or Qwen depending on selection).
   * Download the generated summary report.
5. **Analyze Predictions & Explanations**:
   * Click the **EXPLANATION & REASONING** button.
   * Click **Generate Explanation & Reasoning** (triggers the CJPE XLNet classification and occlusion scoring).
   * Review the prediction outcome and confidence.
   * Browse ranked influential passages and click **Locate in PDF** to jump directly to that page/sentence.
   * Download the explanation PDF report.
