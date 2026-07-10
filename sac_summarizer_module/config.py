# =============================================================================
# Structure-Aware Chunking (SAC) Legal Summarizer - Configuration Constants
#
# These parameters are fixed according to:
# Sonowal & Sadhu (2025), "Structure-Aware Chunking for Abstractive
# Summarization of Long Legal Documents"
# =============================================================================

# Maximum token budget per chunk fed to the summarization model (§3.1)
MAX_TOKENS = 1024

# PBA (Proportional Budget Allocation) word budgets per section (§3.3.2)
# Fixed-ratio split: 30% / 50% / 20% of a target 500-word summary
BUDGET_FACTS = 150
BUDGET_ANALYSIS = 250
BUDGET_CONCLUSION = 100

# Sliding-window overlap size in sentences (§3.3.1)
# Prepend the last 2 sentences of the previous chunk to maintain local continuity
OVERLAP_SENTENCES = 2
