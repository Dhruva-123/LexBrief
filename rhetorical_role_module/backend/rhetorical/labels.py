# Single source of truth for LegalSeg rhetorical roles and indices.

# Exact label-to-index mapping from the original trained Hierarchical BiLSTM-CRF checkpoint
TAG_TO_IDX = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "None": 3,
    "Facts": 4,
    "Issue": 5,
    "Arguments of Petitioner": 6,
    "Arguments of Respondent": 7,
    "Reasoning": 8,
    "Decision": 9
}

IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}

# Seven rhetorical roles as requested by the user
RHETORICAL_ROLES = [
    "Facts",
    "Issue",
    "Argument of Petitioner",
    "Argument of Respondent",
    "Reasoning",
    "Decision",
    "None"
]

# Mapping between user role names and the model's exact tag names
USER_TO_MODEL = {
    "Facts": "Facts",
    "Issue": "Issue",
    "Argument of Petitioner": "Arguments of Petitioner",
    "Argument of Respondent": "Arguments of Respondent",
    "Reasoning": "Reasoning",
    "Decision": "Decision",
    "None": "None"
}

MODEL_TO_USER = {v: k for k, v in USER_TO_MODEL.items()}

# Centrally configured color codes for each role (used in UI badges)
# Colors are selected for readability, high-contrast, and premium design aesthetics.
ROLE_COLORS = {
    "Facts": "#1E40AF",                   # Deep Blue
    "Issue": "#D97706",                   # Warm Amber
    "Argument of Petitioner": "#059669",  # Emerald Green
    "Argument of Respondent": "#DC2626",  # Crimson Red
    "Reasoning": "#7C3AED",               # Purple
    "Decision": "#C026D3",                # Magenta/Fuchsia
    "None": "#6B7280"                     # Neutral Gray
}

# Pastel colors specifically for PDF text highlights to ensure high legibility of the legal text
ROLE_HIGHLIGHT_COLORS = {
    "Facts": "#C7D2FE",                   # Light Indigo
    "Issue": "#FDE68A",                   # Light Amber
    "Argument of Petitioner": "#A7F3D0",  # Light Emerald
    "Argument of Respondent": "#FCA5A5",  # Light Red
    "Reasoning": "#DDD6FE",               # Light Purple
    "Decision": "#FBCFE8",                # Light Pink
    "None": "#E5E7EB"                     # Light Gray
}

def get_role_color(role: str) -> str:
    """Returns hex color code for the given user role in UI badges."""
    return ROLE_COLORS.get(role, "#6B7280")

def get_role_highlight_color(role: str) -> str:
    """Returns hex color code for the given user role in PDF text highlighting."""
    return ROLE_HIGHLIGHT_COLORS.get(role, "#E5E7EB")
