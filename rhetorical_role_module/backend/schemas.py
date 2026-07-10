from dataclasses import dataclass, asdict
from typing import Optional, List

@dataclass
class SentenceAnalysis:
    """
    Schema for a single analyzed sentence.
    """
    sentence_id: int
    text: str
    role: str
    role_id: int
    confidence: Optional[float]          # Will be null/None since CRF transition decoding doesn't yield calibrated probabilities
    page_number: Optional[int]
    bbox: Optional[List[float]]          # Overall bounding box: [x0, y0, x1, y1]
    rects: Optional[List[List[float]]] = None  # Precise word/line rectangles for multi-line highlight rendering
    prediction_source: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentAnalysis:
    """
    Schema for a fully analyzed document.
    """
    sentences: List[SentenceAnalysis]
    is_scanned: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sentences": [s.to_dict() for s in self.sentences],
            "is_scanned": self.is_scanned,
            "error_message": self.error_message
        }
