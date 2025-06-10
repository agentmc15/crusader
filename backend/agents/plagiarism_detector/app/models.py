from pydantic import BaseModel
from typing import List, Optional

class PlagiarismMatch(BaseModel):
    source: str
    similarity_score: float
    matched_text: str
    source_text: str
    start_position: int
    end_position: int

class PlagiarismResult(BaseModel):
    overall_similarity: float
    matches: List[PlagiarismMatch]
    verdict: str  # "clear", "minor_concerns", "major_concerns"
    confidence: float
