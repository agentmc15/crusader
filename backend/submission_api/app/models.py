from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

class Author(BaseModel):
    name: str
    email: EmailStr
    affiliation: str
    orcid: Optional[str] = None

class SubmissionRequest(BaseModel):
    title: str
    abstract: str
    authors: List[Author]
    subject_area: str
    manuscript_file: str  # File path or base64 content
    supplementary_files: Optional[List[str]] = None
    funding_info: Optional[str] = None
    ethics_statement: Optional[str] = None
    conflict_of_interest: Optional[str] = None

class SubmissionResponse(BaseModel):
    submission_id: str
    status: str
    created_at: datetime
    estimated_review_time: str

class SubmissionStatus(BaseModel):
    submission_id: str
    status: str
    current_stage: str
    progress_percentage: float
    agent_results: List[Dict[str, Any]]
    last_updated: datetime
