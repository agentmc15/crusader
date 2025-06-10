import httpx
from typing import Dict, Any
from .models import SubmissionRequest, SubmissionResponse, SubmissionStatus
from ..shared.utils import generate_submission_id, create_audit_entry, setup_logging
from ..shared.db import Submission, get_db
from ..config.settings import settings
from datetime import datetime

logger = setup_logging("submission-api")

class SubmissionService:
    
    async def create_submission(self, request: SubmissionRequest) -> SubmissionResponse:
        """Create a new manuscript submission."""
        submission_id = generate_submission_id()
        
        # Create submission record
        submission_data = {
            "id": submission_id,
            "title": request.title,
            "abstract": request.abstract,
            "authors": [author.dict() for author in request.authors],
            "manuscript_url": request.manuscript_file,
            "metadata": {
                "subject_area": request.subject_area,
                "funding_info": request.funding_info,
                "ethics_statement": request.ethics_statement,
                "conflict_of_interest": request.conflict_of_interest,
                "supplementary_files": request.supplementary_files
            },
            "status": "submitted"
        }
        
        # TODO: Save to database
        
        # Trigger orchestrator workflow
        await self._trigger_review_workflow(submission_id, submission_data)
        
        logger.info(f"Created submission {submission_id}")
        
        return SubmissionResponse(
            submission_id=submission_id,
            status="submitted",
            created_at=datetime.utcnow(),
            estimated_review_time="3-5 business days"
        )
    
    async def get_submission_status(self, submission_id: str) -> SubmissionStatus:
        """Get the current status of a submission."""
        # TODO: Fetch from database and orchestrator
        
        return SubmissionStatus(
            submission_id=submission_id,
            status="in_review",
            current_stage="plagiarism_detection",
            progress_percentage=25.0,
            agent_results=[],
            last_updated=datetime.utcnow()
        )
    
    async def _trigger_review_workflow(self, submission_id: str, submission_data: Dict[str, Any]):
        """Trigger the orchestrator to start the review workflow."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.ORCHESTRATOR_URL}/trigger",
                    json={
                        "submission_id": submission_id,
                        "submission_data": submission_data
                    }
                )
                response.raise_for_status()
                logger.info(f"Triggered workflow for submission {submission_id}")
        except Exception as e:
            logger.error(f"Failed to trigger workflow for {submission_id}: {str(e)}")
            raise
