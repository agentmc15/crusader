from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from .models import SubmissionRequest, SubmissionResponse, SubmissionStatus
from .services import SubmissionService
from ..shared.utils import setup_logging

logger = setup_logging("submission-api")

app = FastAPI(
    title="APR 2.0 Submission API",
    description="API for manuscript submissions in the Agentic Peer-Review system",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

submission_service = SubmissionService()

@app.post("/submissions", response_model=SubmissionResponse)
async def create_submission(request: SubmissionRequest):
    """Submit a new manuscript for review."""
    try:
        return await submission_service.create_submission(request)
    except Exception as e:
        logger.error(f"Submission creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Submission creation failed")

@app.get("/submissions/{submission_id}", response_model=SubmissionStatus)
async def get_submission_status(submission_id: str):
    """Get the status of a submission."""
    try:
        return await submission_service.get_submission_status(submission_id)
    except Exception as e:
        logger.error(f"Failed to get submission status: {str(e)}")
        raise HTTPException(status_code=404, detail="Submission not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "submission-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
