from fastapi import FastAPI, HTTPException
from .services import PlagiarismDetectionService
from ...shared.agents import AgentRequest
from ...shared.utils import setup_logging

logger = setup_logging("plagiarism-detector")

app = FastAPI(
    title="Plagiarism Detection Agent",
    description="Agent for detecting potential plagiarism in manuscripts",
    version="0.1.0"
)

plagiarism_service = PlagiarismDetectionService()

@app.post("/analyze")
async def analyze_plagiarism(request: AgentRequest):
    """Analyze manuscript for potential plagiarism."""
    try:
        return await plagiarism_service.process(request)
    except Exception as e:
        logger.error(f"Plagiarism analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "plagiarism_detector"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
