from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import httpx
from ..shared.utils import setup_logging

logger = setup_logging("orchestrator")

app = FastAPI(
    title="APR 2.0 Orchestrator",
    description="Workflow orchestration service",
    version="0.1.0"
)

class TriggerRequest(BaseModel):
    submission_id: str
    submission_data: Dict[str, Any]

# Simple in-memory state storage for demo
workflow_states: Dict[str, Dict[str, Any]] = {}

@app.post("/trigger")
async def trigger_workflow(request: TriggerRequest, background_tasks: BackgroundTasks):
    """Trigger a new workflow."""
    logger.info(f"Triggering workflow for {request.submission_id}")
    
    workflow_states[request.submission_id] = {
        "submission_id": request.submission_id,
        "status": "processing",
        "current_stage": "queued",
        "progress": 0
    }
    
    # Start background processing
    background_tasks.add_task(process_workflow, request.submission_id, request.submission_data)
    
    return {"message": "Workflow triggered", "submission_id": request.submission_id}

async def process_workflow(submission_id: str, submission_data: Dict[str, Any]):
    """Process the workflow in background."""
    try:
        # Simulate workflow progression
        stages = ["intake", "plagiarism_detection", "ethics_review", "technical_review", "synthesis", "completed"]
        
        for i, stage in enumerate(stages):
            workflow_states[submission_id].update({
                "current_stage": stage,
                "progress": (i + 1) / len(stages) * 100
            })
            
            logger.info(f"Processing stage {stage} for {submission_id}")
            await asyncio.sleep(2)  # Simulate processing time
        
        workflow_states[submission_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Workflow failed for {submission_id}: {str(e)}")
        workflow_states[submission_id].update({
            "status": "error",
            "error": str(e)
        })

@app.get("/status/{submission_id}")
async def get_status(submission_id: str):
    """Get workflow status."""
    if submission_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    return workflow_states[submission_id]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "orchestrator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
