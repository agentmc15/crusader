from typing import Dict, Any, List
import httpx
from ..shared.utils import setup_logging
from ..config.settings import settings

logger = setup_logging("orchestrator-nodes")

class WorkflowState:
    """Represents the state of a submission workflow."""
    
    def __init__(self, submission_id: str, submission_data: Dict[str, Any]):
        self.submission_id = submission_id
        self.submission_data = submission_data
        self.agent_results: List[Dict[str, Any]] = []
        self.current_stage = "initialized"
        self.status = "processing"
        self.errors: List[str] = []
    
    def add_agent_result(self, result: Dict[str, Any]):
        """Add an agent result to the workflow state."""
        self.agent_results.append(result)
    
    def has_errors(self) -> bool:
        """Check if there are any errors in the workflow."""
        return len(self.errors) > 0 or any(
            result.get("status") == "error" for result in self.agent_results
        )
    
    def get_agent_result(self, agent_name: str) -> Dict[str, Any]:
        """Get result from a specific agent."""
        for result in self.agent_results:
            if result.get("agent_name") == agent_name:
                return result
        return {}

async def call_agent(agent_url: str, state: WorkflowState) -> Dict[str, Any]:
    """Call an agent service and return the result."""
    try:
        request_data = {
            "submission_id": state.submission_id,
            "manuscript_content": state.submission_data.get("abstract", ""),  # Simplified
            "metadata": state.submission_data.get("metadata", {}),
            "previous_agent_results": state.agent_results
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{agent_url}/analyze", json=request_data)
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Failed to call agent at {agent_url}: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "agent_url": agent_url
        }

# Workflow node functions
async def submission_intake_node(state: WorkflowState) -> WorkflowState:
    """Initial submission processing."""
    logger.info(f"Processing submission intake for {state.submission_id}")
    state.current_stage = "intake"
    return state

async def plagiarism_detection_node(state: WorkflowState) -> WorkflowState:
    """Plagiarism detection step."""
    logger.info(f"Running plagiarism detection for {state.submission_id}")
    state.current_stage = "plagiarism_detection"
    
    result = await call_agent(settings.PLAGIARISM_DETECTOR_URL, state)
    state.add_agent_result(result)
    
    return state

async def ethics_review_node(state: WorkflowState) -> WorkflowState:
    """Ethics review step."""
    logger.info(f"Running ethics review for {state.submission_id}")
    state.current_stage = "ethics_review"
    
    result = await call_agent(settings.ETHICS_AGENT_URL, state)
    state.add_agent_result(result)
    
    return state

async def content_triage_node(state: WorkflowState) -> WorkflowState:
    """Content triage and routing."""
    logger.info(f"Running content triage for {state.submission_id}")
    state.current_stage = "content_triage"
    
    result = await call_agent(settings.CONTENT_TRIAGE_AGENT_URL, state)
    state.add_agent_result(result)
    
    return state

async def technical_review_node(state: WorkflowState) -> WorkflowState:
    """Parallel technical review by multiple agents."""
    logger.info(f"Running technical review for {state.submission_id}")
    state.current_stage = "technical_review"
    
    # Run multiple agents in parallel (simplified sequential for now)
    agents = [
        (settings.METHODOLOGY_REVIEWER_URL, "methodology_reviewer"),
        (settings.STATISTICAL_REVIEWER_URL, "statistical_reviewer"),
        (settings.DATA_INTEGRITY_AGENT_URL, "data_integrity_agent"),
        (settings.REPRODUCIBILITY_AGENT_URL, "reproducibility_agent"),
        (settings.CLARITY_AGENT_URL, "clarity_agent")
    ]
    
    for agent_url, agent_name in agents:
        result = await call_agent(agent_url, state)
        result["agent_name"] = agent_name  # Ensure agent name is set
        state.add_agent_result(result)
    
    return state

async def synthesis_node(state: WorkflowState) -> WorkflowState:
    """Synthesize all agent results."""
    logger.info(f"Synthesizing results for {state.submission_id}")
    state.current_stage = "synthesis"
    
    # Aggregate all agent verdicts
    verdicts = []
    total_confidence = 0
    agent_count = 0
    
    for result in state.agent_results:
        if result.get("status") == "completed":
            verdicts.append(result.get("verdict", "unknown"))
            total_confidence += result.get("confidence_score", 0)
            agent_count += 1
    
    # Simple decision logic
    if agent_count > 0:
        avg_confidence = total_confidence / agent_count
        
        # Check for any major issues
        if any(v in ["major_concerns", "inadequate", "reject"] for v in verdicts):
            final_decision = "reject"
        elif any(v in ["minor_concerns", "needs_improvement"] for v in verdicts):
            final_decision = "revisions_required"
        else:
            final_decision = "accept"
    else:
        final_decision = "error"
        avg_confidence = 0
    
    # Add synthesis result
    synthesis_result = {
        "agent_name": "synthesis_agent",
        "status": "completed",
        "verdict": final_decision,
        "confidence_score": avg_confidence,
        "evidence": [f"Aggregated results from {agent_count} agents"],
        "agent_verdicts": verdicts
    }
    
    state.add_agent_result(synthesis_result)
    state.current_stage = "completed"
    state.status = "completed"
    
    return state

async def error_handler_node(state: WorkflowState) -> WorkflowState:
    """Handle workflow errors."""
    logger.error(f"Handling errors for {state.submission_id}")
    state.current_stage = "error"
    state.status = "error"
    return state
