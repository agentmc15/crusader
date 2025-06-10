import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import logging

def generate_submission_id() -> str:
    """Generate a unique submission ID."""
    return f"sub_{uuid.uuid4().hex[:12]}"

def generate_agent_id() -> str:
    """Generate a unique agent execution ID."""
    return f"agent_{uuid.uuid4().hex[:8]}"

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def create_audit_entry(
    event_type: str,
    entity_id: str,
    details: Dict[str, Any],
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized audit log entry."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "entity_id": entity_id,
        "agent_id": agent_id,
        "details": details,
        "audit_id": str(uuid.uuid4())
    }

def setup_logging(service_name: str) -> logging.Logger:
    """Setup standardized logging for services."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {service_name} - %(levelname)s - %(message)s'
    )
    return logging.getLogger(service_name)

class AgentResponse:
    """Standardized agent response format."""
    
    def __init__(self, agent_name: str, submission_id: str):
        self.agent_name = agent_name
        self.submission_id = submission_id
        self.execution_id = generate_agent_id()
        self.timestamp = datetime.utcnow().isoformat()
        
    def success(self, verdict: str, confidence: float, evidence: list, recommendations: list = None):
        return {
            "agent_name": self.agent_name,
            "submission_id": self.submission_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "status": "completed",
            "verdict": verdict,
            "confidence_score": confidence,
            "evidence": evidence,
            "recommendations": recommendations or [],
            "processing_time_ms": None  # To be filled by caller
        }
    
    def error(self, error_message: str, error_code: str = "PROCESSING_ERROR"):
        return {
            "agent_name": self.agent_name,
            "submission_id": self.submission_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "status": "error",
            "error_code": error_code,
            "error_message": error_message
        }
