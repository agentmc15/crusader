from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
import time
from ..utils import AgentResponse, setup_logging

class AgentRequest(BaseModel):
    submission_id: str
    manuscript_content: str
    metadata: Dict[str, Any]
    previous_agent_results: List[Dict[str, Any]] = []

class BaseAgent(ABC):
    """Base class for all review agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = setup_logging(agent_name)
    
    async def process(self, request: AgentRequest) -> Dict[str, Any]:
        """Process a submission and return agent verdict."""
        start_time = time.time()
        response_helper = AgentResponse(self.agent_name, request.submission_id)
        
        try:
            self.logger.info(f"Processing submission {request.submission_id}")
            result = await self._process_submission(request)
            
            processing_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time
            
            self.logger.info(f"Completed processing {request.submission_id} in {processing_time}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {request.submission_id}: {str(e)}")
            return response_helper.error(str(e))
    
    @abstractmethod
    async def _process_submission(self, request: AgentRequest) -> Dict[str, Any]:
        """Implement the specific agent logic."""
        pass
