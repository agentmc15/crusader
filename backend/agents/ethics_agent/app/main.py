from fastapi import FastAPI, HTTPException
from ...shared.agents import BaseAgent, AgentRequest
from ...shared.utils import AgentResponse, setup_logging
from typing import List, Dict, Any

logger = setup_logging("ethics-agent")

class EthicsReviewService(BaseAgent):
    """Agent for reviewing ethical compliance."""
    
    def __init__(self):
        super().__init__("ethics_agent")
    
    async def _process_submission(self, request: AgentRequest) -> Dict[str, Any]:
        response = AgentResponse(self.agent_name, request.submission_id)
        
        ethics_score = self._assess_ethics_compliance(request.manuscript_content, request.metadata)
        
        if ethics_score >= 0.8:
            verdict = "compliant"
            confidence = 0.9
        elif ethics_score >= 0.5:
            verdict = "minor_concerns"
            confidence = 0.8
        else:
            verdict = "major_concerns"
            confidence = 0.95
        
        evidence = [
            f"Ethics compliance score: {ethics_score:.2f}",
            f"IRB approval mentioned: {'Yes' if self._has_irb_approval(request.manuscript_content) else 'No'}",
            f"Informed consent discussed: {'Yes' if self._has_informed_consent(request.manuscript_content) else 'No'}"
        ]
        
        recommendations = self._generate_ethics_recommendations(request.manuscript_content)
        
        return response.success(verdict, confidence, evidence, recommendations)
    
    def _assess_ethics_compliance(self, text: str, metadata: Dict[str, Any]) -> float:
        score = 0.0
        text_lower = text.lower()
        
        # Check for ethics statement in metadata
        if metadata.get("ethics_statement"):
            score += 0.3
        
        # Check for IRB approval
        if self._has_irb_approval(text):
            score += 0.3
        
        # Check for informed consent
        if self._has_informed_consent(text):
            score += 0.2
        
        # Check for conflict of interest disclosure
        if metadata.get("conflict_of_interest") or "conflict of interest" in text_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    def _has_irb_approval(self, text: str) -> bool:
        terms = ["irb", "institutional review board", "ethics committee", "ethics approval"]
        return any(term in text.lower() for term in terms)
    
    def _has_informed_consent(self, text: str) -> bool:
        terms = ["informed consent", "consent form", "participant consent"]
        return any(term in text.lower() for term in terms)
    
    def _generate_ethics_recommendations(self, text: str) -> List[str]:
        recommendations = []
        if not self._has_irb_approval(text):
            recommendations.append("Include IRB approval information")
        if not self._has_informed_consent(text):
            recommendations.append("Describe informed consent procedures")
        return recommendations

app = FastAPI(title="Ethics Review Agent", version="0.1.0")
ethics_service = EthicsReviewService()

@app.post("/analyze")
async def analyze_ethics(request: AgentRequest):
    try:
        return await ethics_service.process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "ethics_agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
