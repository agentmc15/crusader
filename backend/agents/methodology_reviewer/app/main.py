from fastapi import FastAPI, HTTPException
from ...shared.agents import BaseAgent, AgentRequest
from ...shared.utils import AgentResponse, setup_logging
import re
from typing import List, Dict, Any

logger = setup_logging("methodology-reviewer")

class MethodologyReviewService(BaseAgent):
    """Agent for reviewing research methodology."""
    
    def __init__(self):
        super().__init__("methodology_reviewer")
        self.methodology_keywords = [
            "methodology", "method", "procedure", "protocol", "design",
            "sample", "participants", "data collection", "analysis",
            "statistical", "experimental", "control", "variable"
        ]
    
    async def _process_submission(self, request: AgentRequest) -> Dict[str, Any]:
        """Review the methodology section of the manuscript."""
        response = AgentResponse(self.agent_name, request.submission_id)
        
        # Extract methodology-related content
        methodology_score = self._assess_methodology(request.manuscript_content)
        
        # Determine verdict based on methodology quality
        if methodology_score >= 0.8:
            verdict = "excellent"
            confidence = 0.9
        elif methodology_score >= 0.6:
            verdict = "good"
            confidence = 0.8
        elif methodology_score >= 0.4:
            verdict = "needs_improvement"
            confidence = 0.85
        else:
            verdict = "inadequate"
            confidence = 0.9
        
        evidence = [
            f"Methodology clarity score: {methodology_score:.2f}",
            f"Contains methodology section: {'Yes' if self._has_methodology_section(request.manuscript_content) else 'No'}",
            f"Statistical methods described: {'Yes' if self._has_statistical_methods(request.manuscript_content) else 'No'}"
        ]
        
        recommendations = self._generate_recommendations(methodology_score, request.manuscript_content)
        
        return response.success(verdict, confidence, evidence, recommendations)
    
    def _assess_methodology(self, text: str) -> float:
        """Assess the quality of methodology description."""
        score = 0.0
        text_lower = text.lower()
        
        # Check for methodology section
        if self._has_methodology_section(text):
            score += 0.3
        
        # Check for key methodology concepts
        keyword_count = sum(1 for keyword in self.methodology_keywords if keyword in text_lower)
        score += min(keyword_count / len(self.methodology_keywords), 0.4)
        
        # Check for statistical methods
        if self._has_statistical_methods(text):
            score += 0.2
        
        # Check for sample description
        if any(term in text_lower for term in ["sample size", "participants", "subjects"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _has_methodology_section(self, text: str) -> bool:
        """Check if text contains a methodology section."""
        methodology_headers = [
            "methodology", "methods", "experimental design",
            "research design", "study design"
        ]
        text_lower = text.lower()
        return any(header in text_lower for header in methodology_headers)
    
    def _has_statistical_methods(self, text: str) -> bool:
        """Check if statistical methods are described."""
        statistical_terms = [
            "statistical analysis", "p-value", "confidence interval",
            "regression", "anova", "chi-square", "t-test"
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in statistical_terms)
    
    def _generate_recommendations(self, score: float, text: str) -> List[str]:
        """Generate recommendations based on methodology assessment."""
        recommendations = []
        
        if score < 0.6:
            recommendations.append("Expand the methodology section with more detailed descriptions")
        
        if not self._has_methodology_section(text):
            recommendations.append("Include a dedicated methodology section")
        
        if not self._has_statistical_methods(text):
            recommendations.append("Describe statistical analysis methods used")
        
        if score < 0.4:
            recommendations.append("Provide clearer justification for methodological choices")
            recommendations.append("Include information about data collection procedures")
        
        return recommendations

app = FastAPI(
    title="Methodology Review Agent",
    description="Agent for reviewing research methodology in manuscripts",
    version="0.1.0"
)

methodology_service = MethodologyReviewService()

@app.post("/analyze")
async def analyze_methodology(request: AgentRequest):
    """Analyze manuscript methodology."""
    try:
        return await methodology_service.process(request)
    except Exception as e:
        logger.error(f"Methodology analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "methodology_reviewer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
