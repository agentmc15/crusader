import re
import hashlib
from typing import List, Dict, Any
from .models import PlagiarismMatch, PlagiarismResult
from ...shared.agents import BaseAgent, AgentRequest
from ...shared.utils import AgentResponse

class PlagiarismDetectionService(BaseAgent):
    """Plagiarism detection agent using text similarity analysis."""
    
    def __init__(self):
        super().__init__("plagiarism_detector")
        # In a real implementation, this would connect to databases like Crossref, arXiv, etc.
        self.known_sources = self._load_sample_database()
    
    def _load_sample_database(self) -> Dict[str, str]:
        """Load sample database of known texts for demo purposes."""
        return {
            "sample_paper_1": "The rapid advancement of artificial intelligence has transformed multiple industries.",
            "sample_paper_2": "Machine learning algorithms require substantial computational resources for training.",
            "sample_paper_3": "Peer review is a critical component of the scholarly publishing process."
        }
    
    async def _process_submission(self, request: AgentRequest) -> Dict[str, Any]:
        """Detect potential plagiarism in the manuscript."""
        response = AgentResponse(self.agent_name, request.submission_id)
        
        # Extract text content (simplified - would use proper PDF/document parsing)
        text_content = request.manuscript_content
        
        # Perform plagiarism detection
        result = await self._check_plagiarism(text_content)
        
        # Determine verdict
        if result.overall_similarity < 0.15:
            verdict = "clear"
            confidence = 0.9
        elif result.overall_similarity < 0.3:
            verdict = "minor_concerns"
            confidence = 0.8
        else:
            verdict = "major_concerns"
            confidence = 0.95
        
        evidence = [
            f"Overall similarity score: {result.overall_similarity:.2%}",
            f"Number of matches found: {len(result.matches)}",
            f"Highest individual match: {max([m.similarity_score for m in result.matches], default=0):.2%}"
        ]
        
        recommendations = []
        if verdict != "clear":
            recommendations.append("Review highlighted sections for potential plagiarism")
            recommendations.append("Verify proper citation of sources")
        
        return response.success(verdict, confidence, evidence, recommendations)
    
    async def _check_plagiarism(self, text: str) -> PlagiarismResult:
        """Check text against known sources."""
        matches = []
        
        # Simple text similarity check (in production, use sophisticated NLP methods)
        sentences = self._extract_sentences(text)
        
        for source_id, source_text in self.known_sources.items():
            for sentence in sentences:
                similarity = self._calculate_similarity(sentence, source_text)
                if similarity > 0.7:  # Threshold for potential match
                    match = PlagiarismMatch(
                        source=source_id,
                        similarity_score=similarity,
                        matched_text=sentence,
                        source_text=source_text,
                        start_position=text.find(sentence),
                        end_position=text.find(sentence) + len(sentence)
                    )
                    matches.append(match)
        
        overall_similarity = len(matches) / max(len(sentences), 1) if sentences else 0
        
        if overall_similarity < 0.15:
            verdict = "clear"
        elif overall_similarity < 0.3:
            verdict = "minor_concerns"
        else:
            verdict = "major_concerns"
        
        return PlagiarismResult(
            overall_similarity=overall_similarity,
            matches=matches,
            verdict=verdict,
            confidence=0.85
        )
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified implementation)."""
        # In production, use proper similarity algorithms like cosine similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
