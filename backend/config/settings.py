from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://apr_user:apr_pass@db:5432/apr_db"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Services
    ORCHESTRATOR_URL: str = "http://orchestrator:8001"
    SUBMISSION_API_URL: str = "http://submission-api:8000"
    
    # Agent URLs
    PLAGIARISM_DETECTOR_URL: str = "http://plagiarism-detector:8002"
    METHODOLOGY_REVIEWER_URL: str = "http://methodology-reviewer:8003"
    ETHICS_AGENT_URL: str = "http://ethics-agent:8004"
    STATISTICAL_REVIEWER_URL: str = "http://statistical-reviewer:8005"
    DATA_INTEGRITY_AGENT_URL: str = "http://data-integrity-agent:8006"
    REPRODUCIBILITY_AGENT_URL: str = "http://reproducibility-agent:8007"
    CLARITY_AGENT_URL: str = "http://clarity-agent:8008"
    CONTENT_TRIAGE_AGENT_URL: str = "http://content-triage-agent:8009"
    
    class Config:
        env_file = ".env"

settings = Settings()
