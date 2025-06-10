from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from ..config.settings import settings

Base = declarative_base()

class Submission(Base):
    __tablename__ = "submissions"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    abstract = Column(Text)
    authors = Column(JSON)
    manuscript_url = Column(String)
    metadata = Column(JSON)
    status = Column(String, default="submitted")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AgentExecution(Base):
    __tablename__ = "agent_executions"
    
    execution_id = Column(String, primary_key=True)
    submission_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    verdict = Column(String)
    confidence_score = Column(Float)
    evidence = Column(JSON)
    recommendations = Column(JSON)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    audit_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    agent_id = Column(String)
    details = Column(JSON)

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
