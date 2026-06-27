import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON
from backend.database import Base

class BackgroundJob(Base):
    __tablename__ = "background_jobs"

    id             = Column(String(36), primary_key=True, index=True)
    name           = Column(String(255), nullable=False, index=True)
    correlation_id = Column(String(100), nullable=True, index=True)
    user_id        = Column(Integer, nullable=True, index=True)
    organization_id = Column(Integer, nullable=True, index=True)
    api_key_id     = Column(String(255), nullable=True, index=True)
    status         = Column(String(50), default="Queued", nullable=False, index=True)
    retry_count    = Column(Integer, default=0, nullable=False)
    max_retries    = Column(Integer, default=3, nullable=False)
    payload        = Column(JSON, nullable=True)
    created_at     = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    started_at     = Column(DateTime, nullable=True)
    completed_at   = Column(DateTime, nullable=True)
    failure_reason = Column(String(1024), nullable=True)
    progress_pct   = Column(Integer, default=0, nullable=False)
    metadata_json  = Column(JSON, nullable=True)
