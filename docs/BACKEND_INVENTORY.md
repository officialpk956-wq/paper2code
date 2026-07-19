# Backend Inventory

## Overview
- **File Count**: ~40 in `backend/`, ~186 in `core/`
- **Service Count**: 1 (Monolithic FastAPI instance in `backend/server.py`)
- **Route Count**: ~60+ routes heavily concentrated in `server.py`
- **Database Models**: `User`, `Problem`, `InterviewQuestion`, `Roadmap`, `LearnerProgress`, `Paper`, `PaperModule`
- **Repositories**: `UserRepository`
- **Workers**: 0 (Everything runs synchronously in the main event loop)
- **Pipelines**: `core/model_builder.py`, `core/transformer_builder.py`, Knowledge Graph generation (RAG)
- **External Integrations**: OpenAI / LLM APIs

## Dependency Map
FastAPI -> SQLite -> SQLAlchemy
Core AI -> subprocess -> local disk
