# ARCHITECTURE COUNCIL REPORT
**Compiled by Agents A1-A5**

## [A1] System Architect Findings
- **Evidence**: `backend/server.py` is an 1800-line monolith. 
- **Risk**: P1. Domain boundaries do not exist. User auth, paper parsing, and database transactions occur in the same lexical scope.
- **Fix Recommendation**: Adopt Hexagonal Architecture or standard FastAPI Routers. Split into `api/`, `domain/`, and `infrastructure/`. (Effort: 3 weeks)

## [A4] Technical Debt Auditor Findings
- **Evidence**: `core/model_builder.py` relies heavily on massive inline string processing rather than structured abstract syntax trees (AST).
- **Risk**: P2. High maintenance burden.
