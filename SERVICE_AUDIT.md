# Service Layer Audit

## Overview
The service layer is practically non-existent. Most business logic resides directly in the route handlers in `server.py`.

### Critical Findings
- **Tight Coupling**: Controllers (routes) are tightly coupled to the ORM (SQLAlchemy). 
- **Missing Abstraction**: Operations like fetching progress, compiling Dojo code, and saving to the database happen in the same function.
- **Lack of Dependency Injection**: Repositories aren't injected; they are instantiated inline or logic is written inline.
- **Maintainability Risk**: High. Modifying database schemas or external APIs requires rewriting route logic.
