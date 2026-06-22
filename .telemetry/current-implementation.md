## Current Implementation

**SDK:** None — custom implementation
**Captured:** 2026-06-11

### Initialization

No analytics SDK is initialized. The application uses two independent persistence mechanisms:

1. **Client-side:** Browser `localStorage` API, accessed directly throughout `static/index.html`. No initialization step — localStorage is always available.
2. **Server-side:** SQLAlchemy ORM models (`LearnerProgress`, `AssessmentAttempt`, `TutorAnalytics`) initialized via FastAPI dependency injection (`get_db`). Database tables are auto-created by SQLAlchemy on startup.

Identity is established on page load: if `localStorage.learner_id` is empty, a UUID is generated and stored. This ID is then attached to all server requests via the `X-Learner-ID` HTTP header.

### Client vs Server

**Both.** Tracking calls are made in the browser (localStorage writes) and on the server (SQLAlchemy database writes). There is overlap — exercise results are saved to both localStorage and the server database independently.

- **Browser:** localStorage writes for dojo progress, submissions, notes, UI state, theme, estimator count, lab notebook
- **Server:** Database writes for module progress, exercise submissions, assessment attempts, tutor interactions

### Call Routing

**Scattered — direct calls, no wrapper.**

- Client-side: `localStorage.setItem()` and `localStorage.getItem()` called inline wherever data needs to be persisted. No centralized tracking function.
- Server-side: Each FastAPI endpoint handler (`@app.post(...)`) directly creates or updates SQLAlchemy model instances. No middleware, no tracking wrapper, no event queue.

### Identity Management

- **Client identity:** `localStorage.learner_id` — anonymous UUID, generated once on first visit via `crypto.randomUUID()` with a Math.random fallback.
- **Server identity:** The `X-Learner-ID` header is read by FastAPI endpoints using `Header(alias="X-Learner-ID", default="")`.
- **No identify() call:** There is no formal identify step — the UUID is generated and stored locally, then passed as a header. No traits (email, name, plan) are ever attached.
- **No group() call:** No group or account hierarchy.
- **No reset/logout:** The anonymous ID persists indefinitely. There is no logout flow or identity reset mechanism.

### Environment Variables

- `DATABASE_URL` — Optional. If set, connects to PostgreSQL instead of default SQLite (`./paper2code.db`).
- No analytics-related environment variables (no write keys, API keys, or tracking configuration).

### Error Handling

- **Server-side:** All API endpoint handlers wrap database operations in try/except blocks. Errors are logged via `logger.error()` and return HTTP 500 with the error message. Tracking failures (e.g., database write fails) are surfaced to the user as API errors.
- **Client-side:** No error handling around localStorage calls. If localStorage is full or unavailable (e.g., private browsing in some browsers), writes fail silently.

### Shutdown / Flush

Not applicable. There is no analytics SDK that buffers events. localStorage writes are synchronous and immediate. Server-side database writes are committed within the request lifecycle via SQLAlchemy session management.
