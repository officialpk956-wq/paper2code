# PHASE 0: LAUNCH BLOCKER IMPLEMENTATION TASKS

## Task 1: Fix LearnerProgress Entity ID Access Crash
**Ticket ID**: PHASE0-001
**Priority**: CRITICAL
**Effort**: 2.5 hours
**Status**: 🟥 NOT STARTED

### Description
The LearnerProgress data model uses polymorphic (entity_type, entity_id) but three routes still reference old columns (module_id, paper_id), causing 500 crashes.

### Files to Modify
- `backend/routers/learning.py` (lines 422-423)
- `backend/routers/tutor.py` (line 47)
- `backend/routers/assessment.py` (line 43)

### Acceptance Criteria
- [ ] All 3 routes return HTTP 200 on valid LearnerProgress
- [ ] All 3 routes return HTTP 400 on invalid entity_id
- [ ] Test suite passes: `pytest tests/test_learning_progress_entity_fix.py -v`
- [ ] No AttributeError exceptions in Sentry logs

### Test Cases Required
1. Valid entity_type="paper_module" with entity_id="1" → 200 OK
2. Invalid entity_id (non-numeric) → 400 Bad Request
3. Missing entity_id → 400 Bad Request
4. Concurrent requests to same LearnerProgress → consistent results
5. Edge case: null entity_id → 400 Bad Request

---

## Task 2: Unify Email Verification & Fix Dual Dispatch
**Ticket ID**: PHASE0-002
**Priority**: CRITICAL
**Effort**: 3.0 hours
**Status**: 🟥 NOT STARTED

### Description
Registration sends TWO verification emails with different tokens. Fix: single email, single token, owned by auth_service.

### Files to Modify
- `backend/modules/auth/api/v1.py` (lines 89-101: remove duplicate token gen)
- `backend/modules/auth/services/auth_service.py` (ensure single token dispatch)

### Acceptance Criteria
- [ ] Exactly 1 verification email per registration
- [ ] Single token in email is valid and single-use
- [ ] Verification marks user as email_verified
- [ ] Test suite passes: `pytest tests/test_auth_registration_fix.py -v`
- [ ] Celery logs show 1 send_verification_email task per registration

### Test Cases Required
1. Register new user → 1 email sent ✅
2. Click verification link → user marked verified ✅
3. Try same token again → "Token already used" error ✅
4. Register with existing email → 400 Bad Request ✅
5. Email send failure → graceful error response ✅
6. Race: two simultaneous registrations same email → one succeeds, one fails ✅

---

## Task 3: Fix Authorization Engine Column Reference (IDOR Vulnerability)
**Ticket ID**: PHASE0-003
**Priority**: CRITICAL
**Effort**: 2.0 hours
**Status**: 🟥 NOT STARTED

### Description
Authz engine checks `paper.owner_id` but model has `paper.uploaded_by`. Creates IDOR: users can delete other users' papers.

### Files to Modify
- `backend/modules/authz/engine.py` (lines 80-89: correct column reference)

### Security Impact
- CVSS 7.4 (High): Insecure Direct Object Reference
- Any authenticated user can delete any paper
- Data loss vector

### Acceptance Criteria
- [ ] Paper owner CAN delete own papers (HTTP 204)
- [ ] Non-owner CANNOT delete paper (HTTP 403)
- [ ] Admin CAN delete any paper (HTTP 204)
- [ ] Test suite passes: `pytest tests/test_authz_paper_ownership_fix.py -v`
- [ ] Zero IDOR vulnerabilities detected in security scan

### Test Cases Required
1. Owner deletes own paper → 204 OK ✅
2. Non-owner deletes paper → 403 Forbidden ✅
3. Admin deletes any paper → 204 OK ✅
4. Non-existent paper → 404 Not Found (not 403) ✅
5. Concurrent delete attempts → second is 404, not 500 ✅
6. Try to delete via direct ID vs slug → same protection ✅
7. Manual IDOR test: fetch another user's API token, try delete → 403 ✅

---

## Task 4: Fix Weekly Leaderboard Points Column Reference
**Ticket ID**: PHASE0-004
**Priority**: CRITICAL
**Effort**: 3.5 hours
**Status**: 🟥 NOT STARTED

### Description
Weekly leaderboard ranks by lifetime User.points instead of User.weekly_points. A user inactive for 6 months ranks #1 because they scored 10k points historically.

### Files to Modify
- `backend/routers/leaderboard.py` (lines 84-111: period-specific queries)
- `backend/tasks/scheduled_tasks.py` (verify weekly reset logic)

### Acceptance Criteria
- [ ] Weekly leaderboard shows `weekly_points`, not lifetime
- [ ] Tie-breaking is deterministic (User ID as secondary sort)
- [ ] Inactive users excluded from weekly/monthly
- [ ] User with 0 weekly_points not shown on weekly board
- [ ] Test suite passes: `pytest tests/test_leaderboard_period_fix.py -v`
- [ ] Weekly reset task runs correctly (zeros out weekly_points each Monday)

### Test Cases Required
1. Weekly board: User with 0 weekly_points, 10k lifetime → not ranked ✅
2. Weekly board: User with 500 weekly_points → ranks correctly ✅
3. Determinism: Query weekly board twice → same ordering ✅
4. Tie-breaking: Two users with 500 weekly_points → consistent rank order ✅
5. Alltime board: Same query shows lifetime points ✅
6. Monthly board (if exists) → shows monthly_points ✅
7. Inactive user (last_active > 7 days ago) → excluded from weekly ✅
8. Weekly reset: crons at start of week, weekly_points→0 ✅

---

## Task 5: Fix GitHub Actions CD Workflow Trigger Name
**Ticket ID**: PHASE0-005
**Priority**: CRITICAL
**Effort**: 0.5 hours
**Status**: 🟥 NOT STARTED

### Description
CD workflow trigger waits for workflow named "CI", but actual workflow is "Paper2Code CI". Exact string match fails, deployments never trigger.

### Files to Modify
- `.github/workflows/cd.yml` (line 5: update workflow name in trigger)

### Deployment Impact
- CRITICAL: Zero deployments after every main merge
- Code accumulates; fixes never shipped
- Users never see new features

### Acceptance Criteria
- [ ] CI workflow name matches cd.yml workflows list (exact string)
- [ ] Test branch push → CI runs → CD automatically triggers
- [ ] CD workflow completes successfully (Render deployment happens)
- [ ] Main branch merge → full CI/CD pipeline executes

### Test Procedure
1. `git checkout -b test/cd-fix && echo "test" >> README.md && git push origin test/cd-fix`
2. Navigate to GitHub Actions tab
3. Verify "Paper2Code CI" workflow runs and completes
4. Verify CD workflow automatically triggers (workflow_run event fires)
5. Verify both workflows show "success" (green checkmarks)
6. Verify Render deployment happened (check Render dashboard)
7. Merge to main, repeat

---

## Execution Plan

### Day 1 Morning (Tasks 1 & 2 in parallel)
- **Developer A**: Task #1 (LearnerProgress entity_id) - 2.5h
- **Developer B**: Task #2 (Email verification) - 3.0h
- Result: Adaptive learning and registration working

### Day 1 Afternoon (Tasks 3, 4, 5 sequential)
- **Developer A or B**: Task #3 (Authz engine) - 2.0h
- **Developer A or B**: Task #4 (Leaderboard) - 3.5h
- **DevOps**: Task #5 (CD workflow) - 0.5h
- Result: All permissions, competitions, deployments working

### Day 1 Evening
- Run full Playwright E2E test suite
- Verify Sentry: zero new errors in above domains
- Deploy to staging
- Smoke test: registration → adaptive learning → dojo → leaderboard

### Day 2 Morning
- Code review all 5 PRs
- Merge to main
- Verify CD pipeline triggers automatically
- Deploy to production

---

## Verification Checklist

- [ ] All 5 tasks completed
- [ ] All test suites pass locally
- [ ] Code review approved (no security issues)
- [ ] Staging deployment successful
- [ ] Production deployment successful
- [ ] Sentry: zero regressions
- [ ] Smoke test: new user registration → adaptive learning → dojo works
- [ ] Leaderboard: verified fair ranking
- [ ] Mobile: verified responsive editor (Phase 2)

