import pytest
from fastapi.testclient import TestClient
from backend.models import Paper, PaperChallenge, PaperChallengePart, PaperPartSubmission

@pytest.fixture
def test_paper_challenges_seed(db_session):
    paper = Paper(title="Test Paper Challenges")
    db_session.add(paper)
    db_session.commit()

    challenge = PaperChallenge(paper_id=paper.id, title="Challenge 1", is_published=True)
    db_session.add(challenge)
    db_session.commit()

    part1 = PaperChallengePart(
        challenge_id=challenge.id,
        title="Part 1",
        description_md="desc 1",
        setup_code="x = 1",
        starter_code="def f(): pass",
        test_code="assert True",
        xp_reward=50,
        order_idx=1
    )
    db_session.add(part1)
    db_session.flush()
    
    part2 = PaperChallengePart(
        challenge_id=challenge.id,
        title="Part 2",
        description_md="desc 2",
        setup_code="y = 2",
        starter_code="def g(): pass",
        test_code="assert True",
        unlock_requires_part_id=part1.id,
        xp_reward=75,
        order_idx=2
    )
    db_session.add(part2)
    db_session.commit()

    return {
        "paper_id": paper.id,
        "challenge_id": challenge.id,
        "part1_id": part1.id,
        "part2_id": part2.id
    }

def test_get_paper_challenges(auth_client: TestClient, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    part2_id = test_paper_challenges_seed["part2_id"]

    res = auth_client.get(f"/api/papers/{paper_id}/challenges")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    parts = data[0]["parts"]
    assert len(parts) == 2

    # Part 2 should be locked
    part2 = next(p for p in parts if p["id"] == part2_id)
    assert part2["is_locked"] == True

def test_run_challenge_part_locked(auth_client: TestClient, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    challenge_id = test_paper_challenges_seed["challenge_id"]
    part2_id = test_paper_challenges_seed["part2_id"]

    res = auth_client.post(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part2_id}/run",
        json={"code": "def g(): pass"}
    )
    assert res.status_code == 403
    assert res.json()["detail"] == "Complete the previous part first"

# We mock E2B run to prevent network calls during testing
from unittest.mock import patch

@patch("backend.routers.paper_challenges.run_code_in_sandbox")
def test_run_challenge_part_success(mock_run, auth_client: TestClient, test_paper_challenges_seed, db_session):
    mock_run.return_value = {"passed": True, "stdout": "Test Passed", "time_ms": 100}

    paper_id = test_paper_challenges_seed["paper_id"]
    challenge_id = test_paper_challenges_seed["challenge_id"]
    part1_id = test_paper_challenges_seed["part1_id"]
    part2_id = test_paper_challenges_seed["part2_id"]

    res = auth_client.post(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part1_id}/run",
        json={"code": "def f(): return True"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["passed"] == True
    assert data["xp_earned"] == 50

    # Part 2 should now be unlocked and runable
    res2 = auth_client.post(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part2_id}/run",
        json={"code": "def g(): pass"}
    )
    # Part 2 requires mocked run, which passes
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["passed"] == True
    assert data2["xp_earned"] == 75

def test_admin_create_challenge_and_part(admin_client: TestClient, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    
    # Create challenge
    res = admin_client.post(
        f"/api/admin/papers/{paper_id}/challenges",
        json={"title": "Admin Created Challenge", "order_idx": 10}
    )
    assert res.status_code == 201
    c_id = res.json()["id"]

    # Create part
    res2 = admin_client.post(
        f"/api/admin/challenges/{c_id}/parts",
        json={
            "title": "Admin Part",
            "description_md": "desc",
            "starter_code": "code",
            "test_code": "test",
            "xp_reward": 100
        }
    )
    assert res2.status_code == 201

    # Publish
    res3 = admin_client.patch(
        f"/api/admin/challenges/{c_id}",
        json={"is_published": True}
    )
    assert res3.status_code == 200

def test_non_admin_cannot_create_challenge(auth_client, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    res = auth_client.post(
        f"/api/admin/papers/{paper_id}/challenges",
        json={"title": "Hacked Challenge", "order_idx": 0}
    )
    assert res.status_code == 403

@patch("backend.routers.paper_challenges.run_code_in_sandbox")
def test_run_part_no_xp_on_failure(mock_run, auth_client, test_paper_challenges_seed, db_session):
    mock_run.return_value = {
        "passed": False,
        "stdout": "",
        "stderr": "AssertionError: wrong output",
        "time_ms": 200
    }
    paper_id = test_paper_challenges_seed["paper_id"]
    challenge_id = test_paper_challenges_seed["challenge_id"]
    part1_id = test_paper_challenges_seed["part1_id"]

    res = auth_client.post(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part1_id}/run",
        json={"code": "def f(): return None"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["passed"] is False
    assert data["xp_earned"] == 0

def test_solution_code_never_returned(auth_client, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    challenge_id = test_paper_challenges_seed["challenge_id"]
    part1_id = test_paper_challenges_seed["part1_id"]

    # Check challenges list
    r1 = auth_client.get(f"/api/papers/{paper_id}/challenges")
    assert "solution_code" not in r1.text

    # Check single part
    r2 = auth_client.get(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part1_id}"
    )
    assert "solution_code" not in r2.text

def test_test_code_never_returned(auth_client, test_paper_challenges_seed):
    paper_id = test_paper_challenges_seed["paper_id"]
    challenge_id = test_paper_challenges_seed["challenge_id"]
    part1_id = test_paper_challenges_seed["part1_id"]

    r1 = auth_client.get(f"/api/papers/{paper_id}/challenges")
    assert "test_code" not in r1.text

    r2 = auth_client.get(
        f"/api/papers/{paper_id}/challenges/{challenge_id}/parts/{part1_id}"
    )
    assert "test_code" not in r2.text

