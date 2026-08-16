import pytest
from backend.models import Problem, Paper

def test_search_results_urls(db_session, client):
    # Setup problem with id and slug
    p = Problem(
        id="42",
        slug="numpy-basics",
        title="Numpy Basics",
        category="Math",
        difficulty="Easy",
        description="Learn numpy basics here",
        is_retired=False,
    )
    db_session.add(p)
    
    # Setup paper
    paper = Paper(
        title="Attention is All You Need",
        abstract="A transformer model for sequence tasks",
        visibility="public"
    )
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)

    # 1. Search for problem
    resp = client.get("/api/search?q=numpy&types=problems")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    res = data["results"][0]
    assert res["id"] == "42"
    assert res["url"] == "/dojo/numpy-basics"

    # 2. Search for paper (ensure no regression)
    resp2 = client.get("/api/search?q=Attention&types=papers")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["total"] == 1
    res2 = data2["results"][0]
    assert res2["id"] == paper.id
    assert res2["url"] == f"/papers/{paper.id}"

    # 3. Search for legacy problem with no slug
    p_legacy = Problem(
        id="legacy-id",
        slug=None,
        title="Legacy Problem",
        category="Math",
        difficulty="Easy",
        description="Legacy problem description",
        is_retired=False,
    )
    db_session.add(p_legacy)
    db_session.commit()

    resp3 = client.get("/api/search?q=Legacy&types=problems")
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["total"] == 1
    res3 = data3["results"][0]
    assert res3["id"] == "legacy-id"
    assert res3["url"] == "/dojo/legacy-id"
