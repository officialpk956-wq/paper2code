import pytest
from backend.models import Paper

def test_implement_no_graph(client, db_session):
    # Seed a paper with architecture_graph=None
    paper = Paper(
        title="Test Implement No Graph",
        abstract="Test abstract",
        architecture_graph=None,
        visibility="public"
    )
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)

    r = client.get(f"/api/papers/{paper.id}/implement")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "no_graph"
    assert "starter_code" in data
    assert len(data["starter_code"]) > 0

    db_session.delete(paper)
    db_session.commit()

def test_implement_with_graph(client, db_session):
    # Seed a paper with architecture_graph={"nodes": [{"id":"l0","type":"conv2d"},{"id":"l1","type":"linear"}], "edges":[]}
    paper = Paper(
        title="Test Implement With Graph",
        abstract="Test abstract",
        architecture_graph={"nodes": [{"id":"l0","type":"conv2d"},{"id":"l1","type":"linear"}], "edges":[]},
        visibility="public"
    )
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)

    r = client.get(f"/api/papers/{paper.id}/implement")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "class" in data["starter_code"]
    assert "shapes" in data
    assert "layer_docs" in data

    db_session.delete(paper)
    db_session.commit()

def test_implement_404(client):
    r = client.get("/api/papers/99999/implement")
    assert r.status_code == 404
