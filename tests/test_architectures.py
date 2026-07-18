def test_knowledge_relations_transformer(client):
    r = client.get("/api/architectures/transformer/knowledge-relations")
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) >= 2
    assert any(c["relation"] == "INCOMPATIBLE" for c in data["constraints"])
    assert any(c["relation"] == "COMPATIBLE" for c in data["constraints"])

def test_knowledge_relations_unknown_slug(client):
    r = client.get("/api/architectures/unknown-arch-xyz/knowledge-relations")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == []
    assert data["constraints"] == []

def test_compare_endpoint_two_papers(client, db_session):
    from backend.models import Paper

    pa = Paper(title="transformer", architecture_graph={"nodes": [{"id": "1", "type": "linear", "label": "L1", "params": {}}], "edges": [], "metadata": {}})
    pb = Paper(title="resnet", architecture_graph={"nodes": [{"id": "1", "type": "conv2d", "label": "C1", "params": {}}], "edges": [], "metadata": {}})
    db_session.add_all([pa, pb])
    db_session.commit()

    r = client.get(f"/api/architectures/compare?paper_a={pa.id}&paper_b={pb.id}")
    assert r.status_code == 200
    data = r.json()
    assert "diff" in data
    assert "paper_a" in data
    assert "paper_b" in data

    db_session.delete(pa)
    db_session.delete(pb)
    db_session.commit()

def test_compare_endpoint_missing_graph(client, db_session):
    from backend.models import Paper

    pa = Paper(title="transformer missing", architecture_graph=None)
    pb = Paper(title="resnet full", architecture_graph={"nodes": [], "edges": []})
    db_session.add_all([pa, pb])
    db_session.commit()

    r = client.get(f"/api/architectures/compare?paper_a={pa.id}&paper_b={pb.id}")
    assert r.status_code == 200
    assert r.json()["status"] == "incomplete"

    db_session.delete(pa)
    db_session.delete(pb)
    db_session.commit()

def test_compare_404(client):
    r = client.get("/api/architectures/compare?paper_a=99999&paper_b=99998")
    assert r.status_code == 404

