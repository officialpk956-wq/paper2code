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



def _graph(node_type):
    return {"nodes": [{"id": "1", "type": node_type, "label": "L1", "params": {}}], "edges": [], "metadata": {}}


def test_compare_resolves_catalogue_slugs_against_free_form_titles(client, db_session):
    """Slugs are hyphenated lowercase; titles are free-form. 'vit', 'resnet-50'
    and 'u-net' all 404ed under raw substring matching, which made the compare
    page dead for its primary use case."""
    from backend.models import Paper

    for title in ("ResNet50", "Vision Transformer", "U-Net", "MobileNetV2"):
        db_session.add(Paper(title=title, architecture_graph=_graph("conv2d")))
    db_session.commit()

    for a, b, ta, tb in (("resnet-50", "vit", "ResNet50", "Vision Transformer"),
                         ("u-net", "mobilenet-v2", "U-Net", "MobileNetV2")):
        r = client.get(f"/api/architectures/compare?a_slug={a}&b_slug={b}")
        assert r.status_code == 200, r.text
        assert (r.json()["paper_a"]["title"], r.json()["paper_b"]["title"]) == (ta, tb)


def test_compare_prefers_exact_then_shortest_then_graphed(client, db_session):
    from backend.models import Paper

    db_session.add(Paper(title="ResNet50", architecture_graph=_graph("conv2d")))
    db_session.add(Paper(title="ResNet18", architecture_graph=_graph("conv2d")))
    db_session.add(Paper(title="ResNet", architecture_graph=None))  # exact name, no graph
    db_session.add(Paper(title="Vision Transformer", architecture_graph=_graph("linear")))
    db_session.commit()

    r = client.get("/api/architectures/compare?a_slug=resnet&b_slug=vit")
    # exact title wins the resolution even without a graph -> reported as incomplete, by name
    assert r.status_code == 200 and r.json().get("status") == "incomplete"
    assert "ResNet" in r.json()["message"]

    r = client.get("/api/architectures/compare?a_slug=resnet-18&b_slug=vit")
    assert r.status_code == 200 and r.json()["paper_a"]["title"] == "ResNet18"


def test_compare_404_names_the_slug_that_failed(client, db_session):
    from backend.models import Paper

    db_session.add(Paper(title="ResNet50", architecture_graph=_graph("conv2d")))
    db_session.commit()
    r = client.get("/api/architectures/compare?a_slug=resnet-50&b_slug=bert")
    assert r.status_code == 404
    assert "'bert'" in r.json()["detail"] and "resnet" not in r.json()["detail"].lower()
