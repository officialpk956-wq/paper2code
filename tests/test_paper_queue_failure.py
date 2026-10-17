import asyncio
from unittest.mock import patch

from backend.models import Task
from backend.repositories.task_repository import TaskRepository
from backend.routers.papers_pipeline import _enqueue_paper


def test_confirm_queue_failure_does_not_charge_storage(client, db_session, regular_user_headers):
    from backend.models import User

    with (
        patch("backend.services.storage_service.get_object_size", return_value=1024),
        patch("backend.redis_config.cache_redis", None),
        patch(
            "backend.routers.papers_pipeline.generate_code_from_pdf_task.apply_async",
            side_effect=ConnectionError("queue unavailable"),
        ),
    ):
        response = client.post(
            "/api/papers/confirm-upload",
            headers=regular_user_headers,
            json={
                "key": "papers/queue-failure.pdf",
                "paper_name": "queue-confirm-test",
                "visibility": "private",
                "terms_accepted": True,
            },
        )
    assert response.status_code == 503
    task = db_session.query(Task).filter_by(input_ref="queue-confirm-test").one()
    assert task.status == "failed"
    user = db_session.query(User).filter_by(id=task.user_id).one()
    assert not user.storage_bytes_used


def test_upload_queue_failure_is_sanitized_and_persisted(client, db_session, regular_user_headers):
    with (
        patch("backend.services.storage_service.store_pdf", return_value="retained-test.pdf"),
        patch(
            "backend.routers.papers_pipeline.generate_code_from_pdf_task.apply_async",
            side_effect=ConnectionError("redis://secret-password@localhost:6379"),
        ),
        patch("backend.services.storage_service.cleanup") as cleanup,
    ):
        response = client.post(
            "/api/papers/upload",
            headers=regular_user_headers,
            data={"terms_accepted": "true", "visibility": "private"},
            files={"file": ("queue-test.pdf", b"%PDF-1.4 fixture", "application/pdf")},
        )
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
    assert "secret-password" not in response.text
    task = db_session.query(Task).filter_by(input_ref="queue-test").one()
    assert task.status == "failed"
    assert task.error == response.json()["detail"]
    cleanup.assert_not_called()  # Delivery can be ambiguous after a transport error.


def test_dispatch_disables_result_subscription_and_retries(db_session):
    task = TaskRepository(db_session).create("paper.codegen", input_ref="queue-test")
    with patch(
        "backend.routers.papers_pipeline.generate_code_from_pdf_task.apply_async"
    ) as publish:
        asyncio.run(_enqueue_paper(db_session, task, "input.pdf", "paper", 1, "private", True))
    publish.assert_called_once_with(
        args=(task.id, "input.pdf", "paper", 1, "private", True),
        task_id=task.id,
        ignore_result=True,
        retry=False,
    )


def test_dispatch_does_not_block_event_loop(db_session):
    import threading

    task = TaskRepository(db_session).create("paper.codegen", input_ref="queue-test")
    released = threading.Event()

    def publish(**kwargs):
        assert released.wait(2), "Publisher blocked the event loop"

    async def run():
        asyncio.get_running_loop().call_later(0.05, released.set)
        await _enqueue_paper(db_session, task, "input.pdf")

    with patch(
        "backend.routers.papers_pipeline.generate_code_from_pdf_task.apply_async",
        side_effect=publish,
    ):
        asyncio.run(run())
