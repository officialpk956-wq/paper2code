import time
import pytest
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.modules.jobs.models import BackgroundJob
from backend.modules.jobs.queue import RedisQueue
from backend.modules.jobs.retry import calculate_backoff_delay, handle_job_failure
from backend.modules.jobs.event_bus import EventBus
from backend.modules.jobs.workflow import WorkflowOrchestrator
from backend.modules.jobs.scheduler import Scheduler
from backend.modules.jobs.worker import WorkerPool

def test_backoff_delay_calculation():
    # Test delay bounds
    d1 = calculate_backoff_delay(0)
    assert 1.8 <= d1 <= 2.2 # 2.0 base +/- 10%
    
    d2 = calculate_backoff_delay(1)
    assert 3.6 <= d2 <= 4.4 # 4.0 base +/- 10%

    d_max = calculate_backoff_delay(10, max_delay=10.0)
    assert 9.0 <= d_max <= 11.0

def test_event_bus_publishing():
    bus = EventBus()
    received_data = []
    
    def dummy_handler(event):
        received_data.append(event.data)
        
    bus.subscribe("UserRegistered", dummy_handler)
    bus.publish("UserRegistered", "v1", {"email": "event@example.com"})
    
    assert len(received_data) == 1
    assert received_data[0]["email"] == "event@example.com"

def test_workflow_orchestration_success_and_compensation():
    orchestrator = WorkflowOrchestrator("UploadWorkflow")
    
    step1_called = False
    step2_called = False
    comp1_called = False
    
    def step1_action(ctx):
        nonlocal step1_called
        step1_called = True
        ctx["step1"] = "done"
        return ctx

    def step1_compensation(ctx):
        nonlocal comp1_called
        comp1_called = True

    def step2_action_fail(ctx):
        nonlocal step2_called
        step2_called = True
        raise ValueError("Failed on step 2")

    orchestrator.add_step("Step1", step1_action, step1_compensation)
    orchestrator.add_step("Step2", step2_action_fail)

    with pytest.raises(ValueError):
        orchestrator.execute({"initial": True})

    assert step1_called is True
    assert step2_called is True
    assert comp1_called is True # Step 1 rolled back successfully

def test_scheduler_interval_tasks():
    scheduler = Scheduler()
    called = 0
    
    def dummy_action():
        nonlocal called
        called += 1

    scheduler.schedule_interval("token_cleanup", 5, dummy_action, run_once=True)
    scheduler.trigger_run_now()
    assert called == 1

def test_worker_pool_job_execution(db_session: Session, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    import backend.modules.jobs.worker
    
    test_session_maker = sessionmaker(bind=db_session.bind)
    monkeypatch.setattr(backend.modules.jobs.worker, "SessionLocal", test_session_maker)

    queue = RedisQueue()
    pool = WorkerPool(queue, "test_queue", concurrency=1)

    
    job_payload = {"input": "test"}
    handled_payload = None
    
    def test_handler(payload, job, db):
        nonlocal handled_payload
        handled_payload = payload
        job.progress_pct = 50
        db.commit()

    pool.register_handler("test_job", test_handler)
    
    # Enqueue job
    job = pool.create_and_enqueue_job(db_session, "test_job", job_payload)
    assert job.status == "Queued"
    
    # Process job synchronously using worker loop single step run
    task = queue.dequeue("test_queue")
    assert task is not None
    job_id, payload = task
    
    pool._execute_job(job_id, payload)
    
    # Refetch from DB to confirm status Succeeded
    db_session.expire_all()
    db_job = db_session.get(BackgroundJob, job.id)
    assert db_job.status == "Succeeded"
    assert db_job.progress_pct == 100
    assert handled_payload == job_payload
