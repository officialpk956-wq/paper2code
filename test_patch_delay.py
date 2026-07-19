from backend.celery_app import celery_app
from backend.tasks.paper_tasks import generate_code_from_pdf_task

try:
    generate_code_from_pdf_task.delay("task-1", "fake-ref", "fake-paper", 1, "public", False)
    print("SUCCESSFULLY CALLED DELAY WITH MONKEY PATCH!")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
