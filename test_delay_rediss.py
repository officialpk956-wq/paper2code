import os
os.environ["REDIS_URL"] = "rediss://localhost:6379/0"
os.environ["CELERY_RESULT_BACKEND"] = "rediss://localhost:6379/1"

import backend.celery_app

from backend.tasks.paper_tasks import generate_code_from_pdf_task

print("Calling delay...")
try:
    generate_code_from_pdf_task.delay(1, "dummy.pdf", "dummy", 1, "public", True)
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
