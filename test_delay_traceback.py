import os
import traceback
# Mock Render environment
os.environ["REDIS_URL"] = "rediss://red-xxxxxxxxxx:xxxxxxxxxxxx@singapore-redis.render.com:6379"

from backend.tasks.paper_tasks import generate_code_from_pdf_task

try:
    generate_code_from_pdf_task.delay("task-1", "fake-ref", "fake-paper", 1, "public", False)
except Exception as e:
    print(f"EXCEPTION RAISED: {e}")
    traceback.print_exc()
