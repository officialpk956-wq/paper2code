from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.repositories.task_repository import TaskRepository

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])

@router.get("/{task_id}")
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    task = TaskRepository(db).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not getattr(current_user, "is_admin", False):
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
    return {
        "id": task.id,
        "status": task.status,
        "type": task.type,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
