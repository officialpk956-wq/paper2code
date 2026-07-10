from typing import Any

from sqlalchemy.orm import Session

from backend.models import Task


class TaskRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, type: str, user_id: int | None = None, input_ref: str | None = None) -> Task:
        task = Task(type=type, user_id=user_id, input_ref=input_ref, status="pending")
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get(self, task_id: str) -> Task | None:
        return self.db.query(Task).filter(Task.id == task_id).first()

    def set_running(self, task_id: str) -> None:
        task = self.get(task_id)
        if task:
            task.status = "running"
            self.db.commit()

    def set_complete(self, task_id: str, result_dict: dict[str, Any]) -> None:
        task = self.get(task_id)
        if task:
            task.status = "completed"
            task.result = result_dict
            self.db.commit()

    def set_failed(self, task_id: str, error_str: str) -> None:
        task = self.get(task_id)
        if task:
            task.status = "failed"
            task.error = error_str
            self.db.commit()

    def set_stage(self, task_id: str, stage: str) -> None:
        """Emit a pipeline stage label into task.result while the task is still running.
        Frontend polls /api/tasks/{id} and reads result.stage for progress display."""
        task = self.get(task_id)
        if task:
            current = dict(task.result) if task.result else {}
            current["stage"] = stage
            task.result = current
            self.db.commit()
