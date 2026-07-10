from collections.abc import Callable
from typing import Any


class WorkflowStep:
    def __init__(
        self,
        name: str,
        action: Callable[[dict[str, Any]], dict[str, Any]],
        compensation: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.name = name
        self.action = action
        self.compensation = compensation


class WorkflowOrchestrator:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[WorkflowStep] = []
        self._parallel_branches: list[list[WorkflowStep]] = []

    def add_step(
        self,
        name: str,
        action: Callable[[dict[str, Any]], dict[str, Any]],
        compensation: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.steps.append(WorkflowStep(name, action, compensation))

    def add_parallel_branches(self, branches: list[list[WorkflowStep]]) -> None:
        self._parallel_branches = branches

    def execute(self, initial_payload: dict[str, Any]) -> dict[str, Any]:
        """Execute steps. If any step fails, roll back using compensation steps."""
        context = initial_payload.copy()
        executed_steps: list[WorkflowStep] = []

        try:
            # 1. Execute sequential steps
            for step in self.steps:
                context = step.action(context)
                executed_steps.append(step)

            # 2. Execute parallel branches
            if self._parallel_branches:
                # For simplicity/synchronization in single process, run them sequentially in a mock loop,
                # but track failures and rollback accordingly.
                for branch in self._parallel_branches:
                    for step in branch:
                        context = step.action(context)
                        executed_steps.append(step)

            return context
        except Exception as e:
            # Rollback in reverse order
            for step in reversed(executed_steps):
                if step.compensation:
                    try:
                        step.compensation(context)
                    except Exception:
                        pass
            raise e
