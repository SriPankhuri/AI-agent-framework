from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import collections

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    DAG = "dag"
    STATE_MACHINE = "state_machine"

@dataclass
class Task:
    """
    Unified Task model compatible with AgentController
    """

    def __init__(self, id, action, params=None, depends_on=None):
        self.id = id

        # Controller expects these names
        self.name = id
        self.tool_name = action
        self.args = params or {}
        self.depends_on = depends_on or []

class TaskResult:
    """Audit-friendly result container."""
    def __init__(self, task_id: str, success: bool, output: Any = None, error: str = None):
        self.task_id = task_id
        self.success = success
        self.output = output
        self.error = error

    def __repr__(self):
        return f"TaskResult({self.task_id}: {'SUCCESS' if self.success else 'FAILURE'})"

# --- Strategies ---

class ExecutionStrategy(ABC):
    @abstractmethod
    def execute(self, tasks: List[Task], controller: Any, context: dict) -> List[TaskResult]:
        pass

class SequentialStrategy(ExecutionStrategy):
    def execute(self, tasks: List[Task], controller: Any, context: dict) -> List[TaskResult]:
        results = []
        for task in tasks:
            # Context injection: merge global context and results from previous tasks
            params = {**context, **task.params}
            result = controller.execute(task.action, params)
            result.task_id = task.id
            results.append(result)
            
            if not result.success: break
            context[f"{task.id}_output"] = result.output
        return results

class DAGStrategy(ExecutionStrategy):
    """
    Implementation of Directed Acyclic Graph execution.
    Fulfills the requirement for non-linear task flows.
    """
    def execute(self, tasks: List[Task], controller: Any, context: dict) -> List[TaskResult]:
        results_map = {}
        # Simple topological sort/dependency resolution
        pending = tasks.copy()
        final_results = []

        while pending:
            # Find tasks with met dependencies
            ready = [t for t in pending if all(d in results_map for d in t.dependencies)]
            if not ready:
                if pending: raise Exception("Circular dependency or missing task detected in DAG")
                break
            
            for task in ready:
                params = {**context, **task.params}
                # Inject dependency data into params
                params["_dep_results"] = {d: results_map[d].output for d in task.dependencies}
                
                result = controller.execute(task.action, params)
                result.task_id = task.id
                results_map[task.id] = result
                final_results.append(result)
                pending.remove(task)
                
                if not result.success: return final_results
        
        return final_results

# --- Core Flow ---

class TaskFlow:
    """
    The main SDK entry point for defining agentic compositions.
    """
    def __init__(self, name: str, mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.name = name
        self.mode = mode
        self.tasks: List[Task] = []
        self.context: Dict[str, Any] = {}
        self._strategies = {
            ExecutionMode.SEQUENTIAL: SequentialStrategy(),
            ExecutionMode.DAG: DAGStrategy()
        }

    def add_task(self, task: Task) -> 'TaskFlow':
        self.tasks.append(task)
        return self

    def set_context(self, key: str, value: Any) -> 'TaskFlow':
        self.context[key] = value
        return self

    def execute(self, controller: Any) -> List[TaskResult]:
        # Log to observability via controller
        controller.logger.info(f"Starting Flow: {self.name} in {self.mode.value} mode")
        strategy = self._strategies[self.mode]
        return strategy.execute(self.tasks, controller, self.context.copy())
