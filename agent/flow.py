from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes for task flows."""
    SEQUENTIAL = "sequential"
    DAG = "dag"  # For future implementation
    STATE_MACHINE = "state_machine"  # For future implementation


@dataclass
class Task:
    """A single task in the flow."""
    id: str
    action: str
    params: dict = None
    dependencies: list[str] = None  # For DAG mode
    on_success: Optional[str] = None  # For state machine mode
    on_failure: Optional[str] = None  # For state machine mode
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.dependencies is None:
            self.dependencies = []


class TaskResult:
    """Result of a task execution."""
    def __init__(self, task_id: str, success: bool, output: Any = None, error: str = None):
        self.task_id = task_id
        self.success = success
        self.output = output
        self.error = error
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        return f"TaskResult({self.task_id}: {status})"


class AgentController(ABC):
    """Abstract base class for agent controllers."""
    
    @abstractmethod
    def execute(self, action: str, params: dict) -> TaskResult:
        """Execute an action with given parameters."""
        pass


class ExecutionStrategy(ABC):
    """Abstract strategy for task execution."""
    
    @abstractmethod
    def execute(self, tasks: list[Task], controller: AgentController, 
                context: dict) -> list[TaskResult]:
        """Execute tasks according to the strategy."""
        pass


class SequentialStrategy(ExecutionStrategy):
    """Execute tasks one after another."""
    
    def execute(self, tasks: list[Task], controller: AgentController,
                context: dict) -> list[TaskResult]:
        results = []
        
        for task in tasks:
            # Merge context into params
            params = {**context, **task.params}
            
            # Execute task
            result = controller.execute(task.action, params)
            result.task_id = task.id
            results.append(result)
            
            # Update context with result
            context[f"{task.id}_output"] = result.output
            
            # Stop on failure (can be made configurable)
            if not result.success:
                break
        
        return results


class DAGStrategy(ExecutionStrategy):
    """Execute tasks based on dependency graph (future implementation)."""
    
    def execute(self, tasks: list[Task], controller: AgentController,
                context: dict) -> list[TaskResult]:
        raise NotImplementedError("DAG execution will be implemented in future")


class StateMachineStrategy(ExecutionStrategy):
    """Execute tasks as a state machine (future implementation)."""
    
    def execute(self, tasks: list[Task], controller: AgentController,
                context: dict) -> list[TaskResult]:
        raise NotImplementedError("State machine execution will be implemented in future")


class TaskFlow:
    """
    Orchestrates task execution using pluggable strategies.
    
    Supports sequential execution now, with extension points for
    DAG and state machine patterns.
    """
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.mode = mode
        self.tasks: list[Task] = []
        self.context: dict = {}
        self.strategies = {
            ExecutionMode.SEQUENTIAL: SequentialStrategy(),
            ExecutionMode.DAG: DAGStrategy(),
            ExecutionMode.STATE_MACHINE: StateMachineStrategy()
        }
    
    def add_task(self, task: Task) -> 'TaskFlow':
        """Add a task to the flow (builder pattern)."""
        self.tasks.append(task)
        return self
    
    def set_context(self, key: str, value: Any) -> 'TaskFlow':
        """Set a context variable available to all tasks."""
        self.context[key] = value
        return self
    
    def execute(self, controller: AgentController) -> list[TaskResult]:
        """
        Execute all tasks using the configured strategy.
        
        Args:
            controller: The agent controller to execute tasks
            
        Returns:
            List of task results
        """
        strategy = self.strategies[self.mode]
        return strategy.execute(self.tasks, controller, self.context.copy())
    
    def clear(self):
        """Clear all tasks and context."""
        self.tasks.clear()
        self.context.clear()


# Example implementation of AgentController
class SimpleAgentController(AgentController):
    """Example controller for demonstration."""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.action_handlers: dict[str, Callable] = {}
    
    def register_action(self, action: str, handler: Callable):
        """Register an action handler."""
        self.action_handlers[action] = handler
    
    def execute(self, action: str, params: dict) -> TaskResult:
        """Execute an action."""
        if self.memory:
            self.memory.record(f"Action: {action}", "started", **params)
        
        try:
            if action not in self.action_handlers:
                raise ValueError(f"Unknown action: {action}")
            
            output = self.action_handlers[action](params)
            
            if self.memory:
                self.memory.record(f"Action: {action}", "completed", result=output)
            
            return TaskResult(task_id="", success=True, output=output)
        
        except Exception as e:
            error_msg = str(e)
            if self.memory:
                self.memory.record(f"Action: {action}", "failed", error=error_msg)
            
            return TaskResult(task_id="", success=False, error=error_msg)


# Example usage
if __name__ == "__main__":
    # Create controller with action handlers
    controller = SimpleAgentController()
    
    controller.register_action("fetch_data", 
        lambda p: {"data": f"Fetched from {p.get('source', 'unknown')}"})
    controller.register_action("process_data",
        lambda p: {"processed": f"Processed {p.get('fetch_data_output', {})}"})
    controller.register_action("save_result",
        lambda p: {"saved": True, "location": p.get('destination', 'default')})
    
    # Create task flow
    flow = TaskFlow(mode=ExecutionMode.SEQUENTIAL)
    
    flow.add_task(Task(
        id="fetch",
        action="fetch_data",
        params={"source": "api.example.com"}
    )).add_task(Task(
        id="process",
        action="process_data"
    )).add_task(Task(
        id="save",
        action="save_result",
        params={"destination": "/tmp/output.json"}
    ))
    
    # Set shared context
    flow.set_context("user_id", "12345")
    
    # Execute
    print("=== Executing Task Flow ===")
    results = flow.execute(controller)
    
    for result in results:
        print(f"\n{result}")
        if result.success:
            print(f"  Output: {result.output}")
        else:
            print(f"  Error: {result.error}")
    
    # Demonstrate extensibility
    print("\n=== Flow Structure ===")
    print(f"Mode: {flow.mode.value}")
    print(f"Tasks: {len(flow.tasks)}")
    for task in flow.tasks:
        print(f"  - {task.id}: {task.action}")
        if task.dependencies:
            print(f"    Dependencies: {task.dependencies}")
