import uuid
import logging
from typing import Dict, Any, List
from agent.planner import Planner
from agent.memory import Memory
from tools.tool_registry import ToolRegistry
from observability.logger import AgentLogger

class AgentController:
    """
    The Orchestrator responsible for managing the lifecycle of an agentic workflow.
    It handles input processing, state management, and tool execution loops.
    """
    def __init__(self, llm_client, memory_backend="sqlite"):
        self.id = str(uuid.uuid4())
        self.logger = AgentLogger(name=f"Controller-{self.id}")
        
        # Core Components
        self.llm = llm_client
        self.memory = Memory(backend=memory_backend)
        self.planner = Planner(llm_client=self.llm)
        self.tools = ToolRegistry()
        
        self.logger.info("Agent Controller Initialized.")

    def execute_workflow(self, task_input: str) -> Dict[str, Any]:
        """
        Main entry point to execute a task flow from input to output.
        """
        self.logger.info(f"Starting workflow for task: {task_input}")
        
        # 1. Initialize State/Memory
        session_id = self.memory.create_session(task_input)
        
        # 2. Planning Phase
        # The planner decides the DAG or sequence of steps
        plan = self.planner.generate_plan(task_input)
        self.logger.info(f"Plan generated with {len(plan['steps'])} steps.")

        # 3. Execution Loop (Orchestration)
        results = []
        for step in plan['steps']:
            step_result = self._execute_step(step, session_id)
            results.append(step_result)
            
            # Check for Guardrails / Break conditions
            if step_result.get("status") == "failed":
                self.logger.error(f"Workflow halted at step: {step['name']}")
                break

        # 4. Final Output Synthesis
        final_output = self.llm.generate_summary(task_input, results)
        
        # 5. Audit & Persistence
        self.memory.finalize_session(session_id, final_output)
        return {"session_id": session_id, "output": final_output}

    def _execute_step(self, step: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Orchestrates a single task: Tool Selection -> Execution -> Memory Update
        """
        step_name = step.get("name")
        tool_name = step.get("tool")
        tool_input = step.get("args")

        self.logger.info(f"Executing Step: {step_name} using tool: {tool_name}")

        try:
            # Retrieve context from memory
            context = self.memory.get_context(session_id)
            
            # Execute the tool
            tool_output = self.tools.call(tool_name, tool_input, context)
            
            # Update memory with tool results (Observability/Audit)
            self.memory.update_log(session_id, step_name, tool_output)
            
            return {"step": step_name, "status": "success", "data": tool_output}
        
        except Exception as e:
            self.logger.error(f"Step {step_name} failed: {str(e)}")
            return {"step": step_name, "status": "failed", "error": str(e)}
