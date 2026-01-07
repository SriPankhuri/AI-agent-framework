import uuid
from typing import Dict, Any, List
from agent.planner import Planner
from agent.memory import Memory
from agent.flow import TaskFlow
from tools.tool_registry import ToolRegistry
from observability.logger import AgentLogger

class AgentController:
    """
    Orchestrates the agentic workflow. 
    Actively manages the state transition between the Planner and the Executors.
    """
    def __init__(self, llm_client, storage_type="apache_sqlite"):
        self.controller_id = str(uuid.uuid4())
        self.logger = AgentLogger(name=f"Controller-{self.controller_id}")
        
        # Dependencies
        self.llm = llm_client
        self.memory = Memory(backend=storage_type) # Connects to storage (Apache Cassandra/SQLite)
        self.planner = Planner(llm_client=self.llm)
        self.tools = ToolRegistry()
        
        self.logger.info("Framework Controller initialized and ready for ingress.")

    def execute_workflow(self, workflow: TaskFlow, initial_input: str) -> Dict[str, Any]:
        """
        Executes a composed task flow.
        Satisfies the requirement: "Orchestrate agentic workflows from input to output."
        """
        session_id = str(uuid.uuid4())
        self.logger.info(f"Execution started. Session: {session_id} | Flow: {workflow.name}")
        
        # 1. Store initial task in persistent memory for auditing
        self.memory.initialize_session(session_id, initial_input)

        # 2. Planning / DAG Validation
        # The planner can dynamically adjust the workflow based on the input
        executable_plan = self.planner.prepare_steps(workflow, initial_input)
        
        results = {}

        # 3. Execution Loop (The Orchestration Heart)
        for task in executable_plan:
            # Check dependencies (DAG Logic)
            if not self._check_dependencies(task, results):
                self.logger.error(f"Task {task.name} blocked by dependency failure.")
                break
                
            # Execute logic
            output = self._run_task_unit(task, session_id, results)
            results[task.name] = output

            if output.get("status") == "error":
                self.logger.warning(f"Aborting flow at {task.name} due to error.")
                break

        # 4. Final Output Action
        final_response = self.llm.synthesize(initial_input, results)
        
        # 5. Finalize Audit Trail
        self.memory.close_session(session_id, final_response)
        
        return {
            "session_id": session_id,
            "status": "completed",
            "output": final_response
        }

    def _run_task_unit(self, task: Any, session_id: str, previous_results: Dict) -> Dict:
        """
        Internal executor logic for a single unit of work.
        """
        self.logger.info(f"Routing task '{task.name}' to tool '{task.tool_name}'")
        
        try:
            # Context injection from memory and previous steps
            context = self.memory.get_session_context(session_id)
            
            # Execute tool call
            tool_result = self.tools.execute(
                name=task.tool_name, 
                args=task.args, 
                context=context,
                history=previous_results
            )
            
            # Observability: Log result to persistent store
            self.memory.log_step(session_id, task.name, tool_result)
            
            return {"status": "success", "data": tool_result}
            
        except Exception as e:
            self.logger.error(f"Execution error in {task.name}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _check_dependencies(self, task: Any, results: Dict) -> bool:
        """Verifies if all parent tasks in the DAG completed successfully."""
        for dep in task.depends_on:
            if dep not in results or results[dep].get("status") != "success":
                return False
        return True
