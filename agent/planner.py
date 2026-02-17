import json
from typing import List, Dict, Any
from agent.flow import Task, ExecutionMode
# Use the LLM Client we'll optimize with OpenVINO
from llm.llm_client import HFLocalLLM


class Planner:
    """
    The Brain of the Agent. 
    Uses LLM reasoning to decompose goals into executable Task objects.
    """
    def __init__(self,  llm_client):
        self.llm = llm_client
        self.system_prompt = (
            "You are a Task Decomposition Engine. Break the user's goal into a "
            "structured JSON list of tasks. Each task must have: "
            "'id', 'action' (tool name), and 'params' (input for tool)."
        )

    def generate_plan(self, goal: str, mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> List[Task]:
        """
        Calls the LLM to create a dynamic plan based on the user's goal.
        """
        # 1. Construct the reasoning prompt
        prompt = f"{self.system_prompt}\n\nGoal: {goal}\n\nReturn JSON format only."

        # 2. Get LLM response (This will be the OpenVINO-optimized call)
        raw_response = self.llm.generate(prompt)
        
        # 3. Parse LLM output into framework-compatible Task objects
        try:
            plan_data = self._parse_json(raw_response)
            tasks = []
            for item in plan_data.get("tasks", []):
                tasks.append(Task(
                    id=item['id'],
                    action=item['action'],
                    params=item.get('params', {}),
                    dependencies=item.get('dependencies', [])
                ))
            return tasks
        except Exception as e:
            # Fallback to a basic template if LLM fails (Guardrail)
            print(f"Planning failed, using fallback: {e}")
            return self._get_fallback_plan(goal)

    def _parse_json(self, text: str) -> Dict:
        """Helper to extract JSON from LLM prose."""
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])

    def _get_fallback_plan(self, goal: str) -> List[Task]:
        """Ensures the agent is reliable even if the LLM output is malformed."""
        return [
            Task(id="analysis", action="llm_tool", params={"query": f"Analyze: {goal}"}),
            Task(id="summary", action="report_tool", dependencies=["analysis"])
        ]
    

    # --- Compatibility wrapper for controller ---
    def prepare_steps(self, workflow, user_input):
        """
        Returns the tasks in execution order.
        Currently no dynamic planning — just linear execution.
        """
        return workflow.tasks
