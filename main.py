import sys
from llm.llm_client import HFLocalLLM
from agent.controller import AgentController
from agent.flow import TaskFlow, Task
from tools.tool_registry import ToolRegistry
from observability.logger import AgentLogger

def main():
    # 1. Setup Observability
    logger = AgentLogger(name="Framework-Demo")
    logger.info("Initializing AI Agent Framework...")

    # 2. Initialize Optimized LLM (Intel OpenVINO)
    # This uses the local Intel-optimized model defined in your llm_client

    llm = HFLocalLLM()



    # 3. Define the Agentic Workflow (DAG-based Composition)
    # We define a flow: Web Search -> Information Extraction -> Final Report
    research_flow = TaskFlow(name="MarketResearchFlow")
    
    research_flow.add_task(Task(
        id="market_research",
        action="llm_tool",
        params={"query": "Research the AI market"}
    ))

    research_flow.add_task(Task(
        id="summary",
        action="llm_tool",
        params={"query": "Summarize the research"}
    ))


    # 4. Initialize the Orchestrator
    # The controller handles the Apache-based state/memory and execution logic
    controller = AgentController(llm_client=llm)


    # 5. Execute the Workflow
    print("\n--- Starting Agentic Workflow ---\n")
    try:
        user_goal = "Analyze current AI market trends"
        result = controller.execute_workflow( research_flow,"Research the AI market and summarize it")


        
        print("\n--- Workflow Completed Successfully ---")
        print(f"Session ID: {result['session_id']}")
        print(f"Final Report: {result['output']}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
