import sys
from llm.llm_client import OpenVINOClient
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
    llm = OpenVINOClient(model_id="llama-3-8b-ov", device="CPU")

    # 3. Define the Agentic Workflow (DAG-based Composition)
    # We define a flow: Web Search -> Information Extraction -> Final Report
    research_flow = TaskFlow(name="MarketResearchFlow")
    
    research_flow.add_task(Task(
        name="gather_data",
        tool="web_search",
        input_query="Intel Gaudi 3 AI accelerator benchmarks 2024"
    ))
    
    research_flow.add_task(Task(
        name="analyze_specs",
        tool="llm_analyzer",
        depends_on="gather_data"
    ))

    # 4. Initialize the Orchestrator
    # The controller handles the Apache-based state/memory and execution logic
    controller = AgentController(
        llm_client=llm,
        memory_backend="persistent_storage"  # e.g., Apache Cassandra/SQLite
    )

    # 5. Execute the Workflow
    print("\n--- Starting Agentic Workflow ---\n")
    try:
        result = controller.execute_workflow(research_flow)
        
        print("\n--- Workflow Completed Successfully ---")
        print(f"Session ID: {result['session_id']}")
        print(f"Final Report: {result['output']}")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
