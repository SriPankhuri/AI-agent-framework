import time
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from observability.logger import AgentLogger

class ToolType(Enum):
    WEB = "web"
    FILE = "file"
    DATA = "data"
    ML_INFERENCE = "ml_inference" # For Intel OpenVINO tools
    SYSTEM = "system"

@dataclass
class Tool:
    name: str
    type: ToolType
    description: str
    handler: Callable
    keywords: List[str]
    required_params: List[str] = field(default_factory=list)
    # Metadata for Intel-specific tagging (e.g., "optimized": True)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ToolResult:
    def __init__(self, tool_name: str, success: bool, output: Any = None, 
                 error: str = None, latency_ms: float = 0.0):
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error
        self.latency_ms = latency_ms

    def __repr__(self):
        return f"ToolResult({self.tool_name}: {'SUCCESS' if self.success else 'FAILURE'} in {self.latency_ms:.2f}ms)"

class ToolRegistry:
    """
    Registry and Execution Engine for Agent Tools.
    Includes performance monitoring for Intel DevCloud benchmarking.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.logger = AgentLogger(name="ToolRegistry")
        self.usage_history: List[Dict] = []

    def register(self, tool: Tool) -> 'ToolRegistry':
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name} ({tool.type.value})")
        return self

    def execute(self, tool_name: str, params: Dict) -> ToolResult:
        """
        Executes a tool and measures performance.
        Satisfies the 'monitor and audit' and 'performance targets' requirements.
        """
        if tool_name not in self.tools:
            return ToolResult(tool_name, False, error=f"Tool {tool_name} not found")

        tool = self.tools[tool_name]
        
        # Validation
        missing = [p for p in tool.required_params if p not in params]
        if missing:
            return ToolResult(tool_name, False, error=f"Missing params: {missing}")

        # Execution with Latency Benchmarking
        start_time = time.perf_counter()
        try:
            output = tool.handler(params)
            latency = (time.perf_counter() - start_time) * 1000
            
            result = ToolResult(tool_name, True, output=output, latency_ms=latency)
            self._log_usage(result, params, tool.metadata)
            return result
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            result = ToolResult(tool_name, False, error=str(e), latency_ms=latency)
            self._log_usage(result, params, tool.metadata)
            return result

    def _log_usage(self, result: ToolResult, params: Dict, metadata: Dict):
        """Internal helper for auditing and benchmarking."""
        log_entry = {
            "tool": result.tool_name,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "intel_optimized": metadata.get("optimized", False),
            "timestamp": time.time()
        }
        self.usage_history.append(log_entry)
        
        # Log to the primary observability system
        if log_entry["intel_optimized"]:
            self.logger.log_intel_metric(result.tool_name, result.latency_ms, metadata.get("device", "CPU"))
        else:
            self.logger.info(f"Tool {result.tool_name} finished in {result.latency_ms:.2f}ms")

    def get_benchmarks(self) -> Dict:
        """Returns data for the Design Doc / Benchmark deliverable."""
        if not self.usage_history: return {}
        
        optimized = [l for l in self.usage_history if l["intel_optimized"]]
        standard = [l for l in self.usage_history if not l["intel_optimized"]]
        
        return {
            "avg_latency_optimized": sum(l["latency_ms"] for l in optimized) / len(optimized) if optimized else 0,
            "avg_latency_standard": sum(l["latency_ms"] for l in standard) / len(standard) if standard else 0,
            "total_calls": len(self.usage_history)
        }
