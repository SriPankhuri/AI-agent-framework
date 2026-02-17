import json
import sqlite3 # Base for Apache-style persistence
from datetime import datetime
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field, asdict

@dataclass
class MemoryEntry:
    """A single memory entry representing a task execution."""
    session_id: str
    task_id: str
    action: str
    status: str  
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)

class Memory:
    """
    State Management & Audit Log.
    Optimized for persistent storage to satisfy Apache-ready requirements.
    """
    def __init__(self, backend_type: str = "sqlite"):
        self.backend_type = backend_type
        # In-memory cache for speed during execution
        self.entries: List[MemoryEntry] = []
        
        # In a real Apache setup, you'd initialize a Cassandra or HBase connection here
        self._init_db()

    def _init_db(self):
        """Initialize the persistent audit log table."""
        if self.backend_type == "sqlite":
            with sqlite3.connect("agent_audit_trail.db") as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        session_id TEXT, task_id TEXT, action TEXT, 
                        status TEXT, result TEXT, error TEXT, 
                        timestamp TEXT, metadata TEXT
                    )
                """)

    def record(self, session_id: str, task_id: str, action: str, status: str, 
               result: Any = None, error: str = None, **metadata) -> MemoryEntry:
        """
        Records an execution step both in memory and in the persistent DB.
        This provides the 'Audit' capability required by the framework.
        """
        entry = MemoryEntry(
            session_id=session_id,
            task_id=task_id,
            action=action,
            status=status,
            result=result,
            error=error,
            metadata=metadata
        )
        
        self.entries.append(entry)
        self._persist_to_disk(entry)
        return entry

    def _persist_to_disk(self, entry: MemoryEntry):
        """Standardizing data for Apache-ready storage (JSON strings)."""
        if self.backend_type == "sqlite":
            with sqlite3.connect("agent_audit_trail.db") as conn:
                conn.execute(
                    "INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry.session_id, entry.task_id, entry.action, entry.status, 
                     json.dumps(entry.result), entry.error, entry.timestamp, json.dumps(entry.metadata))
                )

    def get_session_history(self, session_id: str) -> List[MemoryEntry]:
        """Retrieves context for the current agentic loop."""
        return [e for e in self.entries if e.session_id == session_id]

    def get_summary(self, session_id: Optional[str] = None) -> dict:
        """Generates performance metrics for benchmarking."""
        target_list = [e for e in self.entries if e.session_id == session_id] if session_id else self.entries
        
        if not target_list:
            return {'total': 0, 'success_rate': 0}
        
        completed = sum(1 for e in target_list if e.status == 'completed')
        return {
            'total_steps': len(target_list),
            'completed': completed,
            'failed': sum(1 for e in target_list if e.status == 'failed'),
            'success_rate': (completed / len(target_list)) * 100
        }

    def clear_session_cache(self):
        """Clears RAM while keeping the persistent audit trail on disk."""
        self.entries.clear()

    # --- Compatibility layer for controller ---

    def initialize_session(self, session_id: str, initial_input: str):
        """Start a session"""
        self.record(session_id, "session_start", "input", "started", result=initial_input)

    def log_step(self, session_id: str, task_name: str, result: dict):
        """Log each step execution"""
        status = result.get("status", "unknown")
        self.record(session_id, task_name, "tool_execution", status, result=result)

    def close_session(self, session_id: str, final_output: str):
        """Close session"""
        self.record(session_id, "session_end", "output", "completed", result=final_output)

    def get_session_context(self, session_id: str) -> dict:
        """Provide previous steps as context"""
        history = self.get_session_history(session_id)
        return {entry.task_id: entry.result for entry in history if entry.result is not None}
