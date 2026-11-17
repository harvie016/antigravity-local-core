"""
core/agents/models.py
Pydantic v2 data models for the Multi-Agent Subsystem.

Defines:
- AgentRoleType: Canonical role types with alias support (Researcher <-> Retriever).
- ToolParameter / ToolParameterField: Parameter descriptor for agent tools.
- ToolDefinition: Declarative specification of agent tools.
- ToolCallRequest, ToolExecutionResult: Invocation request and response models.
- AgentRoleDefinition: Role profile specification, allowed tools, prompts, budgets.
- HandoffPacket: 5-Component self-contained handoff protocol report.
- AgentMessageType, AgentMessage: Message envelope for agent communication.
- SubtaskStatus, SubtaskNode, TaskDAG: Directed Acyclic Graph for task decomposition
  with Kahn's algorithm cycle detection and topological sorting.
"""

from __future__ import annotations
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from collections import deque, defaultdict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class AgentRoleType(str, Enum):
    """
    Canonical role types supported across the multi-agent system.
    Supports string coercion and alias resolution (e.g., 'researcher' -> 'retriever').
    """
    ORCHESTRATOR = "orchestrator"
    RETRIEVER = "retriever"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESEARCHER = "retriever"  # Direct enum member alias

    @classmethod
    def _missing_(cls, value: Any) -> Optional[AgentRoleType]:
        if isinstance(value, str):
            clean = value.lower().strip()
            if clean in ("researcher", "research"):
                return cls.RETRIEVER
            for member in cls:
                if member.value == clean:
                    return member
        return None


class ToolParameter(BaseModel):
    """
    Metadata specification for an individual tool parameter.
    Compatible with JSON Schema and MCP parameter definitions.
    """
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Parameter argument name")
    type: str = Field("string", description="Parameter data type: string, integer, number, boolean, object, array")
    description: str = Field("", description="Human-readable explanation of parameter purpose and semantics")
    required: bool = Field(default=True, description="Whether parameter is mandatory")
    default: Optional[Any] = Field(default=None, description="Default value if parameter is omitted")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed literal choices if constrained")


# Backward compatibility alias
ToolParameterField = ToolParameter


class ToolDefinition(BaseModel):
    """
    Declarative specification of an agent-executable tool.
    Includes parameter schema, execution constraints, and optional callable handler.
    """
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique tool identifier, e.g. 'read_file', 'rag_search'")
    description: str = Field(..., description="Detailed description of tool functionality for model prompt context")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Ordered list of accepted parameters")
    read_only: bool = Field(default=True, description="Whether tool performs read-only operations without side effects")
    timeout_seconds: Optional[float] = Field(default=30.0, description="Max execution duration before timeout guard")
    max_output_length: Optional[int] = Field(default=800, description="Max output length in characters before truncation")
    path_params: List[str] = Field(default_factory=list, description="Names of parameters requiring path jail verification")
    permission_scope: Optional[str] = Field(default=None, description="Granular permission category (fs, rag, memory, eval)")
    handler: Optional[Callable[..., Any]] = Field(default=None, exclude=True, description="Underlying callable implementation")

    @property
    def func(self) -> Optional[Callable[..., Any]]:
        """Alias for handler callable."""
        return self.handler



class ToolCallRequest(BaseModel):
    """Invocation request emitted by an agent to execute a registered tool."""
    model_config = ConfigDict(extra="ignore")

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique invocation UUID")
    tool_name: str = Field(..., description="Name of tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Key-value arguments matching tool parameters")
    caller_role: Optional[AgentRoleType] = Field(default=None, description="Role of the calling agent")


class ToolExecutionResult(BaseModel):
    """Execution telemetry and payload returned after invoking a tool."""
    model_config = ConfigDict(extra="ignore")

    call_id: str = Field(..., description="Matching invocation call ID")
    tool_name: str = Field(..., description="Name of executed tool")
    status: str = Field(..., description="Status indicator: 'ok' | 'error'")
    output: Any = Field(default=None, description="Execution payload or string output")
    error_message: Optional[str] = Field(default=None, description="Descriptive error explanation if status is 'error'")
    execution_time_ms: float = Field(default=0.0, description="Wall-clock execution duration in milliseconds")
    truncated: bool = Field(default=False, description="Flag indicating output was truncated to prevent context overflow")


class AgentRoleDefinition(BaseModel):
    """
    Specification envelope defining a canonical agent role:
    capabilities, system prompts, allowed tools, turn limits, and token budgets.
    """
    model_config = ConfigDict(extra="ignore")

    role_type: AgentRoleType = Field(..., description="Canonical role enumeration")
    name: str = Field(..., description="Human-readable title, e.g. 'Researcher & Knowledge Retriever'")
    description: str = Field(..., description="High-level description of responsibilities")
    system_prompt: str = Field(..., description="Comprehensive system instructions framing the role's behavior")
    allowed_tools: List[str] = Field(default_factory=list, description="List of registered tool names permitted for this role")
    max_execution_turns: int = Field(default=5, ge=1, le=50, description="Turn execution circuit breaker")
    context_budget_tokens: int = Field(default=4096, ge=512, le=32768, description="Allocated context token budget")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature for LLM generation")
    can_delegate_to: List[AgentRoleType] = Field(default_factory=list, description="Roles this agent is permitted to delegate to")
    exemplar_queries: List[str] = Field(default_factory=list, description="Bilingual exemplar queries for router semantic prototype")


class HandoffPacket(BaseModel):
    """
    5-Component self-contained handoff report between agent roles.
    Adheres strictly to the Teamwork Handoff Protocol:
    1. Observation
    2. Logic Chain
    3. Caveats
    4. Conclusion
    5. Verification Method
    """
    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique subtask or task ID")
    sender_role: AgentRoleType = Field(..., description="Role generating the handoff")
    recipient_role: AgentRoleType = Field(..., description="Target role receiving the handoff")
    observation: str = Field(..., description="1. Observation: directly observed facts, paths, line numbers, tool results")
    logic_chain: str = Field(..., description="2. Logic Chain: step-by-step reasoning from observation to conclusion")
    caveats: str = Field(default="No caveats.", description="3. Caveats: known assumptions, uninvestigated areas, or limitations")
    conclusion: str = Field(..., description="4. Conclusion: actionable assessment or completed deliverable")
    verification_method: str = Field(..., description="5. Verification Method: commands, assertions, or inspection criteria")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Map of artifact paths to descriptions")
    timestamp: float = Field(default_factory=time.time, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary coordination metadata")

    @property
    def target_role(self) -> AgentRoleType:
        """Alias for recipient_role."""
        return self.recipient_role

    def format_markdown(self) -> str:
        """Render standard 5-component handoff markdown report."""
        lines = [
            f"# Handoff Report — Task {self.task_id}",
            f"- **Sender**: `{self.sender_role.value}`",
            f"- **Recipient**: `{self.recipient_role.value}`",
            f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(self.timestamp))}`",
            "",
            "## 1. Observation",
            self.observation.strip(),
            "",
            "## 2. Logic Chain",
            self.logic_chain.strip(),
            "",
            "## 3. Caveats",
            self.caveats.strip() if self.caveats else "No caveats.",
            "",
            "## 4. Conclusion",
            self.conclusion.strip(),
            "",
            "## 5. Verification Method",
            self.verification_method.strip(),
            "",
        ]
        if self.artifacts:
            lines.append("## Artifacts")
            for name, path in self.artifacts.items():
                lines.append(f"- **{name}**: `{path}`")
            lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Alias for format_markdown."""
        return self.format_markdown()


    def is_complete(self) -> bool:
        """Validate that all 5 components are populated with non-trivial text."""
        return all(
            bool(str(getattr(self, field, "")).strip())
            for field in ("observation", "logic_chain", "caveats", "conclusion", "verification_method")
        )


class AgentMessageType(str, Enum):
    """Categorization of inter-agent messages."""
    TASK_DISPATCH = "task_dispatch"
    HANDOFF = "handoff"
    QUERY = "query"
    RESPONSE = "response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Structured message exchanged between agents on the message bus."""
    model_config = ConfigDict(extra="ignore")

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = Field(default=None, description="Correlating task identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    sender_role: AgentRoleType = Field(..., description="Role of the dispatching agent")
    recipient_role: Union[AgentRoleType, str] = Field(default="all", description="Target role or broadcast 'all'")
    message_type: AgentMessageType = Field(default=AgentMessageType.TASK_DISPATCH)
    content: str = Field(..., description="Human or LLM-readable message content")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured payload or intermediate artifacts")
    reply_to_id: Optional[str] = Field(default=None, description="Previous message ID being answered")
    handoff: Optional[HandoffPacket] = Field(default=None, description="Optional attached handoff packet")
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubtaskStatus(str, Enum):
    """Execution state of an individual subtask node in a TaskDAG."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubtaskNode(BaseModel):
    """A discrete unit of work within a TaskDAG."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique subtask identifier")
    title: str = Field(..., description="Short summary title of the subtask")
    description: str = Field(..., description="Detailed instructions or prompt for this subtask")
    assigned_role: AgentRoleType = Field(..., description="Role designated to execute this node")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite subtasks")
    status: Union[SubtaskStatus, str] = Field(default=SubtaskStatus.PENDING)
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Inputs passed from predecessor nodes")
    output_data: Optional[Any] = Field(default=None, description="Output generated upon completion")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    handoff: Optional[HandoffPacket] = Field(default=None, description="Associated handoff packet")
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_id_and_result(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support node_id as alias for id
            if "node_id" in data and "id" not in data:
                data["id"] = data["node_id"]
            # Support result as alias for output_data
            if "result" in data and "output_data" not in data:
                data["output_data"] = data["result"]
        return data

    @property
    def node_id(self) -> str:
        """Alias for id."""
        return self.id

    @property
    def result(self) -> Optional[Any]:
        """Alias for output_data."""
        return self.output_data

    @result.setter
    def result(self, val: Any) -> None:
        self.output_data = val


class TaskDAG(BaseModel):
    """
    Directed Acyclic Graph of subtasks.
    Enforces topological order, validates absence of cycles using Kahn's algorithm,
    and aggregates intermediate outputs into a coherent result.
    """
    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = Field(..., description="Overarching user task or objective")
    nodes: Dict[str, SubtaskNode] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def dag_id(self) -> str:
        """Alias for task_id."""
        return self.task_id

    def add_subtask(self, node: SubtaskNode) -> None:
        """Add a subtask node to the DAG. Raises ValueError if cycle is created."""
        if node.id in self.nodes:
            raise ValueError(f"Subtask with ID '{node.id}' already exists in DAG.")
        self.nodes[node.id] = node
        if self.has_cycle():
            del self.nodes[node.id]
            raise ValueError(f"Adding subtask '{node.id}' ({node.title}) creates a cycle in TaskDAG.")

    def add_node(self, node: SubtaskNode) -> None:
        """Alias for add_subtask."""
        self.add_subtask(node)

    def add_dependency(self, subtask_id: str, depends_on_id: str) -> None:
        """Declare that `subtask_id` depends on the completion of `depends_on_id`."""
        if subtask_id not in self.nodes:
            raise KeyError(f"Subtask '{subtask_id}' not found in DAG.")
        if depends_on_id not in self.nodes:
            raise KeyError(f"Dependency '{depends_on_id}' not found in DAG.")
        if depends_on_id not in self.nodes[subtask_id].dependencies:
            self.nodes[subtask_id].dependencies.append(depends_on_id)
        if self.has_cycle():
            self.nodes[subtask_id].dependencies.remove(depends_on_id)
            raise ValueError(f"Dependency '{depends_on_id}' creates a cycle in TaskDAG.")

    def has_cycle(self) -> bool:
        """Detect if graph contains any directed cycle using Kahn's algorithm."""
        in_degree: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = defaultdict(list)

        for nid, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    adj[dep_id].append(nid)
                    in_degree[nid] += 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return visited_count != len(self.nodes)

    def validate_dag(self) -> bool:
        """
        Validate that the graph is acyclic using Kahn's algorithm.
        Raises ValueError if a cycle is detected.
        """
        if self.has_cycle():
            raise ValueError("Cycle detected in TaskDAG!")
        return True

    def topological_sort(self) -> List[str]:
        """
        Returns subtask IDs in a valid topological execution order.
        Raises ValueError if a cycle exists.
        """
        in_degree: Dict[str, int] = {k: 0 for k in self.nodes}
        adj: Dict[str, List[str]] = defaultdict(list)

        for nid, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    raise ValueError(f"Node '{nid}' references non-existent dependency '{dep_id}'.")
                adj[dep_id].append(nid)
                in_degree[nid] += 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            unresolved = [nid for nid, deg in in_degree.items() if deg > 0]
            raise ValueError(f"Cycle detected in TaskDAG! Unresolvable nodes: {unresolved}")

        return order

    def topological_order(self) -> List[str]:
        """Alias for topological_sort."""
        return self.topological_sort()

    def get_runnable_tasks(self) -> List[SubtaskNode]:
        """
        Return all subtask nodes in PENDING state whose dependencies are all COMPLETED.
        """
        runnable = []
        for node in self.nodes.values():
            node_status = node.status.value if isinstance(node.status, SubtaskStatus) else str(node.status)
            if node_status != "pending":
                continue
            deps_met = True
            for dep_id in node.dependencies:
                dep_node = self.nodes.get(dep_id)
                if not dep_node:
                    deps_met = False
                    break
                dep_status = dep_node.status.value if isinstance(dep_node.status, SubtaskStatus) else str(dep_node.status)
                if dep_status != "completed":
                    deps_met = False
                    break
            if deps_met:
                runnable.append(node)
        return runnable

    def get_runnable_nodes(self) -> List[SubtaskNode]:
        """Alias for get_runnable_tasks."""
        return self.get_runnable_tasks()

    def mark_completed(
        self, subtask_id: str, output: Any = None, handoff: Optional[HandoffPacket] = None
    ) -> None:
        """Mark a subtask as COMPLETED and store its output."""
        if subtask_id not in self.nodes:
            raise KeyError(f"Subtask '{subtask_id}' not found in DAG.")
        node = self.nodes[subtask_id]
        node.status = SubtaskStatus.COMPLETED
        node.output_data = output
        node.handoff = handoff
        node.completed_at = time.time()

    def mark_node_completed(
        self, node_id: str, result: Any = None, handoff: Optional[HandoffPacket] = None
    ) -> None:
        """Alias for mark_completed."""
        self.mark_completed(subtask_id=node_id, output=result, handoff=handoff)

    def mark_failed(self, subtask_id: str, error: str) -> None:
        """Mark a subtask as FAILED and mark all dependent child tasks as SKIPPED."""
        if subtask_id not in self.nodes:
            raise KeyError(f"Subtask '{subtask_id}' not found in DAG.")
        node = self.nodes[subtask_id]
        node.status = SubtaskStatus.FAILED
        node.error = error
        node.completed_at = time.time()

        # Cascade SKIPPED status to all downstream dependent nodes
        adj: Dict[str, List[str]] = defaultdict(list)
        for nid, n in self.nodes.items():
            for dep in n.dependencies:
                adj[dep].append(nid)

        queue = deque(adj[subtask_id])
        while queue:
            curr = queue.popleft()
            curr_node = self.nodes.get(curr)
            if curr_node and curr_node.status == SubtaskStatus.PENDING:
                curr_node.status = SubtaskStatus.SKIPPED
                curr_node.error = f"Prerequisite '{subtask_id}' failed."
                queue.extend(adj[curr])

    def mark_node_failed(self, node_id: str, error: str) -> None:
        """Alias for mark_failed."""
        self.mark_failed(subtask_id=node_id, error=error)

    def is_finished(self) -> bool:
        """Return True if all nodes have reached a terminal state (COMPLETED, FAILED, or SKIPPED)."""
        if not self.nodes:
            return True
        for node in self.nodes.values():
            s = node.status.value if isinstance(node.status, SubtaskStatus) else str(node.status)
            if s not in ("completed", "failed", "skipped"):
                return False
        return True

    def is_completed(self) -> bool:
        """Alias for is_finished."""
        return self.is_finished()

    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate outputs and handoffs of all completed subtasks into a summary dictionary."""
        results: Dict[str, Any] = {}
        for nid, node in self.nodes.items():
            s = node.status.value if isinstance(node.status, SubtaskStatus) else str(node.status)
            results[nid] = {
                "title": node.title,
                "role": node.assigned_role.value,
                "status": s,
                "output": node.output_data,
                "error": node.error,
                "has_handoff": node.handoff is not None,
            }
        return results
