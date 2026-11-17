"""
core/agents/coordinator.py
Inter-agent coordination subsystem:
- Structured AgentMessage bus with topic-based pub/sub and audit trails.
- TaskDAG subtask decomposition and topological execution.
- 5-Component self-contained Handoff Protocol (HandoffPacket).
- AgentCoordinator for orchestration, DAG execution, and role delegation.
"""

from __future__ import annotations
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

from core.agents.models import (
    AgentMessage,
    AgentMessageType,
    AgentRoleType,
    HandoffPacket,
    SubtaskNode,
    SubtaskStatus,
    TaskDAG,
)
from core.agents.router import IntentRouter
from core.agents.tools import ToolRegistry

logger = logging.getLogger("core.agents.coordinator")


class MessageBus:
    """
    Thread-safe publish/subscribe message bus for inter-agent communication.
    Maintains an in-memory audit trail and routes messages to registered callbacks.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[AgentMessage], None]]] = defaultdict(list)
        self._history: List[AgentMessage] = []
        self._lock = threading.RLock()

    def subscribe(
        self, role: Union[AgentRoleType, str], callback: Callable[[AgentMessage], None]
    ) -> None:
        """Subscribe a callback to messages directed to a specific role or 'all'."""
        key = role.value if isinstance(role, AgentRoleType) else str(role)
        with self._lock:
            self._subscribers[key].append(callback)

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to target recipients and broadcast subscribers."""
        with self._lock:
            self._history.append(message)
            target_key = (
                message.recipient_role.value
                if isinstance(message.recipient_role, AgentRoleType)
                else str(message.recipient_role)
            )
            # Notify direct recipient subscribers
            callbacks = list(self._subscribers.get(target_key, []))
            # If not already broadcast, also notify broadcast subscribers
            if target_key != "all":
                callbacks.extend(self._subscribers.get("all", []))

        # Execute callbacks outside lock to prevent deadlocks
        for cb in callbacks:
            try:
                cb(message)
            except Exception as e:
                logger.warning("Error in MessageBus subscriber callback: %s", e)

    def get_history(
        self,
        task_id: Optional[str] = None,
        role: Optional[Union[AgentRoleType, str]] = None,
    ) -> List[AgentMessage]:
        """Retrieve filtered message history."""
        with self._lock:
            res = self._history
            if task_id:
                res = [m for m in res if m.task_id == task_id]
            if role:
                r_key = role.value if isinstance(role, AgentRoleType) else str(role)
                res = [
                    m
                    for m in res
                    if (m.sender_role.value == r_key)
                    or (isinstance(m.recipient_role, AgentRoleType) and m.recipient_role.value == r_key)
                    or (str(m.recipient_role) in (r_key, "all"))
                ]
            return list(res)

    def clear(self) -> None:
        """Clear message history and subscriber registry."""
        with self._lock:
            self._history.clear()
            self._subscribers.clear()


class AgentCoordinator:
    """
    Central multi-agent coordinator.
    Orchestrates intent routing, task delegation, structured message logging,
    TaskDAG decomposition, and self-contained 5-component handoff synthesis.
    """

    def __init__(
        self,
        router: Optional[IntentRouter] = None,
        tool_registry: Optional[ToolRegistry] = None,
        message_bus: Optional[MessageBus] = None,
    ):
        self.router = router or IntentRouter()
        self.tools = tool_registry or ToolRegistry()
        self.bus = message_bus or MessageBus()

    def dispatch_task(
        self,
        task: str,
        session_id: str = "default_session",
        target_role: Optional[AgentRoleType] = None,
    ) -> HandoffPacket:
        """
        Dispatches a task from the Orchestrator to an appropriate specialized role.
        Emits task dispatch message and synthesizes a 5-component self-contained HandoffPacket.
        """
        dest_role = target_role or self.router.route(task)
        task_id = str(uuid.uuid4())

        # 1. Publish Dispatch Message to Bus
        dispatch_msg = AgentMessage(
            task_id=task_id,
            session_id=session_id,
            sender_role=AgentRoleType.ORCHESTRATOR,
            recipient_role=dest_role,
            message_type=AgentMessageType.TASK_DISPATCH,
            content=f"Task dispatched to {dest_role.value}: '{task}'",
            payload={"task": task, "session_id": session_id},
        )
        self.bus.publish(dispatch_msg)

        # 2. Synthesize 5-Component HandoffPacket
        packet = HandoffPacket(
            task_id=task_id,
            sender_role=AgentRoleType.ORCHESTRATOR,
            recipient_role=dest_role,
            observation=f"Received task query: '{task}' for session '{session_id}'.",
            logic_chain=f"Task routed to {dest_role.value} via dynamic intent classification.",
            caveats="Hermetic offline execution without external paid APIs.",
            conclusion=f"Task delegated to {dest_role.value} for subtask resolution.",
            verification_method=f"Check session '{session_id}' state and tool call logs.",
            artifacts={},
        )

        # 3. Publish Handoff Message
        handoff_msg = AgentMessage(
            task_id=task_id,
            session_id=session_id,
            sender_role=AgentRoleType.ORCHESTRATOR,
            recipient_role=dest_role,
            message_type=AgentMessageType.HANDOFF,
            content=packet.format_markdown(),
            handoff=packet,
        )
        self.bus.publish(handoff_msg)

        return packet

    def decompose_task_into_dag(
        self, task: str, session_id: str = "default_session"
    ) -> TaskDAG:
        """
        Decomposes a complex objective into an executable, acyclic TaskDAG.
        Creates canonical 4-stage pipeline: Research -> Plan -> Execute -> Review.
        """
        dag = TaskDAG(goal=task)

        # Stage 1: Research (Retriever)
        n1 = SubtaskNode(
            id="step_1_research",
            title="Gather Information & Context",
            description=f"Search knowledge base and extract context for: '{task}'",
            assigned_role=AgentRoleType.RETRIEVER,
            dependencies=[],
        )
        dag.add_subtask(n1)

        # Stage 2: Architecture & Planning (Planner)
        n2 = SubtaskNode(
            id="step_2_planning",
            title="Design Implementation Plan",
            description=f"Decompose into detailed implementation steps for: '{task}'",
            assigned_role=AgentRoleType.PLANNER,
            dependencies=["step_1_research"],
        )
        dag.add_subtask(n2)

        # Stage 3: Implementation & Execution (Executor)
        n3 = SubtaskNode(
            id="step_3_execution",
            title="Execute Implementation",
            description=f"Apply code modifications and run actions for: '{task}'",
            assigned_role=AgentRoleType.EXECUTOR,
            dependencies=["step_2_planning"],
        )
        dag.add_subtask(n3)

        # Stage 4: Verification & Review (Reviewer)
        n4 = SubtaskNode(
            id="step_4_review",
            title="Verify & Audit Results",
            description=f"Audit outputs, run tests, and emit final handoff for: '{task}'",
            assigned_role=AgentRoleType.REVIEWER,
            dependencies=["step_3_execution"],
        )
        dag.add_subtask(n4)

        return dag

    def execute_dag(
        self,
        dag: TaskDAG,
        session_id: str = "default_session",
        loop: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes a TaskDAG in topological order.
        If an execution loop is provided, delegates each runnable subtask to loop.run().
        """
        # Validate absence of cycles before running
        dag.validate_dag()

        while not dag.is_finished():
            runnable = dag.get_runnable_tasks()
            if not runnable:
                # If not finished but no tasks are runnable, break to prevent hang
                break

            for node in runnable:
                node.status = SubtaskStatus.RUNNING
                node.started_at = time.time()

                # Build input context from dependencies
                dep_outputs = {}
                for dep_id in node.dependencies:
                    dep_node = dag.nodes.get(dep_id)
                    if dep_node:
                        dep_outputs[dep_id] = dep_node.output_data

                node.input_data["dependency_outputs"] = dep_outputs

                # Execute node via loop or fallback
                if loop is not None and hasattr(loop, "run"):
                    try:
                        packet = loop.run(
                            task=node.description,
                            session_id=session_id,
                            target_role=node.assigned_role,
                        )
                        dag.mark_completed(node.id, output=packet.conclusion, handoff=packet)
                    except Exception as ex:
                        logger.error("Execution failed for node '%s': %s", node.id, ex)
                        dag.mark_failed(node.id, error=str(ex))
                else:
                    # Deterministic simulated completion
                    output_summary = f"Simulated output for {node.title} by {node.assigned_role.value}"
                    packet = HandoffPacket(
                        task_id=dag.task_id,
                        sender_role=node.assigned_role,
                        recipient_role=AgentRoleType.ORCHESTRATOR,
                        observation=f"Executed subtask {node.id}.",
                        logic_chain=f"Processed with inputs: {list(dep_outputs.keys())}",
                        caveats="Simulated execution without runtime loop.",
                        conclusion=output_summary,
                        verification_method="Static verification.",
                    )
                    dag.mark_completed(node.id, output=output_summary, handoff=packet)

        return dag.aggregate_results()
