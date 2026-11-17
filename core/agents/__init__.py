"""
core/agents
Multi-Agent Interaction & Declarative Tools Subsystem.
Milestone 3 Core Architecture.
"""

from core.agents.models import (
    AgentRoleType,
    ToolParameter,
    ToolParameterField,
    ToolDefinition,
    ToolCallRequest,
    ToolExecutionResult,
    AgentRoleDefinition,
    HandoffPacket,
    AgentMessageType,
    AgentMessage,
    SubtaskStatus,
    SubtaskNode,
    TaskDAG,
)

from core.agents.roles import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    RETRIEVER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    ROLE_ORCHESTRATOR,
    ROLE_RETRIEVER,
    ROLE_RESEARCHER,
    ROLE_PLANNER,
    ROLE_EXECUTOR,
    ROLE_REVIEWER,
    ROLE_REGISTRY,
    get_role_definition,
    get_all_roles,
    create_orchestrator_role,
    create_researcher_role,
    create_planner_role,
    create_executor_role,
    create_reviewer_role,
)

from core.agents.router import (
    RouteDecision,
    IntentRouter,
)

from core.agents.tools import (
    assert_safe_path,
    is_safe_path,
    truncate_tool_output,
    execute_with_timeout,
    agent_tool,
    ToolRegistry,
    make_file_tools,
    make_rag_tools,
    make_memory_tools,
    safe_eval,
    make_safe_eval_tool,
)

from core.agents.coordinator import (
    MessageBus,
    AgentCoordinator,
)

from core.agents.loop import (
    CircuitState,
    CircuitBreaker,
    LocalDeterministicEngine,
    ResilientAgentLoop,
)

__all__ = [
    # Models
    "AgentRoleType",
    "ToolParameter",
    "ToolParameterField",
    "ToolDefinition",
    "ToolCallRequest",
    "ToolExecutionResult",
    "AgentRoleDefinition",
    "HandoffPacket",
    "AgentMessageType",
    "AgentMessage",
    "SubtaskStatus",
    "SubtaskNode",
    "TaskDAG",
    # Roles
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "RETRIEVER_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "EXECUTOR_SYSTEM_PROMPT",
    "REVIEWER_SYSTEM_PROMPT",
    "ROLE_ORCHESTRATOR",
    "ROLE_RETRIEVER",
    "ROLE_RESEARCHER",
    "ROLE_PLANNER",
    "ROLE_EXECUTOR",
    "ROLE_REVIEWER",
    "ROLE_REGISTRY",
    "get_role_definition",
    "get_all_roles",
    "create_orchestrator_role",
    "create_researcher_role",
    "create_planner_role",
    "create_executor_role",
    "create_reviewer_role",
    # Router
    "RouteDecision",
    "IntentRouter",
    # Tools
    "assert_safe_path",
    "is_safe_path",
    "truncate_tool_output",
    "execute_with_timeout",
    "agent_tool",
    "ToolRegistry",
    "make_file_tools",
    "make_rag_tools",
    "make_memory_tools",
    "safe_eval",
    "make_safe_eval_tool",
    # Coordinator
    "MessageBus",
    "AgentCoordinator",
    # Loop
    "CircuitState",
    "CircuitBreaker",
    "LocalDeterministicEngine",
    "ResilientAgentLoop",
]
