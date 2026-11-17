"""
core/adapters
Multi-Environment Adapters Subsystem (Milestone 4).
Connects the universal multi-agent system to Antigravity, Claude Code, Codex, and local execution.
"""

from core.adapters.models import (
    AgentContext,
    AgentMessage,
    AgentResponse,
    EnvironmentType,
    RoleEnum,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from core.adapters.base import (
    BaseEnvironmentAdapter,
    GenericLocalAdapter,
)
from core.adapters.antigravity_adapter import (
    AntigravityEnvironmentAdapter,
)
from core.adapters.claude_adapter import (
    ClaudeCodeEnvironmentAdapter,
)
from core.adapters.codex_adapter import (
    CodexEnvironmentAdapter,
)
from core.adapters.detector import (
    EnvironmentDetector,
    get_adapter,
)

create_adapter = get_adapter

__all__ = [
    # Models
    "EnvironmentType",
    "ToolParameter",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "RoleEnum",
    "AgentMessage",
    "AgentContext",
    "AgentResponse",
    # Adapters
    "BaseEnvironmentAdapter",
    "GenericLocalAdapter",
    "AntigravityEnvironmentAdapter",
    "ClaudeCodeEnvironmentAdapter",
    "CodexEnvironmentAdapter",
    # Detection & Factory
    "EnvironmentDetector",
    "get_adapter",
    "create_adapter",
]
