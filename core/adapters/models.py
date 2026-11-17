"""
core/adapters/models.py
Canonical Pydantic v2 data models for multi-environment adapters.
Provides environment-agnostic representations for tools, messages, contexts, and responses.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EnvironmentType(str, Enum):
    """Supported assistant and IDE runtime environments."""
    ANTIGRAVITY = "antigravity"
    CLAUDE = "claude"
    CODEX = "codex"
    GENERIC_LOCAL = "generic_local"
    AUTO = "auto"


class ToolParameter(BaseModel):
    """Canonical parameter specification for a tool."""
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str = "string"  # "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """Canonical environment-agnostic tool definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    name: str
    description: str
    category: str = "general"
    parameters: List[ToolParameter] = Field(default_factory=list)
    handler: Optional[Callable[..., Any]] = Field(default=None, exclude=True)

    @classmethod
    def from_agent_tool(cls, agent_tool_def: Any) -> ToolDefinition:
        """
        Convert core.agents.models.ToolDefinition or object with similar attributes
        to canonical ToolDefinition.
        """
        if isinstance(agent_tool_def, ToolDefinition):
            return agent_tool_def

        params: List[ToolParameter] = []
        raw_params = getattr(agent_tool_def, "parameters", None)
        if raw_params:
            if isinstance(raw_params, list):
                for p in raw_params:
                    if isinstance(p, ToolParameter):
                        params.append(p)
                    elif isinstance(p, dict):
                        params.append(ToolParameter(**p))
                    else:
                        params.append(
                            ToolParameter(
                                name=getattr(p, "name", ""),
                                type=getattr(p, "type", "string"),
                                description=getattr(p, "description", ""),
                                required=getattr(p, "required", True),
                                default=getattr(p, "default", None),
                                enum_values=getattr(p, "enum", getattr(p, "enum_values", None)),
                            )
                        )
            elif hasattr(raw_params, "fields"):
                for p in raw_params.fields:
                    params.append(
                        ToolParameter(
                            name=getattr(p, "name", ""),
                            type=getattr(p, "type", "string"),
                            description=getattr(p, "description", ""),
                            required=getattr(p, "required", True),
                            default=getattr(p, "default", None),
                            enum_values=getattr(p, "enum", getattr(p, "enum_values", None)),
                        )
                    )

        return cls(
            name=getattr(agent_tool_def, "name", "tool"),
            description=getattr(agent_tool_def, "description", ""),
            category=getattr(agent_tool_def, "category", "general"),
            parameters=params,
            handler=getattr(agent_tool_def, "handler", None),
        )


class ToolCall(BaseModel):
    """Canonical model for a tool call invocation requested by an agent."""
    model_config = ConfigDict(extra="ignore")

    call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Canonical representation of tool execution result returned to model."""
    model_config = ConfigDict(extra="ignore")

    call_id: str
    tool_name: str
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class RoleEnum(str, Enum):
    """Canonical conversation roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentMessage(BaseModel):
    """Canonical conversation message."""
    model_config = ConfigDict(extra="ignore")

    role: RoleEnum
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Aggregated agent context including session history, RAG chunks, and memory facts."""
    model_config = ConfigDict(extra="ignore")

    session_id: str
    messages: List[AgentMessage] = Field(default_factory=list)
    rag_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    memory_facts: List[str] = Field(default_factory=list)
    env_metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Unified response returned by AgentRuntime across all environments."""
    model_config = ConfigDict(extra="ignore")

    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    environment: EnvironmentType
    raw_response: Optional[Dict[str, Any]] = None
