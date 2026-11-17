"""
core/adapters/base.py
Base abstraction for environment adapters and generic local implementation.
Follows Bezos API-First modular architecture: business logic remains independent
from environment-specific tool schemas and messaging protocols.
"""

from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from core.adapters.models import (
    AgentContext,
    AgentResponse,
    EnvironmentType,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger("core.adapters.base")


class BaseEnvironmentAdapter(ABC):
    """
    Unified environment adapter interface.
    Translates between canonical agent structures and environment-specific protocols.
    """
    environment_type: EnvironmentType

    @abstractmethod
    def export_tool_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert a canonical ToolDefinition into the environment's tool schema."""
        pass

    @abstractmethod
    def parse_tool_call(self, raw_call: Any) -> ToolCall:
        """Parse an environment-specific tool call object/dict into a canonical ToolCall."""
        pass

    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Any:
        """Format a ToolResult into the payload expected by the environment's model."""
        pass

    @abstractmethod
    def build_system_prompt(self, base_prompt: str, context: AgentContext) -> Any:
        """Enrich base system prompt with RAG context, memory facts, and environment rules."""
        pass

    @abstractmethod
    def parse_response(self, raw_response: Any) -> AgentResponse:
        """Convert an environment-specific response into a canonical AgentResponse."""
        pass

    @abstractmethod
    def sync_workspace_artifacts(
        self, tools: List[ToolDefinition], target_dir: Optional[str] = None
    ) -> List[str]:
        """Generate and synchronize platform-specific artifact files (manifests, slash commands, skills)."""
        pass


class GenericLocalAdapter(BaseEnvironmentAdapter):
    """
    Default adapter for standalone / local offline execution.
    Uses standard JSON Schema for tools and plain markdown for prompts and artifacts.
    """
    environment_type = EnvironmentType.GENERIC_LOCAL

    def export_tool_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Export tool definition as standard JSON Schema."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in tool.parameters:
            prop: Dict[str, Any] = {
                "type": p.type.lower() if p.type else "string",
                "description": p.description or "",
            }
            if p.enum_values:
                prop["enum"] = p.enum_values
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def parse_tool_call(self, raw_call: Any) -> ToolCall:
        """Parse dictionary or object tool call into canonical ToolCall."""
        if isinstance(raw_call, ToolCall):
            return raw_call
        if isinstance(raw_call, dict):
            call_id = raw_call.get("call_id") or raw_call.get("id") or str(uuid.uuid4())
            name = (
                raw_call.get("tool_name")
                or raw_call.get("name")
                or raw_call.get("function", {}).get("name")
                or "unknown_tool"
            )
            raw_args = raw_call.get("arguments") or raw_call.get("function", {}).get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {"raw_input": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            return ToolCall(call_id=call_id, tool_name=name, arguments=args)

        # Fallback attribute access
        call_id = getattr(raw_call, "id", str(uuid.uuid4()))
        name = getattr(raw_call, "tool_name", getattr(raw_call, "name", "unknown_tool"))
        args = getattr(raw_call, "arguments", {})
        return ToolCall(call_id=call_id, tool_name=name, arguments=args if isinstance(args, dict) else {})

    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Format tool result as simple dictionary."""
        return {
            "tool_name": result.tool_name,
            "call_id": result.call_id,
            "output": result.output,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    def build_system_prompt(self, base_prompt: str, context: AgentContext) -> str:
        """Build enriched system prompt with memory facts and RAG chunks."""
        sections = [base_prompt.strip()]

        if context.memory_facts:
            facts_block = "\n".join(f"- {fact}" for fact in context.memory_facts)
            sections.append(f"## Known Facts & User Preferences\n{facts_block}")

        if context.rag_chunks:
            chunks_block = "\n\n".join(
                f"[{c.get('source', 'doc')}]: {c.get('content', '')}"
                for c in context.rag_chunks
            )
            sections.append(f"## Relevant Knowledge Base Context\n{chunks_block}")

        return "\n\n".join(sections)

    def parse_response(self, raw_response: Any) -> AgentResponse:
        """Parse raw response string or dict into AgentResponse."""
        if isinstance(raw_response, AgentResponse):
            return raw_response

        if isinstance(raw_response, str):
            return AgentResponse(
                content=raw_response,
                environment=self.environment_type,
            )

        if isinstance(raw_response, dict):
            content = raw_response.get("content") or raw_response.get("text") or ""
            raw_calls = raw_response.get("tool_calls", [])
            tool_calls = [self.parse_tool_call(tc) for tc in raw_calls]
            citations = raw_response.get("citations", [])
            return AgentResponse(
                content=content,
                tool_calls=tool_calls,
                citations=citations,
                environment=self.environment_type,
                raw_response=raw_response,
            )

        # Fallback for object with content attribute
        content = getattr(raw_response, "content", str(raw_response))
        return AgentResponse(
            content=content,
            environment=self.environment_type,
        )

    def sync_workspace_artifacts(
        self, tools: List[ToolDefinition], target_dir: Optional[str] = None
    ) -> List[str]:
        """Generate local tools manifest JSON."""
        out_dir = Path(target_dir) if target_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "tools_manifest.json"

        data = {
            "environment": self.environment_type.value,
            "tool_count": len(tools),
            "tools": [self.export_tool_schema(t) for t in tools],
        }
        manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return [str(manifest_path.resolve())]
