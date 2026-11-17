"""
core/adapters/codex_adapter.py
Codex / OpenAI Environment Adapter (Milestone 4, F16).
Implements standard OpenAI JSON Schema tool calling, chat completions transformer,
and pluggable endpoint integration (OpenAI, Ollama, vLLM, Sber GigaChat, local fallback).
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from core.adapters.base import BaseEnvironmentAdapter
from core.adapters.models import (
    AgentContext,
    AgentResponse,
    EnvironmentType,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger("core.adapters.codex")


class CodexEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Adapter for Codex, OpenAI Assistants, Ollama, and OpenAI-compatible inference servers.
    - Exports tools in OpenAI JSON Schema format (`type: "function"`).
    - Parses chat completions response objects and tool_calls.
    - Transforms tool outputs into OpenAI `role: "tool"` messages.
    - Supports pluggable base_url (Ollama, vLLM, local gateway).
    """
    environment_type = EnvironmentType.CODEX

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.base_url = base_url or "http://localhost:11434/v1"
        self.api_key = api_key
        self.model = model

    def export_tool_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Convert canonical ToolDefinition into OpenAI JSON Schema function tool format.
        Conforms to OpenAI Chat Completions API specification.
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in tool.parameters:
            prop: Dict[str, Any] = {
                "type": p.type.lower() if p.type else "string",
                "description": p.description or f"Parameter {p.name}",
            }
            if p.enum_values:
                prop["enum"] = p.enum_values
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def parse_tool_call(self, raw_call: Any) -> ToolCall:
        """
        Parse OpenAI Chat Completion tool_call into canonical ToolCall.
        Handles: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}
        """
        if isinstance(raw_call, ToolCall):
            return raw_call

        call_id = str(uuid.uuid4())
        tool_name = "unknown_tool"
        arguments: Dict[str, Any] = {}

        if isinstance(raw_call, dict):
            call_id = raw_call.get("id") or raw_call.get("call_id") or call_id
            fn = raw_call.get("function", {})
            if isinstance(fn, dict):
                tool_name = fn.get("name") or raw_call.get("name") or tool_name
                raw_args = fn.get("arguments") or raw_call.get("arguments") or {}
            else:
                tool_name = raw_call.get("name") or tool_name
                raw_args = raw_call.get("arguments") or {}

            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {"raw_input": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args
        else:
            call_id = getattr(raw_call, "id", call_id)
            fn = getattr(raw_call, "function", None)
            if fn:
                tool_name = getattr(fn, "name", tool_name)
                raw_args = getattr(fn, "arguments", {})
            else:
                tool_name = getattr(raw_call, "name", tool_name)
                raw_args = getattr(raw_call, "arguments", {})

            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {"raw_input": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args

        return ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)

    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """
        Format ToolResult into OpenAI tool response message (`role: "tool"`).
        """
        if result.error:
            content_str = json.dumps({"error": result.error}, ensure_ascii=False)
        elif isinstance(result.output, (dict, list)):
            content_str = json.dumps(result.output, ensure_ascii=False)
        else:
            content_str = str(result.output)

        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.tool_name,
            "content": content_str,
        }

    def build_system_prompt(self, base_prompt: str, context: AgentContext) -> str:
        """
        Enrich base prompt with retrieved facts and RAG chunks.
        """
        sections: List[str] = [
            "# OpenAI / Codex Multi-Agent System Prompt",
            base_prompt.strip(),
        ]

        if context.memory_facts:
            facts_block = "\n".join(f"- {fact}" for fact in context.memory_facts)
            sections.append(f"## Memory Facts & Preferences\n{facts_block}")

        if context.rag_chunks:
            chunks_block = "\n\n".join(
                f"[{c.get('source', 'doc')}]: {c.get('content', '')}"
                for c in context.rag_chunks
            )
            sections.append(f"## Injected RAG Context\n{chunks_block}")

        return "\n\n".join(sections)

    def parse_response(self, raw_response: Any) -> AgentResponse:
        """
        Parse OpenAI ChatCompletion response into canonical AgentResponse.
        Extracts choices[0].message.content and choices[0].message.tool_calls.
        """
        if isinstance(raw_response, AgentResponse):
            return raw_response

        content = ""
        tool_calls: List[ToolCall] = []

        if isinstance(raw_response, dict):
            choices = raw_response.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content") or ""
                raw_tc = msg.get("tool_calls", [])
                for tc in raw_tc:
                    tool_calls.append(self.parse_tool_call(tc))
            else:
                content = raw_response.get("content") or raw_response.get("text") or ""
                for tc in raw_response.get("tool_calls", []):
                    tool_calls.append(self.parse_tool_call(tc))

            return AgentResponse(
                content=content.strip(),
                tool_calls=tool_calls,
                citations=raw_response.get("citations", []),
                environment=self.environment_type,
                raw_response=raw_response,
            )

        if isinstance(raw_response, str):
            return AgentResponse(
                content=raw_response.strip(),
                environment=self.environment_type,
            )

        # Handle OpenAI SDK ChatCompletion object
        if hasattr(raw_response, "choices") and getattr(raw_response, "choices"):
            first_choice = raw_response.choices[0]
            msg = getattr(first_choice, "message", None)
            if msg:
                content = getattr(msg, "content", "") or ""
                sdk_tc = getattr(msg, "tool_calls", None) or []
                for tc in sdk_tc:
                    tool_calls.append(self.parse_tool_call(tc))

        return AgentResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            environment=self.environment_type,
        )

    def sync_workspace_artifacts(
        self, tools: List[ToolDefinition], target_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate OpenAI JSON Schema tools manifest and Ollama Modelfile template.
        """
        out_dir = Path(target_dir) if target_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[str] = []

        # 1. Generate openai_tools.json
        tools_path = out_dir / "openai_tools.json"
        schemas = [self.export_tool_schema(t) for t in tools]
        tools_path.write_text(
            json.dumps({"tools": schemas}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        created_files.append(str(tools_path.resolve()))

        # 2. Generate Ollama Modelfile snippet
        modelfile_path = out_dir / "Modelfile.agent"
        modelfile_content = (
            "# Ollama Modelfile for Local Multi-Agent RAG\n"
            f"FROM {self.model}\n\n"
            "SYSTEM \"\"\"\n"
            "You are an autonomous agent capable of using functions for RAG and memory.\n"
            "\"\"\"\n\n"
            f"# Registered tool count: {len(tools)}\n"
        )
        modelfile_path.write_text(modelfile_content, encoding="utf-8")
        created_files.append(str(modelfile_path.resolve()))

        return created_files
