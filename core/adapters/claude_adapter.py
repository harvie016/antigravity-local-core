"""
core/adapters/claude_adapter.py
Claude Code Environment Adapter (Milestone 4, F15).
Implements Anthropic tool calling schema, claude_skills/manifest.json synchronization,
and .claude/commands/ slash command generators.
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
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

logger = logging.getLogger("core.adapters.claude")


class ClaudeCodeEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Adapter for Anthropic Claude Code CLI & Desktop environments.
    - Exports Anthropic-compliant tool schemas (input_schema).
    - Synchronizes claude_skills/manifest.json with registered agent skills.
    - Generates .claude/commands/ slash command definitions.
    - Enforces concise Karpathy zero-hallucination memory guidelines.
    """
    environment_type = EnvironmentType.CLAUDE

    def export_tool_schema(self, tool: ToolDefinition) -> Dict[str, Any]:
        """
        Convert canonical ToolDefinition into Anthropic tool schema.
        Uses 'name', 'description', and 'input_schema' JSON Schema object.
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
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def parse_tool_call(self, raw_call: Any) -> ToolCall:
        """
        Parse Anthropic 'tool_use' block into canonical ToolCall.
        Handles: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        """
        if isinstance(raw_call, ToolCall):
            return raw_call

        call_id = str(uuid.uuid4())
        tool_name = "unknown_tool"
        arguments: Dict[str, Any] = {}

        if isinstance(raw_call, dict):
            call_id = raw_call.get("id") or raw_call.get("call_id") or call_id
            tool_name = raw_call.get("name") or raw_call.get("tool_name") or tool_name
            # Anthropic uses 'input' for parameters
            args_val = raw_call.get("input")
            if args_val is None:
                args_val = raw_call.get("arguments", {})

            if isinstance(args_val, str):
                try:
                    arguments = json.loads(args_val)
                except Exception:
                    arguments = {"raw_input": args_val}
            elif isinstance(args_val, dict):
                arguments = args_val
        else:
            call_id = getattr(raw_call, "id", call_id)
            tool_name = getattr(raw_call, "name", tool_name)
            args_val = getattr(raw_call, "input", getattr(raw_call, "arguments", {}))
            if isinstance(args_val, dict):
                arguments = args_val
            elif isinstance(args_val, str):
                try:
                    arguments = json.loads(args_val)
                except Exception:
                    arguments = {"raw_input": args_val}

        return ToolCall(call_id=call_id, tool_name=tool_name, arguments=arguments)

    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """
        Format ToolResult into Anthropic 'tool_result' content block.
        """
        if result.error:
            content_str = f"Error: {result.error}"
            is_error = True
        elif isinstance(result.output, (dict, list)):
            content_str = json.dumps(result.output, ensure_ascii=False)
            is_error = False
        else:
            content_str = str(result.output)
            is_error = False

        return {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": content_str,
            "is_error": is_error,
        }

    def build_system_prompt(self, base_prompt: str, context: AgentContext) -> str:
        """
        Enrich base system prompt with Claude Code rules and retrieved context.
        Enforces Karpathy zero-hallucination discipline and Torvalds pragmatism.
        """
        sections: List[str] = [
            "# Claude Code Multi-Agent Context",
            base_prompt.strip(),
            (
                "## Core Engineering Directives\n"
                "- Karpathy Discipline: Never invent file paths, tools, or facts. Quote verbatim evidence.\n"
                "- Torvalds Pragmatism: Deliver clean, flat, working solutions. Avoid unnecessary boilerplate.\n"
                "- Conciseness: Answer directly to the point. Place detailed reports in files and cite paths."
            ),
        ]

        if context.memory_facts:
            facts_block = "\n".join(f"- {fact}" for fact in context.memory_facts)
            sections.append(f"## User Profile & Preferences\n{facts_block}")

        if context.rag_chunks:
            chunks_block = "\n\n".join(
                f"[{c.get('source', 'doc')}]: {c.get('content', '')}"
                for c in context.rag_chunks
            )
            sections.append(f"## Project Documentation Context\n{chunks_block}")

        return "\n\n".join(sections)

    def parse_response(self, raw_response: Any) -> AgentResponse:
        """
        Parse Anthropic Messages API response into canonical AgentResponse.
        Extracts 'text' blocks and 'tool_use' blocks.
        """
        if isinstance(raw_response, AgentResponse):
            return raw_response

        content = ""
        tool_calls: List[ToolCall] = []

        if isinstance(raw_response, dict):
            # Anthropic response payload
            content_blocks = raw_response.get("content", [])
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            content += block.get("text", "")
                        elif b_type == "tool_use":
                            tool_calls.append(self.parse_tool_call(block))
                    elif isinstance(block, str):
                        content += block
            elif isinstance(content_blocks, str):
                content = content_blocks

            if not content and "text" in raw_response:
                content = raw_response["text"]

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

        # Handle Anthropic Message object
        if hasattr(raw_response, "content") and isinstance(raw_response.content, list):
            for block in raw_response.content:
                if getattr(block, "type", "") == "text":
                    content += getattr(block, "text", "")
                elif getattr(block, "type", "") == "tool_use":
                    tool_calls.append(self.parse_tool_call(block))

        return AgentResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            environment=self.environment_type,
        )

    def sync_workspace_artifacts(
        self, tools: List[ToolDefinition], target_dir: Optional[str] = None
    ) -> List[str]:
        """
        Synchronize claude_skills/manifest.json and generate .claude/commands/ slash commands.
        """
        root_dir = Path(target_dir) if target_dir else Path.cwd()
        created_files: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Synchronize claude_skills/manifest.json
        # ----------------------------------------------------------------------
        skills_dir = root_dir / "claude_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = skills_dir / "manifest.json"

        manifest_data: Dict[str, Any] = {"lastUpdated": int(time.time() * 1000), "skills": []}
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "skills" in loaded:
                    manifest_data = loaded
            except Exception as e:
                logger.warning("Could not parse existing claude_skills/manifest.json: %s", e)

        existing_skills = {s.get("skillId"): s for s in manifest_data.get("skills", [])}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Map each tool or agent capability to a skill
        for tool in tools:
            skill_id = f"tool-{tool.name.replace('_', '-')}"
            skill_entry = {
                "skillId": skill_id,
                "name": tool.name,
                "description": tool.description,
                "source": "local-rag-agents",
                "category": tool.category,
                "updatedAt": now_iso,
            }
            existing_skills[skill_id] = skill_entry

        manifest_data["lastUpdated"] = int(time.time() * 1000)
        manifest_data["skills"] = list(existing_skills.values())

        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        created_files.append(str(manifest_path.resolve()))

        # ----------------------------------------------------------------------
        # 2. Generate .claude/commands/ slash commands
        # ----------------------------------------------------------------------
        commands_dir = root_dir / ".claude" / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)

        commands_map = {
            "rag.md": (
                "# /rag Slash Command\n\n"
                "Queries the project knowledge base via the local hybrid RAG pipeline.\n\n"
                "## Usage\n"
                "```bash\n"
                'python -m core.cli --env claude --rag-query "$ARGUMENTS"\n'
                "```\n"
            ),
            "memory.md": (
                "# /memory Slash Command\n\n"
                "Inspects or updates user profile facts and session conversation history.\n\n"
                "## Usage\n"
                "```bash\n"
                'python -m core.cli --env claude --memory-query "$ARGUMENTS"\n'
                "```\n"
            ),
            "agent.md": (
                "# /agent Slash Command\n\n"
                "Runs autonomous multi-agent task execution loop.\n\n"
                "## Usage\n"
                "```bash\n"
                'python -m core.cli --env claude --task "$ARGUMENTS"\n'
                "```\n"
            ),
        }

        for filename, content in commands_map.items():
            cmd_file = commands_dir / filename
            cmd_file.write_text(content, encoding="utf-8")
            created_files.append(str(cmd_file.resolve()))

        return created_files
