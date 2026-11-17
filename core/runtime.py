"""
core/runtime.py
Unified AgentRuntime SDK (Milestone 4, F17).
Connects RAG Pipeline, Memory Manager, Agent Coordinator / Resilient Loop,
and Multi-Environment Adapters into a cohesive, high-level programmatic interface.
"""

from __future__ import annotations
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from core.adapters.base import BaseEnvironmentAdapter
from core.adapters.detector import get_adapter
from core.adapters.models import (
    AgentResponse,
    EnvironmentType,
    ToolCall,
    ToolDefinition,
    ToolParameter,
)
from core.agents.coordinator import AgentCoordinator
from core.agents.loop import ResilientAgentLoop
from core.agents.models import AgentRoleType
from core.agents.tools import ToolRegistry, make_memory_tools, make_rag_tools
from core.memory.manager import MemoryManager
from core.rag.pipeline import RAGPipeline
from core.skills import SkillsIndex

logger = logging.getLogger("core.runtime")


class AgentRuntime:
    """
    High-level Python SDK and execution runtime for multi-agent applications.
    Unifies:
    - Multi-environment tool exporting & response translation (Antigravity, Claude, Codex, Local).
    - Declarative tool registration.
    - Zero-paid RAG search and document indexing.
    - Session and long-term memory lifecycle.
    - Autonomous cascading multi-turn agent loop.
    """

    def __init__(
        self,
        env: Union[EnvironmentType, str] = EnvironmentType.AUTO,
        adapter: Optional[BaseEnvironmentAdapter] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_manager: Optional[MemoryManager] = None,
        agent_coordinator: Optional[AgentCoordinator] = None,
        agent_loop: Optional[ResilientAgentLoop] = None,
        tool_registry: Optional[ToolRegistry] = None,
        skills_index: Optional[SkillsIndex] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        db_path: Optional[str] = None,
        enable_offline_loop: bool = False,
    ):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()

        # 1. Resolve Adapter
        if adapter is not None:
            self.adapter = adapter
            self.env_type = adapter.environment_type
        else:
            self.adapter = get_adapter(env=env, base_path=self.workspace_root)
            self.env_type = self.adapter.environment_type

        # 2. Subsystems: Memory, RAG & Skills
        resolved_db_path = None if (db_path is None or str(db_path).strip() in ("", "None", "null")) else str(db_path)
        self.memory = memory_manager or MemoryManager(db_path=resolved_db_path)
        if rag_pipeline is not None:
            self.rag = rag_pipeline
        else:
            from core.rag.models import RAGConfig
            self.rag = RAGPipeline(config=RAGConfig(vector_store_type="memory", embedding_backend="local_tfidf"))

        self.skills = skills_index or SkillsIndex(skills_root=self.workspace_root / "skills")

        # 3. Tool Registry
        self.tools = tool_registry or ToolRegistry()
        self._canonical_tools: Dict[str, ToolDefinition] = {}

        # Register standard RAG and Memory tools
        self._bind_standard_tools()

        # 4. Coordinator & Optional Offline Runner Loop
        # LLM intelligence is delegated to the host IDE model via MCP;
        # ResilientAgentLoop remains an optional lazy offline runner.
        self.coordinator = agent_coordinator or AgentCoordinator(tool_registry=self.tools)
        self._loop: Optional[ResilientAgentLoop] = (
            agent_loop
            if agent_loop is not None
            else (
                ResilientAgentLoop(
                    memory_manager=self.memory,
                    rag_pipeline=self.rag,
                    tool_registry=self.tools,
                    allowed_root=self.workspace_root,
                    use_gigachat=False,
                )
                if enable_offline_loop
                else None
            )
        )

    @property
    def loop(self) -> ResilientAgentLoop:
        """Lazily instantiate the optional offline runner loop if accessed or needed."""
        if self._loop is None:
            self._loop = ResilientAgentLoop(
                memory_manager=self.memory,
                rag_pipeline=self.rag,
                tool_registry=self.tools,
                allowed_root=self.workspace_root,
                use_gigachat=False,
            )
        return self._loop

    @loop.setter
    def loop(self, value: Optional[ResilientAgentLoop]) -> None:
        self._loop = value

    def _bind_standard_tools(self) -> None:
        """Register built-in RAG and Memory tools into the registry and canonical cache."""
        try:
            rag_tools = make_rag_tools(self.rag)
            for rt in rag_tools:
                self.tools.register(rt)
                c_tool = ToolDefinition.from_agent_tool(rt)
                self._canonical_tools[c_tool.name] = c_tool
        except Exception as e:
            logger.debug("Standard RAG tools not bound: %s", e)

        try:
            mem_tools = make_memory_tools(self.memory)
            for mt in mem_tools:
                self.tools.register(mt)
                c_tool = ToolDefinition.from_agent_tool(mt)
                self._canonical_tools[c_tool.name] = c_tool
        except Exception as e:
            logger.debug("Standard Memory tools not bound: %s", e)

    def register_tool(
        self,
        tool: Union[ToolDefinition, Callable],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[ToolParameter]] = None,
        category: str = "custom",
    ) -> ToolDefinition:
        """
        Register a tool using a ToolDefinition or a callable function.
        Inspects function signatures automatically if parameters are omitted.
        """
        if isinstance(tool, ToolDefinition):
            canonical = tool
        elif callable(tool):
            tool_name = name or tool.__name__
            tool_desc = description or (tool.__doc__.strip() if tool.__doc__ else f"Tool {tool_name}")

            params = parameters
            if params is None:
                params = []
                sig = inspect.signature(tool)
                for p_name, p in sig.parameters.items():
                    p_type = "string"
                    ann = p.annotation
                    ann_str = str(ann).lower()
                    if ann in (int, "int") or "int" in ann_str:
                        p_type = "integer"
                    elif ann in (float, "float") or "float" in ann_str or "number" in ann_str:
                        p_type = "number"
                    elif ann in (bool, "bool") or "bool" in ann_str:
                        p_type = "boolean"
                    elif ann in (list, "list", List) or "list" in ann_str:
                        p_type = "array"
                    elif ann in (dict, "dict", Dict) or "dict" in ann_str:
                        p_type = "object"

                    has_default = p.default is not inspect.Parameter.empty
                    params.append(
                        ToolParameter(
                            name=p_name,
                            type=p_type,
                            description=f"Argument {p_name}",
                            required=not has_default,
                            default=p.default if has_default else None,
                        )
                    )

            canonical = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                category=category,
                parameters=params,
                handler=tool,
            )
        else:
            raise TypeError(f"Expected ToolDefinition or Callable, got {type(tool)}")

        self._canonical_tools[canonical.name] = canonical

        # Wrap into agent tool if handler exists
        if canonical.handler:
            from core.agents.models import ToolDefinition as AgentToolDef, ToolParameter as AgentParam

            agent_params = [
                AgentParam(
                    name=p.name,
                    type=p.type,
                    description=p.description,
                    required=p.required,
                    default=p.default,
                    enum=p.enum_values,
                )
                for p in canonical.parameters
            ]
            agent_tool = AgentToolDef(
                name=canonical.name,
                description=canonical.description,
                parameters=agent_params,
                handler=canonical.handler,
            )
            self.tools.register(agent_tool)

        return canonical

    def get_tools(self) -> List[ToolDefinition]:
        """Return all registered canonical tool definitions."""
        return list(self._canonical_tools.values())

    def export_tools(self) -> List[Dict[str, Any]]:
        """Export all registered tools in the schema of the active environment."""
        return [self.adapter.export_tool_schema(t) for t in self.get_tools()]

    def execute(
        self,
        prompt: str,
        session_id: str = "default_session",
        target_role: Optional[Union[AgentRoleType, str]] = None,
    ) -> AgentResponse:
        """
        Execute an agent task end-to-end and return a unified AgentResponse.
        """
        role_enum: Optional[AgentRoleType] = None
        if target_role:
            if isinstance(target_role, AgentRoleType):
                role_enum = target_role
            else:
                try:
                    role_enum = AgentRoleType(target_role.lower())
                except ValueError:
                    role_enum = None

        # Execute through resilient multi-turn loop
        handoff_packet = self.loop.run(
            task=prompt,
            session_id=session_id,
            target_role=role_enum,
        )

        # Retrieve citations if RAG was used
        citations: List[Dict[str, Any]] = []
        if self.rag:
            try:
                raw_results = self.rag.retrieve(prompt, top_k=3)
                for res in raw_results:
                    chunk = getattr(res, "chunk", None)
                    content = getattr(chunk, "content", "") if chunk else getattr(res, "content", "")
                    src = getattr(chunk, "doc_id", "knowledge_base") if chunk else getattr(res, "source", "knowledge_base")
                    citations.append(
                        {
                            "source": src,
                            "snippet": content[:120],
                            "score": getattr(res, "score", 0.0),
                        }
                    )
            except Exception:
                pass

        # Build raw response dictionary to parse through adapter
        tool_calls: List[ToolCall] = []
        if handoff_packet.artifacts and "tool_calls" in handoff_packet.artifacts:
            for tc_data in handoff_packet.artifacts["tool_calls"]:
                tool_calls.append(self.adapter.parse_tool_call(tc_data))

        raw_payload = {
            "content": handoff_packet.conclusion or handoff_packet.format_markdown(),
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "citations": citations,
            "handoff": handoff_packet.model_dump(),
        }

        response = self.adapter.parse_response(raw_payload)
        response.environment = self.env_type
        response.citations = citations
        return response

    def query_rag(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the RAG subsystem directly and return structured matches."""
        results = self.rag.retrieve(query, top_k=top_k)
        formatted: List[Dict[str, Any]] = []
        for r in results:
            chunk = getattr(r, "chunk", None)
            content = getattr(chunk, "content", "") if chunk else getattr(r, "content", "")
            source = getattr(chunk, "doc_id", "unknown") if chunk else getattr(r, "source", "unknown")
            chunk_id = getattr(chunk, "id", "") if chunk else getattr(r, "chunk_id", "")
            formatted.append(
                {
                    "content": content,
                    "source": source,
                    "score": getattr(r, "score", 0.0),
                    "chunk_id": chunk_id,
                }
            )
        return formatted

    def index_document(self, path: Union[str, Path]) -> Any:
        """Index a single document file or directory into RAG."""
        return self.rag.index_documents([str(path)])

    def query_memory(
        self, session_id: str, query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inspect session conversation history and retrieved memory facts."""
        session_ctx = self.memory.session_memory.get_session(session_id)
        messages = [
            {"role": m.role.value, "content": m.content, "timestamp": m.timestamp}
            for m in (session_ctx.messages if session_ctx else [])
        ]

        facts: List[str] = []
        if query:
            if hasattr(self.memory.long_term_memory, "search_facts"):
                retrieved = self.memory.long_term_memory.search_facts(query, limit=5)
            elif hasattr(self.memory.long_term_memory, "retrieve_facts"):
                retrieved = self.memory.long_term_memory.retrieve_facts(query, top_k=5)
            else:
                retrieved = []
            facts = [f.format_fact() if hasattr(f, "format_fact") else str(f) for f in retrieved]

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages,
            "facts": facts,
        }

    def sync_environment(self, target_dir: Optional[str] = None) -> List[str]:
        """
        Synchronize workspace files (manifests, slash commands, SKILL.md)
        for the active environment.
        """
        tools = self.get_tools()
        return self.adapter.sync_workspace_artifacts(tools=tools, target_dir=target_dir)

    def switch_environment(
        self, env: Union[EnvironmentType, str]
    ) -> BaseEnvironmentAdapter:
        """Dynamically switch the active environment adapter."""
        new_adapter = get_adapter(env=env, base_path=self.workspace_root)
        self.adapter = new_adapter
        self.env_type = new_adapter.environment_type
        return self.adapter

    def status(self) -> Dict[str, Any]:
        """Return diagnostic health and capability status of the runtime."""
        return {
            "environment": self.env_type.value,
            "adapter_class": self.adapter.__class__.__name__,
            "registered_tools_count": len(self.get_tools()),
            "workspace_root": str(self.workspace_root),
            "rag_available": self.rag is not None,
            "memory_available": self.memory is not None,
            "skills_count": len(self.skills) if self.skills else 0,
            "offline_mode": True,
            "api_keys_required": False,
            "llm_mode": "ide_delegated",
            "offline_runner_initialized": self._loop is not None,
        }
