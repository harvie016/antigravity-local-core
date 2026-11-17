"""
Unified Memory Manager Facade.
Coordinates:
- SessionMemoryManager (ephemeral sliding window context)
- LongTermMemoryManager (SQLite FTS5 + Markdown vault)
- Session hydration (injecting persistent facts into pinned system instructions)
- Turn delta synchronization (extracting facts and decisions from dialog turns)
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.memory.models import MessageRole, ChatMessage, SessionContext, FactTriple, EpisodicRecord
from core.memory.session_memory import SessionMemoryManager, truncate_tool_result, count_message_tokens
from core.memory.long_term_memory import LongTermMemoryManager


class MemoryManager:
    """
    Unified Memory Subsystem facade.
    Glues together short-term session memory and persistent long-term memory.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        vault_dir: Optional[Union[str, Path]] = None,
        default_budget: int = 4096,
        session_manager: Optional[SessionMemoryManager] = None,
        long_term_manager: Optional[LongTermMemoryManager] = None,
    ):
        self.session_memory = session_manager or SessionMemoryManager(default_budget=default_budget)
        self.long_term_memory = long_term_manager or LongTermMemoryManager(db_path=db_path, vault_dir=vault_dir)

    # ==========================================================================
    # SESSION HYDRATION
    # ==========================================================================

    def hydrate_session(
        self,
        session_id: str,
        user_query: str = "",
        max_facts: int = 5,
        default_system_prompt: Optional[str] = None,
    ) -> SessionContext:
        """
        Hydrate conversation session by searching relevant facts in LongTermMemory
        and injecting them into a pinned system instruction message.
        """
        ctx = self.session_memory.get_or_create_session(session_id)

        # Retrieve relevant facts
        facts: List[FactTriple] = []
        if user_query:
            facts = self.long_term_memory.search_facts(user_query, limit=max_facts)

        # Also retrieve core user facts/constraints if not already present
        user_facts = self.long_term_memory.list_facts(namespace="user", limit=max_facts)
        seen_keys = {(f.subject, f.predicate) for f in facts}
        for uf in user_facts:
            if (uf.subject, uf.predicate) not in seen_keys and len(facts) < max_facts + 3:
                facts.append(uf)
                seen_keys.add((uf.subject, uf.predicate))

        # Format system prompt block
        fact_lines = [f"- {f.subject} {f.predicate}: {f.object_value}" for f in facts]
        memory_block = ""
        if fact_lines:
            memory_block = (
                "[ДОЛГОВРЕМЕННАЯ ПАМЯТЬ — ПОДТВЕРЖДЕННЫЕ ФАКТЫ]:\n"
                + "\n".join(fact_lines)
            )

        base_prompt = default_system_prompt or "You are an intelligent assistant."
        full_system_content = f"{base_prompt}\n\n{memory_block}".strip() if memory_block else base_prompt

        # Find existing system message or prepend a new pinned one
        system_idx = -1
        for idx, m in enumerate(ctx.messages):
            if m.role == MessageRole.SYSTEM:
                system_idx = idx
                break

        if system_idx >= 0:
            existing = ctx.messages[system_idx]
            if "[ДОЛГОВРЕМЕННАЯ ПАМЯТЬ" in existing.content:
                cleaned = re.sub(
                    r"\n*\[ДОЛГОВРЕМЕННАЯ ПАМЯТЬ[^\n]*\].*",
                    "",
                    existing.content,
                    flags=re.DOTALL,
                ).rstrip()
                if memory_block:
                    existing.content = f"{cleaned}\n\n{memory_block}".strip() if cleaned else memory_block
                else:
                    existing.content = cleaned
            elif memory_block:
                existing.content = f"{existing.content}\n\n{memory_block}".strip()
            existing.pinned = True
            existing.token_count = count_message_tokens(existing)
        else:
            sys_msg = ChatMessage(
                role=MessageRole.SYSTEM,
                content=full_system_content,
                pinned=True,
            )
            # Insert at beginning
            ctx.messages.insert(0, sys_msg)

        return ctx

    # ==========================================================================
    # TURN DELTA SYNCHRONIZATION
    # ==========================================================================

    def sync_turn_delta(
        self, session_or_id: Union[str, SessionContext]
    ) -> List[FactTriple]:
        """
        Extract newly stated facts, user preferences, or decisions from recent messages
        in the session, and commit them to LongTermMemory.
        """
        if isinstance(session_or_id, str):
            ctx = self.session_memory.get_session(session_or_id)
        else:
            ctx = session_or_id

        if not ctx or not ctx.messages:
            return []

        # Analyze latest 4 messages for factual assertions
        recent_messages = ctx.messages[-4:]
        extracted_facts: List[FactTriple] = []

        patterns = [
            # "User prefers strictly remote work format"
            (re.compile(r"(?:user|пользователь)\s+prefers?\s+(.+)", re.IGNORECASE), "user", "User", "предпочтение"),
            (re.compile(r"(?:формат работы|work format)[:\s]+([^\n\.]+)", re.IGNORECASE), "user", "Формат работы", "значение"),
            (re.compile(r"(?:стек|tech stack)[:\s]+([^\n\.]+)", re.IGNORECASE), "user", "Стек технологий", "значение"),
            (re.compile(r"(?:decision|решение)[:\s]+([^\n\.]+)", re.IGNORECASE), "project", "Архитектурное решение", "решение"),
            (re.compile(r"(?:запомни|remember)[:\s]+([^\n\.]+)", re.IGNORECASE), "user", "Инструкция", "значение"),
        ]

        for msg in recent_messages:
            if msg.role not in (MessageRole.USER, MessageRole.ASSISTANT):
                continue
            text = msg.content.strip()

            for regex, ns, default_sub, default_pred in patterns:
                match = regex.search(text)
                if match:
                    val = match.group(1).strip()
                    if val and len(val) >= 3:
                        fact = FactTriple(
                            namespace=ns,
                            subject=default_sub,
                            predicate=default_pred,
                            object_value=val,
                            confidence=1.0,
                            source_citation=f"session:{ctx.session_id}",
                        )
                        # Store in long term memory
                        fact_id = self.long_term_memory.store_fact(fact)
                        fact.id = fact_id
                        extracted_facts.append(fact)

        return extracted_facts

    # ==========================================================================
    # CONVENIENCE HELPERS
    # ==========================================================================

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to the session working context."""
        self.session_memory.add_message(session_id, message)

    def add_tool_result(
        self, session_id: str, tool_call_id: str, output: Any, name: Optional[str] = None
    ) -> None:
        """Add truncated tool execution result to session."""
        processed_output, was_truncated = truncate_tool_result(output)
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content=str(processed_output),
            name=name,
            tool_call_id=tool_call_id,
            metadata={"truncated": was_truncated},
        )
        self.session_memory.add_message(session_id, msg)

    def record_episode(self, episode: EpisodicRecord) -> None:
        """Store task execution episode in long-term memory."""
        self.long_term_memory.record_episode(episode)

    def store_fact(self, fact: FactTriple, sync_vault: bool = True) -> int:
        """Store fact in long-term memory."""
        return self.long_term_memory.store_fact(fact, sync_vault=sync_vault)

    def replace_fact(
        self,
        old_fact_id_or_key: Union[int, str, dict],
        new_fact: FactTriple,
        sync_vault: bool = True,
    ) -> int:
        """Replace an existing fact with version tracking."""
        return self.long_term_memory.replace_fact(old_fact_id_or_key, new_fact, sync_vault=sync_vault)

    def delete_fact(self, fact_id: int, soft: bool = True) -> bool:
        """Delete fact from long-term memory."""
        return self.long_term_memory.delete_fact(fact_id, soft=soft)

    def get_fact(self, fact_id: int, include_deleted: bool = False) -> Optional[FactTriple]:
        """Retrieve a specific fact by ID."""
        return self.long_term_memory.get_fact(fact_id, include_deleted=include_deleted)

    def get_fact_history(self, fact_id: int) -> List[Dict[str, Any]]:
        """Retrieve revision audit history for a fact."""
        return self.long_term_memory.get_fact_history(fact_id)

    def search_facts(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 20,
        include_expired: bool = False,
        include_deleted: bool = False,
    ) -> List[FactTriple]:
        """Search facts via lexical FTS5 matching."""
        return self.long_term_memory.search_facts(
            query=query,
            namespace=namespace,
            limit=limit,
            include_expired=include_expired,
            include_deleted=include_deleted,
        )

    def recall_facts(
        self,
        query: Optional[str] = None,
        namespace: Optional[str] = None,
        min_confidence: float = 0.0,
        include_expired: bool = False,
        include_deleted: bool = False,
        sort_by: str = "relevance",
        limit: int = 20,
    ) -> List[FactTriple]:
        """Recall facts with advanced relational/lexical filtering."""
        return self.long_term_memory.recall_facts(
            query=query,
            namespace=namespace,
            min_confidence=min_confidence,
            include_expired=include_expired,
            include_deleted=include_deleted,
            sort_by=sort_by,
            limit=limit,
        )

    def list_facts(
        self,
        namespace: Optional[str] = None,
        limit: int = 100,
        include_deleted: bool = False,
        include_expired: bool = False,
    ) -> List[FactTriple]:
        """List stored facts."""
        return self.long_term_memory.list_facts(
            namespace=namespace,
            limit=limit,
            include_deleted=include_deleted,
            include_expired=include_expired,
        )
