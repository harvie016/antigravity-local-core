"""
Session Memory Manager for ephemeral working context.
Implements:
- Exact token counting via tiktoken (cl100k_base) with deterministic Cyrillic/Latin character fallback.
- Sliding window buffer trimming (trim_session_context): FIFO eviction of older unpinned messages,
  strictly preserving pinned=True and system prompt.
- Rolling summary accumulation from evicted content.
- Tool result truncation for outputs exceeding max length (default 800 chars).
- Thread-safe Session CRUD and in-memory caching.
"""

from __future__ import annotations
import json
import math
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Tuple

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

from core.memory.models import MessageRole, ChatMessage, SessionContext


# ==============================================================================
# TOKEN COUNTING ENGINE
# ==============================================================================

_ENCODER_CACHE: Dict[str, Any] = {}
_ENCODER_LOCK = threading.RLock()


def _get_tiktoken_encoder(encoding_name: str = "cl100k_base") -> Optional[Any]:
    """Retrieve or cache a tiktoken encoding instance thread-safely."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    with _ENCODER_LOCK:
        if encoding_name not in _ENCODER_CACHE:
            try:
                _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                return None
        return _ENCODER_CACHE.get(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens accurately using tiktoken cl100k_base.
    If tiktoken is unavailable, falls back to a deterministic character estimator
    calibrated for English (~3.8 chars/token) and Cyrillic (~1.8 chars/token).
    """
    if not text:
        return 0

    encoder = _get_tiktoken_encoder(encoding_name)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Deterministic character fallback estimator
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    other_count = len(text) - cyrillic_count
    estimated = int(math.ceil(cyrillic_count / 1.8 + other_count / 3.8))
    return max(1, estimated)


def count_message_tokens(message: ChatMessage, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate tokens for a ChatMessage including content, role, and tool attachments.
    Includes standard turn overhead (~4 tokens).
    """
    # 4 tokens base turn framing (role, formatting)
    tokens = 4
    if message.content:
        tokens += count_tokens(message.content, encoding_name)
    if message.name:
        tokens += count_tokens(message.name, encoding_name)
    if message.tool_call_id:
        tokens += count_tokens(message.tool_call_id, encoding_name)
    if message.tool_calls:
        try:
            serialized_calls = json.dumps(message.tool_calls, ensure_ascii=False)
            tokens += count_tokens(serialized_calls, encoding_name)
        except Exception:
            tokens += 10
    return max(1, tokens)


# ==============================================================================
# TOOL RESULT TRUNCATION
# ==============================================================================

def truncate_tool_result(output: Any, max_length: int = 800) -> Tuple[Any, bool]:
    """
    Truncate tool execution outputs exceeding max_length (default 800 chars)
    to protect session context budgets from exploding.
    Returns (processed_output, was_truncated).
    """
    if output is None:
        return None, False

    if isinstance(output, str):
        if len(output) <= max_length:
            return output, False
        snippet = output[:max_length]
        truncated_str = f"{snippet}\n... [TRUNCATED: original length {len(output)} chars]"
        return truncated_str, True

    if isinstance(output, (dict, list)):
        try:
            dumped = json.dumps(output, ensure_ascii=False)
            if len(dumped) <= max_length:
                return output, False
            snippet = dumped[:max_length]
            truncated_str = f"{snippet}\n... [TRUNCATED: original length {len(dumped)} chars]"
            return truncated_str, True
        except Exception:
            pass

    # Generic string conversion fallback
    str_val = str(output)
    if len(str_val) <= max_length:
        return output, False
    snippet = str_val[:max_length]
    truncated_str = f"{snippet}\n... [TRUNCATED: original length {len(str_val)} chars]"
    return truncated_str, True


# ==============================================================================
# SLIDING WINDOW PRUNING & ROLLING SUMMARY
# ==============================================================================

def trim_session_context(context: SessionContext, budget: Optional[int] = None) -> None:
    """
    Sliding window buffer trimming algorithm:
    1. Computes total token count of all messages.
    2. Strictly preserves pinned=True messages and system instructions (role=system).
    3. Evicts older unpinned messages via FIFO.
    4. Accumulates evicted content into context.rolling_summary.
    """
    target_budget = budget if budget is not None else context.max_token_budget
    total_tokens = sum(msg.token_count for msg in context.messages)
    if total_tokens <= target_budget:
        return

    evicted_texts: List[str] = []
    i = 0
    # FIFO scan from oldest to newest
    while total_tokens > target_budget and i < len(context.messages):
        msg = context.messages[i]
        # Pinned messages and system prompt must NEVER be evicted
        if not msg.pinned and msg.role != MessageRole.SYSTEM:
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            evicted_texts.append(f"{role_val}: {msg.content}")
            total_tokens -= msg.token_count
            context.messages.pop(i)
        else:
            i += 1

    # Accumulate evicted content into rolling summary
    if evicted_texts:
        delta_snippet = " | ".join(evicted_texts[:5])
        if len(delta_snippet) > 250:
            delta_snippet = delta_snippet[:250] + "..."
        if context.rolling_summary:
            context.rolling_summary += f" [Summary: {delta_snippet}]"
        else:
            context.rolling_summary = f"[Summary: {delta_snippet}]"


# ==============================================================================
# SESSION MEMORY MANAGER
# ==============================================================================

class SessionMemoryManager:
    """
    Ephemeral working context manager.
    Coordinates session lifecycle, token accounting, sliding window eviction,
    and in-memory session caching.
    """

    def __init__(self, default_budget: int = 4096, cache_size: int = 1000):
        self.default_budget = default_budget
        self.cache_size = cache_size
        self.sessions: Dict[str, SessionContext] = {}
        self._lock = threading.RLock()

    def _estimate_tokens(self, text: str) -> int:
        """Helper for token estimation backwards-compatibility."""
        return count_tokens(text)

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
        max_token_budget: Optional[int] = None,
    ) -> SessionContext:
        """Create and register a new conversation session."""
        with self._lock:
            s_id = session_id or str(uuid.uuid4())
            ctx = SessionContext(
                session_id=s_id,
                user_id=user_id,
                max_token_budget=max_token_budget or self.default_budget,
            )
            self.sessions[s_id] = ctx
            self._enforce_cache_limit()
            return ctx

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve existing session context by ID without modifying it."""
        with self._lock:
            return self.sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str = "default_user",
        max_budget: Optional[int] = None,
    ) -> SessionContext:
        """Retrieve existing session or instantiate a new one."""
        with self._lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
            return self.create_session(
                session_id=session_id,
                user_id=user_id,
                max_token_budget=max_budget or self.default_budget,
            )

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """
        Append a conversation turn to the session, compute its token cost,
        and execute sliding window trimming if token budget is exceeded.
        """
        with self._lock:
            ctx = self.get_or_create_session(session_id)
            if message.token_count <= 0:
                message.token_count = count_message_tokens(message)
            ctx.messages.append(message)
            ctx.updated_at = time.time()
            trim_session_context(ctx)

    def get_working_context(
        self, session_id: str, max_budget: Optional[int] = None
    ) -> SessionContext:
        """
        Retrieve session working context, applying an optional custom token budget
        to trim history before feeding into agent prompting.
        """
        with self._lock:
            ctx = self.get_or_create_session(session_id)
            if max_budget is not None:
                clone_ctx = ctx.model_copy(deep=True)
                trim_session_context(clone_ctx, budget=max_budget)
                return clone_ctx
            return ctx

    def update_session(self, ctx: SessionContext) -> None:
        """Persist or update an in-memory session context."""
        with self._lock:
            ctx.updated_at = time.time()
            self.sessions[ctx.session_id] = ctx
            self._enforce_cache_limit()

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from memory."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        """Return list of active session IDs."""
        with self._lock:
            return list(self.sessions.keys())

    def clear_all(self) -> None:
        """Clear all in-memory sessions."""
        with self._lock:
            self.sessions.clear()

    def _enforce_cache_limit(self) -> None:
        """Prune oldest sessions if in-memory cache exceeds cache_size."""
        if len(self.sessions) <= self.cache_size:
            return
        # Sort by updated_at ascending and remove oldest
        sorted_keys = sorted(
            self.sessions.keys(),
            key=lambda k: float(
                self.sessions[k].updated_at
                if isinstance(self.sessions[k].updated_at, (int, float))
                else 0.0
            ),
        )
        overflow_count = len(self.sessions) - self.cache_size
        for k in sorted_keys[:overflow_count]:
            self.sessions.pop(k, None)
