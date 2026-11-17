"""
Memory Subsystem for Universal Multi-Agent System.
Exports:
- Models: MessageRole, ChatMessage, SessionContext, FactTriple, EpisodicRecord
- Session Memory: SessionMemoryManager, count_tokens, count_message_tokens, trim_session_context, truncate_tool_result
- Long-Term Memory: LongTermMemoryManager
- Facade: MemoryManager
"""

from core.memory.models import (
    MessageRole,
    ChatMessage,
    SessionContext,
    FactTriple,
    EpisodicRecord,
)
from core.memory.session_memory import (
    SessionMemoryManager,
    count_tokens,
    count_message_tokens,
    trim_session_context,
    truncate_tool_result,
)
from core.memory.long_term_memory import LongTermMemoryManager
from core.memory.manager import MemoryManager

__all__ = [
    "MessageRole",
    "ChatMessage",
    "SessionContext",
    "FactTriple",
    "EpisodicRecord",
    "SessionMemoryManager",
    "count_tokens",
    "count_message_tokens",
    "trim_session_context",
    "truncate_tool_result",
    "LongTermMemoryManager",
    "MemoryManager",
]
