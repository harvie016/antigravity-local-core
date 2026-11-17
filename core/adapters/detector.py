"""
core/adapters/detector.py
Multi-Environment Auto-Detection Subsystem (Milestone 4).
Detects active runtime environment across Antigravity, Claude Code, Codex, and local execution.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, Union

from core.adapters.antigravity_adapter import AntigravityEnvironmentAdapter
from core.adapters.base import BaseEnvironmentAdapter, GenericLocalAdapter
from core.adapters.claude_adapter import ClaudeCodeEnvironmentAdapter
from core.adapters.codex_adapter import CodexEnvironmentAdapter
from core.adapters.models import EnvironmentType

logger = logging.getLogger("core.adapters.detector")


class EnvironmentDetector:
    """
    Auto-detects the active AI assistant or IDE execution environment.
    Evaluation order:
    1. Explicit CLI argument / function override (--env flag)
    2. Environment variables (AGENT_ENV, ANTIGRAVITY_ENV, CLAUDE_CODE, OPENAI_API_BASE)
    3. Filesystem markers (.agents/teamwork, claude_skills/manifest.json, .claude)
    4. Default fallback: generic_local
    """

    @classmethod
    def detect(
        cls,
        cli_arg: Optional[str] = None,
        base_path: Optional[Union[str, Path]] = None,
    ) -> EnvironmentType:
        """Detect the active environment type."""
        # 1. Explicit CLI Argument
        if cli_arg:
            clean_arg = cli_arg.strip().lower()
            if clean_arg in ("antigravity", "gemini"):
                return EnvironmentType.ANTIGRAVITY
            if clean_arg in ("claude", "claude_code", "anthropic"):
                return EnvironmentType.CLAUDE
            if clean_arg in ("codex", "openai", "ollama"):
                return EnvironmentType.CODEX
            if clean_arg in ("generic_local", "local", "generic"):
                return EnvironmentType.GENERIC_LOCAL
            if clean_arg == "auto":
                pass  # Continue to auto-detection
            else:
                try:
                    return EnvironmentType(clean_arg)
                except ValueError:
                    logger.warning("Unrecognized --env argument '%s', falling back to auto-detect", cli_arg)

        # 2. Environment Variables
        env_var = os.getenv("AGENT_ENV") or os.getenv("ENVIRONMENT")
        if env_var:
            clean_var = env_var.strip().lower()
            try:
                return EnvironmentType(clean_var)
            except ValueError:
                pass

        if os.getenv("ANTIGRAVITY_ENV") or os.getenv("ANTIGRAVITY_AGENT_ID"):
            return EnvironmentType.ANTIGRAVITY

        if os.getenv("CLAUDE_CODE") or os.getenv("CLAUDE_PROJECT_DIR"):
            return EnvironmentType.CLAUDE

        if os.getenv("CODEX_ENV") or os.getenv("OPENAI_API_BASE") or os.getenv("OLLAMA_HOST"):
            return EnvironmentType.CODEX

        # 3. Filesystem Markers
        root = Path(base_path) if base_path else Path.cwd()

        # Check Antigravity teamwork metadata
        if (root / ".agents" / "teamwork").is_dir():
            return EnvironmentType.ANTIGRAVITY

        # Check Claude Code manifest or command folder
        if (root / "claude_skills" / "manifest.json").is_file() or (root / ".claude").is_dir():
            return EnvironmentType.CLAUDE

        # Check Codex / OpenAI / Ollama files
        if (root / "openai_tools.json").is_file() or (root / "Modelfile").is_file():
            return EnvironmentType.CODEX

        # 4. Default Fallback
        return EnvironmentType.GENERIC_LOCAL

    @classmethod
    def get_adapter(
        cls,
        env_type: Optional[Union[EnvironmentType, str]] = None,
        cli_arg: Optional[str] = None,
        base_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> BaseEnvironmentAdapter:
        """
        Factory to resolve environment and instantiate the corresponding adapter.
        """
        if env_type is not None and env_type != EnvironmentType.AUTO and env_type != "auto":
            resolved_type = (
                EnvironmentType(env_type.lower()) if isinstance(env_type, str) else env_type
            )
        else:
            resolved_type = cls.detect(cli_arg=cli_arg, base_path=base_path)

        if resolved_type == EnvironmentType.ANTIGRAVITY:
            return AntigravityEnvironmentAdapter()
        elif resolved_type == EnvironmentType.CLAUDE:
            return ClaudeCodeEnvironmentAdapter()
        elif resolved_type == EnvironmentType.CODEX:
            return CodexEnvironmentAdapter(**kwargs)
        else:
            return GenericLocalAdapter()


def get_adapter(
    env: Union[EnvironmentType, str] = EnvironmentType.AUTO, **kwargs
) -> BaseEnvironmentAdapter:
    """Convenience helper to instantiate an environment adapter."""
    return EnvironmentDetector.get_adapter(env_type=env, **kwargs)
