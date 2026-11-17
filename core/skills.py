"""
core/skills.py
Progressive Disclosure Skills Subsystem.

Provides:
- parse_frontmatter(): Pure-Python YAML frontmatter parser using regular expressions.
- SkillInfo: Metadata dataclass describing an indexed skill.
- SkillsIndex: Filesystem discovery, lightweight progressive disclosure catalog,
  and secure on-demand loading of skill instructions with strict path-jail validation.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("core.skills")


class SkillError(Exception):
    """Base exception for skills subsystem errors."""
    pass


class SkillNotFoundError(SkillError, FileNotFoundError, KeyError):
    """
    Raised when a requested skill does not exist.
    Inherits from both FileNotFoundError and KeyError to support all standard lookup idioms.
    """
    pass


class SkillSecurityError(SkillError, ValueError, PermissionError):
    """
    Raised when a skill name or path violates security constraints or path jail.
    Inherits from both ValueError and PermissionError.
    """
    pass


@dataclass
class SkillInfo:
    """Metadata container for an indexed skill."""
    name: str
    description: str
    path: Path
    category: str = "general"
    metadata: Dict[str, str] = field(default_factory=dict)


def parse_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    """
    Pure-Python YAML frontmatter parser using standard library regular expressions.
    Extracts key-value pairs between opening and closing '---' delimiters.

    Returns:
        A tuple of (metadata_dict, body_markdown).
    """
    if not content:
        return {}, ""

    pattern = r"^\s*---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        return {}, content

    raw_meta, body = match.group(1), match.group(2)
    meta: Dict[str, str] = {}
    for line in raw_meta.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Strip outer matching quotes if present
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            meta[key] = val

    return meta, body


class SkillsIndex:
    """
    Progressive disclosure index of skills and operational guidelines.

    Scans skills_root for '<name>/SKILL.md' files and exposes:
    1. list_skills(): Lightweight catalog (name + description) for prompt / MCP progressive disclosure.
    2. skill_load(name): On-demand retrieval of full markdown instructions with path-jail enforcement.
    """

    def __init__(self, skills_root: Union[str, Path] = "skills"):
        root_path = Path(skills_root)
        if not root_path.is_absolute():
            # If default relative 'skills' is missing in cwd, check project root relative to this file
            if not root_path.exists():
                candidate = Path(__file__).resolve().parent.parent / root_path
                if candidate.exists():
                    root_path = candidate
        self.skills_root = root_path.resolve()
        self._skills: Dict[str, SkillInfo] = {}
        self.reload()

    def reload(self) -> None:
        """Scan skills_root and discover all skills/<name>/SKILL.md files."""
        self._skills.clear()
        if not self.skills_root.exists() or not self.skills_root.is_dir():
            logger.warning("Skills root directory %s does not exist or is not a directory.", self.skills_root)
            return

        for root, _, files in os.walk(self.skills_root):
            for filename in files:
                if filename.lower() == "skill.md":
                    file_path = Path(root) / filename
                    # Only discover skills located in subdirectories: skills/<name>/SKILL.md
                    if file_path.parent == self.skills_root:
                        continue
                    self._index_file(file_path)

    def _index_file(self, file_path: Path) -> None:
        try:
            content = file_path.read_text(encoding="utf-8")
            meta, _ = parse_frontmatter(content)

            # Determine skill name: from frontmatter 'name' or from parent directory name
            skill_name = meta.get("name") or file_path.parent.name
            skill_name = skill_name.strip()
            if not skill_name:
                return

            description = meta.get("description") or f"Skill instructions for {skill_name}"
            category = meta.get("category", "general")

            self._skills[skill_name] = SkillInfo(
                name=skill_name,
                description=description,
                path=file_path,
                category=category,
                metadata=meta,
            )
        except Exception as e:
            logger.warning("Failed to index skill file at %s: %s", file_path, e)

    def list_skills(self) -> List[Dict[str, str]]:
        """
        Return lightweight catalog [ {"name": ..., "description": ...} ]
        sorted alphabetically by name for progressive disclosure.
        """
        return [
            {"name": s.name, "description": s.description}
            for s in sorted(self._skills.values(), key=lambda s: s.name)
        ]

    def has_skill(self, name: str) -> bool:
        """Check if skill name is indexed."""
        return name in self._skills

    def get_skill_info(self, name: str) -> Optional[SkillInfo]:
        """Get SkillInfo metadata object if present, else None."""
        return self._skills.get(name)

    def skill_load(self, name: str) -> str:
        """
        Loads and returns the full markdown content of skills/<name>/SKILL.md.

        Security Enforcement:
        1. Name Validation: Strictly validates slug with ^[a-zA-Z0-9_-]+$ (rejects ../, slashes, null bytes).
        2. Path Jail: Ensures resolved path is strictly inside self.skills_root.
        3. Existence Check: Raises SkillNotFoundError (subclasses FileNotFoundError and KeyError).
        """
        if not isinstance(name, str):
            raise SkillSecurityError(f"Invalid skill name type: expected str, got {type(name).__name__}")

        clean_name = name.strip()
        if not clean_name or not re.match(r"^[a-zA-Z0-9_-]+$", clean_name):
            raise SkillSecurityError(
                f"Invalid skill name '{name}'. Name must be alphanumeric with underscores or hyphens."
            )

        # Locate target file path
        if clean_name in self._skills:
            target_path = self._skills[clean_name].path.resolve()
        else:
            target_path = (self.skills_root / clean_name / "SKILL.md").resolve()

        skills_root_resolved = self.skills_root.resolve()

        # Strict path jail check
        try:
            target_path.relative_to(skills_root_resolved)
        except ValueError:
            raise SkillSecurityError(
                f"Path jail violation: '{target_path}' is outside skills root '{skills_root_resolved}'."
            )

        if not target_path.is_file():
            available = ", ".join(sorted(self._skills.keys())) or "none"
            raise SkillNotFoundError(
                f"Skill '{clean_name}' not found at {target_path}. Available skills: {available}"
            )

        return target_path.read_text(encoding="utf-8")

    def load_skill(self, name: str) -> str:
        """Alias for skill_load."""
        return self.skill_load(name)

    def __getitem__(self, name: str) -> str:
        """Mapping lookup idiom: skills_index['skill-name'] -> full markdown."""
        return self.skill_load(name)

    def __contains__(self, name: str) -> bool:
        """Check whether skill name is in index."""
        return self.has_skill(name)

    def __len__(self) -> int:
        """Number of indexed skills."""
        return len(self._skills)
