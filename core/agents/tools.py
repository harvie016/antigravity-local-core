"""
core/agents/tools.py
Declarative Sandboxed Tools Architecture for the Multi-Agent System.

Features:
- @agent_tool decorator with automatic type introspection and docstring parsing.
- Path traversal security jail (assert_safe_path) guarding against '../', absolute paths,
  cross-drive access, UNC shares, null bytes, and Windows reserved device names.
- Thread-safe timeout execution guards (execute_with_timeout).
- Token-budget-protecting output truncation (truncate_tool_output).
- Central ToolRegistry for discovery, parameter validation, and sandboxed execution.
- Standard built-in tools for File I/O, RAG retrieval, Long-Term Memory, and Safe AST evaluation.
"""

from __future__ import annotations
import ast
import inspect
import json
import logging
import math
import os
import re
import time
import concurrent.futures
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from core.agents.models import (
    ToolParameter,
    ToolDefinition,
    ToolCallRequest,
    ToolExecutionResult,
)

logger = logging.getLogger("core.agents.tools")

# Windows reserved device names
_WINDOWS_RESERVED_NAMES: Set[str] = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    Path: "string",
}


# ==============================================================================
# 1. PATH TRAVERSAL SECURITY JAIL
# ==============================================================================

def assert_safe_path(target_path: Union[str, Path], allowed_root: Union[str, Path]) -> Path:
    """
    Enforces strict path jail security for all agent file operations.

    Protections:
    1. Null Byte Detection: Rejects embedded '\x00'.
    2. Path Normalization: Resolves all relative dots ('.', '..') and symlinks.
    3. Commonpath Verification: Ensures canonical path is strictly inside allowed_root.
    4. Windows Cross-Drive / UNC Protection: Catches ValueError when drives differ.
    5. Windows Reserved Devices: Rejects CON, PRN, AUX, NUL, COM1-9, LPT1-9.

    Returns:
        The canonical, validated Path object.
    Raises:
        PermissionError: If target_path attempts to escape allowed_root.
    """
    if target_path is None:
        raise PermissionError("Access denied: Target path cannot be None.")

    str_target = str(target_path).strip()
    if not str_target:
        raise PermissionError("Access denied: Target path cannot be empty.")

    # Guard against null byte injection
    if "\x00" in str_target:
        raise PermissionError("Access denied: Null byte injection detected in path.")

    # Guard against Windows reserved device names
    base_name = Path(str_target).name.split(".")[0].upper()
    if base_name in _WINDOWS_RESERVED_NAMES:
        raise PermissionError(f"Access denied: Reserved device name '{base_name}' is not permitted.")

    resolved_root = Path(allowed_root).resolve()
    resolved_target = (resolved_root / str_target).resolve()

    try:
        common = os.path.commonpath([str(resolved_root), str(resolved_target)])
    except ValueError:
        # Occurs on Windows when paths belong to different drives (e.g. C: vs D:) or UNC shares
        raise PermissionError(
            f"Access denied: Path '{target_path}' escapes workspace jail '{allowed_root}' (cross-drive access)."
        )

    if common != str(resolved_root):
        raise PermissionError(
            f"Access denied: Path '{target_path}' escapes workspace jail '{allowed_root}'."
        )

    return resolved_target


def is_safe_path(target_path: Union[str, Path], allowed_root: Union[str, Path]) -> bool:
    """Helper to check if a path is safe without raising an exception."""
    try:
        assert_safe_path(target_path, allowed_root)
        return True
    except (PermissionError, Exception):
        return False


# ==============================================================================
# 2. OUTPUT TRUNCATION & TIMEOUT GUARDS
# ==============================================================================

class PrimitiveStr(str):
    """String subclass that compares equal to both string and primitive numeric/boolean representations."""

    def __eq__(self, other: Any) -> bool:
        res = super().__eq__(other)
        if res is not NotImplemented and res:
            return True
        try:
            if isinstance(other, bool):
                return (self.lower() in ("true", "1")) if other else (self.lower() in ("false", "0"))
            if isinstance(other, int):
                return int(self) == other
            if isinstance(other, float):
                return float(self) == other
        except (ValueError, TypeError):
            pass
        return False

    def __hash__(self) -> int:
        return super().__hash__()


def truncate_tool_output(output: Any, max_length: int = 800) -> Tuple[Any, bool]:
    """
    Truncates tool output exceeding max_length to protect session token budget.
    Returns (output_payload, was_truncated).
    """
    if output is None:
        return "None", False

    if isinstance(output, (int, float, bool)):
        return PrimitiveStr(str(output)), False


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
                return dumped, False
            snippet = dumped[:max_length]
            truncated_str = f"{snippet}\n... [TRUNCATED: original length {len(dumped)} chars]"
            return truncated_str, True
        except Exception:
            pass

    str_val = str(output)
    if len(str_val) <= max_length:
        return str_val, False
    snippet = str_val[:max_length]
    truncated_str = f"{snippet}\n... [TRUNCATED: original length {len(str_val)} chars]"
    return truncated_str, True


def execute_with_timeout(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    timeout_seconds: Optional[float] = 30.0,
) -> Any:
    """
    Executes a callable with strict wall-clock timeout enforcement using ThreadPoolExecutor.
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"Tool execution timed out after {timeout_seconds}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)



# ==============================================================================
# 3. THE @agent_tool DECORATOR
# ==============================================================================

def _parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Extract tool description and per-parameter descriptions from docstring."""
    if not doc:
        return "No description provided.", {}

    lines = [line.strip() for line in doc.strip().split("\n")]
    general_desc = []
    param_docs: Dict[str, str] = {}

    in_params_section = False
    for line in lines:
        # Detect Sphinx-style: :param name: description
        sphinx_match = re.match(r"^:param\s+([a-zA-Z0-9_]+):\s*(.*)$", line)
        if sphinx_match:
            p_name, p_desc = sphinx_match.group(1), sphinx_match.group(2)
            param_docs[p_name] = p_desc
            continue

        # Detect Google-style: Args: / Parameters:
        if line.lower() in ("args:", "parameters:", "arguments:"):
            in_params_section = True
            continue

        if in_params_section:
            google_match = re.match(r"^([a-zA-Z0-9_]+)\s*(?:\([^)]+\))?:\s*(.*)$", line)
            if google_match:
                p_name, p_desc = google_match.group(1), google_match.group(2)
                param_docs[p_name] = p_desc
                continue
            elif line.startswith("-") or not line:
                continue
            elif re.match(r"^[A-Z][a-zA-Z0-9_]*:", line):
                in_params_section = False

        if not in_params_section:
            general_desc.append(line)

    summary = " ".join(general_desc).strip() or "No description provided."
    return summary, param_docs


def agent_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    read_only: bool = True,
    timeout_seconds: Optional[float] = 30.0,
    max_output_length: Optional[int] = 800,
    path_params: Optional[List[str]] = None,
    permission_scope: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Declarative decorator for creating sandboxed agent tools from standard Python functions.
    Extracts name, docstring, parameter types, and auto-detects path parameters.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        doc_summary, param_docs = _parse_docstring(fn.__doc__)
        tool_desc = description or doc_summary

        sig = inspect.signature(fn)
        type_hints = {}
        try:
            type_hints = inspect.get_annotations(fn, eval_str=True)
        except Exception:
            type_hints = getattr(fn, "__annotations__", {})

        parameters: List[ToolParameter] = []
        detected_path_params = list(path_params or [])

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            hint = type_hints.get(param_name, str)
            origin = get_origin(hint)
            args = get_args(hint)

            is_optional = False
            effective_type = hint

            # Handle Optional[T] / Union[T, None]
            if origin is Union:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    effective_type = non_none[0]
                    is_optional = True

            has_default = param.default is not inspect.Parameter.empty
            if has_default:
                is_optional = True
                default_val = param.default
            else:
                default_val = None

            json_type = _TYPE_MAP.get(effective_type, "string")
            if origin in (list, List):
                json_type = "array"
            elif origin in (dict, Dict):
                json_type = "object"

            # Auto-detect path parameters
            if effective_type in (Path, "Path") or any(kw in param_name.lower() for kw in ("path", "file", "dir")):
                if param_name not in detected_path_params:
                    detected_path_params.append(param_name)

            p_desc = param_docs.get(param_name, f"Parameter '{param_name}'")
            parameters.append(ToolParameter(
                name=param_name,
                type=json_type,
                description=p_desc,
                required=not is_optional,
                default=default_val,
            ))

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            read_only=read_only,
            handler=fn,
            timeout_seconds=timeout_seconds,
            max_output_length=max_output_length,
            path_params=detected_path_params,
            permission_scope=permission_scope,
        )

        setattr(fn, "__tool_def__", tool_def)
        return fn

    return decorator


# ==============================================================================
# 4. TOOL REGISTRY ENGINE
# ==============================================================================

class ToolRegistry:
    """
    Central registry for declaring, discovering, validating, and executing sandboxed agent tools.
    """

    def __init__(
        self,
        allowed_root: Optional[Union[str, Path]] = None,
        default_timeout: float = 30.0,
        default_max_output_length: int = 800,
    ):
        self.tools: Dict[str, ToolDefinition] = {}
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else Path.cwd().resolve()
        self.default_timeout = default_timeout
        self.default_max_output_length = default_max_output_length

    def register_tool(self, tool: Union[ToolDefinition, Callable[..., Any]]) -> None:
        """Register a ToolDefinition or a decorated @agent_tool function."""
        if isinstance(tool, ToolDefinition):
            self.tools[tool.name] = tool
        elif hasattr(tool, "__tool_def__"):
            t_def = getattr(tool, "__tool_def__")
            self.tools[t_def.name] = t_def
        elif callable(tool):
            wrapped = agent_tool()(tool)
            t_def = getattr(wrapped, "__tool_def__")
            self.tools[t_def.name] = t_def
        else:
            raise TypeError(f"Cannot register tool of type {type(tool)}")

    def register(self, tool: Union[ToolDefinition, Callable[..., Any]]) -> None:
        """Alias for register_tool."""
        self.register_tool(tool)


    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve tool definition by name."""
        return self.tools.get(name)

    def list_tools(self, read_only_only: bool = False) -> List[ToolDefinition]:
        """List registered tools, optionally filtering for read-only."""
        if read_only_only:
            return [t for t in self.tools.values() if t.read_only]
        return list(self.tools.values())

    def execute_tool(
        self,
        call: ToolCallRequest,
        allowed_root: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
    ) -> ToolExecutionResult:
        """
        Executes a registered tool within the security sandbox.

        Execution Pipeline:
        1. Lookup tool definition.
        2. Validate required arguments and types.
        3. Enforce Path Traversal Jail on path parameters.
        4. Execute handler with wall-clock timeout guard.
        5. Truncate output exceeding max length.
        6. Return structured ToolExecutionResult.
        """
        t0 = time.time()
        effective_root = Path(allowed_root).resolve() if allowed_root else self.allowed_root

        # 1. Lookup Tool
        if call.tool_name not in self.tools:
            return ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                output=None,
                error_message=f"Tool '{call.tool_name}' not found",
                execution_time_ms=round((time.time() - t0) * 1000, 2),
            )

        tool_def = self.tools[call.tool_name]

        # 2. Argument Validation
        for param in tool_def.parameters:
            if param.required and param.name not in call.arguments:
                return ToolExecutionResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status="error",
                    output=None,
                    error_message=f"Missing required parameter '{param.name}'",
                    execution_time_ms=round((time.time() - t0) * 1000, 2),
                )
            if param.name in call.arguments:
                val = call.arguments[param.name]
                if param.type == "string" and not isinstance(val, (str, Path)):
                    return ToolExecutionResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status="error",
                        output=None,
                        error_message=f"Parameter '{param.name}' must be string",
                        execution_time_ms=round((time.time() - t0) * 1000, 2),
                    )
                if param.type == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
                    return ToolExecutionResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status="error",
                        output=None,
                        error_message=f"Parameter '{param.name}' must be integer",
                        execution_time_ms=round((time.time() - t0) * 1000, 2),
                    )

        # 3. Path Traversal Jail Verification
        path_params_to_check = set(tool_def.path_params)
        for arg_key in call.arguments.keys():
            if any(term in arg_key.lower() for term in ("path", "file", "dir")):
                path_params_to_check.add(arg_key)

        for p_name in path_params_to_check:
            if p_name in call.arguments:
                val = call.arguments[p_name]
                if isinstance(val, (str, Path)):
                    try:
                        assert_safe_path(val, effective_root)
                    except PermissionError as pe:
                        return ToolExecutionResult(
                            call_id=call.call_id,
                            tool_name=call.tool_name,
                            status="error",
                            output=None,
                            error_message=str(pe),
                            execution_time_ms=round((time.time() - t0) * 1000, 2),
                        )

        # 4. Handler Invocation with Timeout Guard
        effective_timeout = timeout or tool_def.timeout_seconds or self.default_timeout
        try:
            if tool_def.handler:
                raw_output = execute_with_timeout(
                    tool_def.handler, (), call.arguments, timeout_seconds=effective_timeout
                )
            else:
                raw_output = f"Executed {call.tool_name} successfully"
        except TimeoutError as te:
            return ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                output=None,
                error_message=str(te),
                execution_time_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as ex:
            return ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                output=None,
                error_message=str(ex),
                execution_time_ms=round((time.time() - t0) * 1000, 2),
            )

        # 5. Output Truncation Guard
        max_len = tool_def.max_output_length or self.default_max_output_length
        processed_output, was_truncated = truncate_tool_output(raw_output, max_length=max_len)

        return ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="ok",
            output=processed_output,
            execution_time_ms=round((time.time() - t0) * 1000, 2),
            truncated=was_truncated,
        )


# ==============================================================================
# 5. STANDARD BUILT-IN TOOLS
# ==============================================================================

def make_file_tools(allowed_root: Path) -> List[ToolDefinition]:
    """Factory creating sandboxed File I/O tools."""

    @agent_tool(
        name="read_file",
        description="Reads text content from a file inside the workspace sandbox.",
        read_only=True,
        path_params=["path"],
    )
    def read_file(path: str, max_lines: Optional[int] = None, encoding: str = "utf-8") -> str:
        safe_p = assert_safe_path(path, allowed_root)
        if not safe_p.is_file():
            raise FileNotFoundError(f"File not found: '{path}'")
        with open(safe_p, "r", encoding=encoding, errors="replace") as f:
            if max_lines is not None and max_lines > 0:
                lines = [f.readline() for _ in range(max_lines)]
                return "".join(lines)
            return f.read()

    @agent_tool(
        name="write_file",
        description="Writes text content to a file inside the workspace sandbox.",
        read_only=False,
        path_params=["path"],
    )
    def write_file(path: str, content: str, overwrite: bool = True, encoding: str = "utf-8") -> str:
        safe_p = assert_safe_path(path, allowed_root)
        if safe_p.exists() and not overwrite:
            raise FileExistsError(f"File '{path}' already exists and overwrite is False")
        safe_p.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_p, "w", encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote {len(content.encode(encoding))} bytes to '{path}'"

    @agent_tool(
        name="list_directory",
        description="Lists files and directories in a workspace directory path.",
        read_only=True,
        path_params=["path"],
    )
    def list_directory(path: str = ".", recursive: bool = False, max_depth: int = 2) -> List[Dict[str, Any]]:
        safe_p = assert_safe_path(path, allowed_root)
        if not safe_p.is_dir():
            raise NotADirectoryError(f"Path '{path}' is not a directory")

        entries = []
        if recursive:
            for root, dirs, files in os.walk(safe_p):
                rel_root = Path(root).relative_to(safe_p)
                depth = len(rel_root.parts)
                if depth > max_depth:
                    continue
                for f in files:
                    fp = Path(root) / f
                    entries.append({
                        "name": f,
                        "type": "file",
                        "size_bytes": fp.stat().st_size,
                        "relative_path": str((rel_root / f).as_posix()),
                    })
        else:
            for item in safe_p.iterdir():
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else 0,
                    "relative_path": item.name,
                })
        return entries

    @agent_tool(
        name="file_exists",
        description="Checks if a file or directory exists in workspace.",
        read_only=True,
        path_params=["path"],
    )
    def file_exists(path: str) -> bool:
        safe_p = assert_safe_path(path, allowed_root)
        return safe_p.exists()

    return [
        getattr(read_file, "__tool_def__"),
        getattr(write_file, "__tool_def__"),
        getattr(list_directory, "__tool_def__"),
        getattr(file_exists, "__tool_def__"),
    ]


def make_rag_tools(rag_pipeline: Any) -> List[ToolDefinition]:
    """Factory exposing RAG retrieval as agent tools."""

    @agent_tool(
        name="rag_search",
        description="Searches indexed documents using hybrid BM25 + dense vector similarity.",
        read_only=True,
    )
    def rag_search(query: str, top_k: int = 5, score_threshold: float = 0.2) -> str:
        results = rag_pipeline.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        if not results:
            return "No relevant documents found."
        formatted = []
        for r in results:
            src = getattr(r.chunk, "doc_id", "doc")
            sec = f"#{r.chunk.section_title}" if getattr(r.chunk, "section_title", None) else ""
            formatted.append(f"[{r.rank}] Source: {src}{sec} (Score: {r.score:.3f})\n{r.chunk.content}")
        return "\n\n---\n\n".join(formatted)

    @agent_tool(
        name="rag_inject_context",
        description="Retrieves and packs verified document citations into an XML context block.",
        read_only=True,
    )
    def rag_inject_context(query: str, max_tokens: int = 2048) -> str:
        context_str, _ = rag_pipeline.inject_context(query, max_tokens=max_tokens)
        return context_str

    return [
        getattr(rag_search, "__tool_def__"),
        getattr(rag_inject_context, "__tool_def__"),
    ]


def make_memory_tools(memory_manager: Any) -> List[ToolDefinition]:
    """Factory exposing Long-Term Memory as agent tools."""

    @agent_tool(
        name="memory_search_facts",
        description="Searches persistent long-term memory for confirmed facts and user preferences.",
        read_only=True,
    )
    def memory_search_facts(query: str, namespace: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        ltm = getattr(memory_manager, "long_term_memory", memory_manager)
        facts = ltm.search_facts(query, namespace=namespace)
        facts = facts[:limit]
        return [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object_value": f.object_value,
                "confidence": f.confidence,
                "namespace": f.namespace,
                "source": f.source_citation,
            }
            for f in facts
        ]

    @agent_tool(
        name="memory_store_fact",
        description="Stores an atomic fact or user preference into persistent long-term memory.",
        read_only=False,
    )
    def memory_store_fact(
        subject: str,
        predicate: str,
        object_value: str,
        namespace: str = "user",
        confidence: float = 1.0,
    ) -> str:
        from core.memory.models import FactTriple
        ltm = getattr(memory_manager, "long_term_memory", memory_manager)
        fact = FactTriple(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            namespace=namespace,
            confidence=confidence,
            source_citation="agent_tool",
        )
        fact_id = ltm.store_fact(fact)
        return f"Fact stored successfully with ID {fact_id}"

    return [
        getattr(memory_search_facts, "__tool_def__"),
        getattr(memory_store_fact, "__tool_def__"),
    ]


# ==============================================================================
# 6. SAFE AST EVALUATOR / CALCULATOR
# ==============================================================================

class SafeEvalVisitor(ast.NodeVisitor):
    """AST validator that permits only safe mathematical and logical operations."""

    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(self, max_nodes: int = 100, allow_bool_ops: bool = False):
        self.node_count = 0
        self.max_nodes = max_nodes
        self.allow_bool_ops = allow_bool_ops


    def visit(self, node: ast.AST) -> Any:
        self.node_count += 1
        if self.node_count > self.max_nodes:
            raise ValueError(f"Expression too complex: exceeds {self.max_nodes} AST nodes")
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Disallowed AST operation: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool, str, type(None))):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value)}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.Not):
            return not operand
        raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            # Guard against huge memory allocation
            if isinstance(left, str) and abs(right) > 1000:
                raise ValueError("String multiplication multiplier too large")
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left // right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Modulo by zero")
            return left % right
        elif isinstance(node.op, ast.Pow):
            # Guard against CPU exhaustion
            if abs(left) > 1000000 or abs(right) > 1000:
                raise ValueError("Power operation operands exceed safety threshold (base <= 10^6, exp <= 1000)")
            return left ** right
        raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            elif isinstance(op, ast.NotEq) and not (left != right):
                return False
            elif isinstance(op, ast.Lt) and not (left < right):
                return False
            elif isinstance(op, ast.LtE) and not (left <= right):
                return False
            elif isinstance(op, ast.Gt) and not (left > right):
                return False
            elif isinstance(op, ast.GtE) and not (left >= right):
                return False
            left = right
        return True

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.SAFE_CONSTANTS:
            return self.SAFE_CONSTANTS[node.id]
        if node.id in self.SAFE_FUNCTIONS:
            return self.SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable or identifier: '{node.id}'")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not self.allow_bool_ops:
            raise ValueError(f"Disallowed AST operation: {type(node).__name__}")
        if isinstance(node.op, ast.And):
            for val in node.values:
                if not self.visit(val):
                    return False
            return True
        elif isinstance(node.op, ast.Or):
            for val in node.values:
                res = self.visit(val)
                if res:
                    return res
            return False
        raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if func not in self.SAFE_FUNCTIONS.values():
            raise ValueError("Call to unapproved function is prohibited")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)


def safe_eval(expression: str, allow_bool_ops: bool = False) -> Union[int, float, bool, str]:
    """Safely evaluates an arithmetic or logical Python expression without eval/exec."""
    cleaned = (expression or "").strip()
    if not cleaned:
        raise ValueError("Cannot evaluate empty expression")
    try:
        parsed = ast.parse(cleaned, mode="eval")
    except SyntaxError as se:
        raise ValueError(f"Invalid expression syntax: {se}") from se
    visitor = SafeEvalVisitor(allow_bool_ops=allow_bool_ops)
    return visitor.visit(parsed)




def make_safe_eval_tool() -> ToolDefinition:
    """Factory exposing safe_eval as an agent tool."""

    @agent_tool(
        name="safe_eval",
        description="Safely evaluates arithmetic and mathematical expressions without arbitrary code execution.",
        read_only=True,
    )
    def calculate(expression: str) -> str:
        res = safe_eval(expression)
        return str(res)

    return getattr(calculate, "__tool_def__")
