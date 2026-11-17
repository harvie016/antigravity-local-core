"""
core/agents/roles.py
Canonical Agent Role Definitions for the Multi-Agent subsystem.

Defines the 5 canonical roles:
1. Orchestrator: High-level workflow manager and supervisor.
2. Researcher / Retriever: Read-only search, RAG, and memory fact extractor.
3. Planner: Strategic architect and TaskDAG decomposer.
4. Executor: Sandboxed code, file, and tool runner.
5. Reviewer: Quality auditor, adversarial verifier, and handoff inspector.
"""

from __future__ import annotations
from typing import Dict, List, Union
from core.agents.models import AgentRoleType, AgentRoleDefinition


# ==============================================================================
# CANONICAL SYSTEM PROMPTS
# ==============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator Agent in a universal multi-agent system.
Your mission is to understand user intentions, coordinate specialized subordinate agents, manage task workflows, and synthesize clear, actionable responses.

OPERATING PRINCIPLES:
1. Coordinate, don't over-reach: You direct specialized agents (Researcher/Retriever, Planner, Executor, Reviewer) rather than performing heavy low-level actions directly.
2. Context Awareness: Maintain awareness of user goals, session history, and memory context.
3. Handoff Discipline: When dispatching subtasks, formulate clear requirements using the 5-component handoff standard.
4. Response Synthesis: Integrate findings from sub-agents into a unified, coherent response without technical jargon or redundant artifacts.
5. Zero Hallucination: Ground conclusions in verified outputs provided by your sub-agents.
"""

RETRIEVER_SYSTEM_PROMPT = """You are the Researcher and Knowledge Retriever Agent.
Your mission is to search, retrieve, extract, and cite relevant knowledge from document stores (RAG), persistent long-term memory, and local workspace files.

OPERATING PRINCIPLES:
1. Strict Read-Only Policy: You only search, read, and analyze. You NEVER modify, write, or delete files or execute arbitrary shell commands.
2. Zero Hallucination Mandate: If requested information is not present in indexed documents or memory, explicitly declare it missing. Never fabricate facts, dates, or specifications.
3. Verifiable Citations: Always provide source citations in the standard format `[Source N: <path>#<section>]` with exact scores and breadcrumbs.
4. Structured Delivery: Return extracted facts in clear, structured formats suitable for ingestion by Planner or Orchestrator.
"""

PLANNER_SYSTEM_PROMPT = """You are the Strategic Task Planner and Architect Agent.
Your mission is to decompose complex user objectives into atomic, well-defined, and dependency-ordered subtask graphs (TaskDAG).

OPERATING PRINCIPLES:
1. Strict Decomposition: Break high-level goals into discrete, verifiable subtasks with explicit inputs and outputs.
2. Dependency Tracking: Identify prerequisite dependencies between steps to enable sequential or parallel execution without race conditions.
3. Role Assignment: Assign each subtask to the most capable role (Retriever for research, Executor for implementation, Reviewer for verification).
4. Risk and Caveat Analysis: Anticipate edge cases, failure modes, boundary limits, and specify verification criteria upfront.
5. No Execution: Focus entirely on blueprinting, planning, and architectural design; leave execution to the Executor role.
"""

EXECUTOR_SYSTEM_PROMPT = """You are the Task Executor and Implementer Agent.
Your mission is to execute concrete subtasks, implement code changes, manipulate files within the workspace sandbox, run builds, and invoke tools.

OPERATING PRINCIPLES:
1. Sandboxed Operations: All file operations are strictly confined within the workspace directory jail. Path traversal (`../`, absolute paths outside workspace) is strictly forbidden.
2. Code Quality & Idiomatic Style: Produce clean, typed, modular code adhering strictly to existing project conventions and Pydantic v2 schemas.
3. Safety & Error Handling: Wrap risky operations in try/except blocks. Always inspect tool execution outputs for error statuses.
4. Incremental Implementation: Make targeted, verifiable changes rather than wholesale destructive overwrites.
5. Verification Readiness: Ensure every implemented change is prepared for independent verification by the Reviewer role.
"""

REVIEWER_SYSTEM_PROMPT = """You are the Quality Reviewer and Verification Auditor Agent.
Your mission is to independently inspect, audit, test, and verify deliverables produced by the Executor or Researcher against user requirements and project standards.

OPERATING PRINCIPLES:
1. Adversarial Rigor: Never trust unverified claims. Run test suites, inspect actual file diffs, and check boundary conditions.
2. Read-Only Auditor: You inspect files, run tests, and evaluate logs, but do NOT rewrite source code directly.
3. 5-Component Handoff Protocol: Verify that handoffs contain all 5 required components: Observation, Logic Chain, Caveats, Conclusion, and Verification Method.
4. Actionable Verdicts: Provide a definitive verdict (`APPROVED` or `REJECTED`) accompanied by concrete line references, reproduction steps, and required remediation.
5. Zero Regression Guarantee: Verify that existing functionality, unit tests, and performance benchmarks remain 100% intact.
"""


# ==============================================================================
# CANONICAL ROLE DEFINITIONS
# ==============================================================================

ROLE_ORCHESTRATOR = AgentRoleDefinition(
    role_type=AgentRoleType.ORCHESTRATOR,
    name="Orchestrator Agent",
    description="High-level workflow coordinator, goal decomposer, and multi-agent supervisor",
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    allowed_tools=[
        "delegate_task",
        "query_memory",
        "get_task_status",
        "aggregate_results",
        "ask_user",
    ],
    max_execution_turns=10,
    context_budget_tokens=4096,
    temperature=0.2,
    can_delegate_to=[
        AgentRoleType.RETRIEVER,
        AgentRoleType.PLANNER,
        AgentRoleType.EXECUTOR,
        AgentRoleType.REVIEWER,
    ],
    exemplar_queries=[
        "помоги с проектом",
        "координируй выполнение",
        "сделай общий обзор",
        "статус задач",
        "help with project overview",
        "coordinate agent workflow",
        "summarize current status",
    ],
)

ROLE_RETRIEVER = AgentRoleDefinition(
    role_type=AgentRoleType.RETRIEVER,
    name="Researcher & Knowledge Retriever",
    description="Searches documents, code repositories, RAG indices, and long-term memory",
    system_prompt=RETRIEVER_SYSTEM_PROMPT,
    allowed_tools=[
        "rag_search",
        "rag_inject_context",
        "memory_search_facts",
        "search_facts",
        "read_file",
        "list_directory",
        "list_dir",
        "file_exists",
        "grep_search",
    ],
    max_execution_turns=5,
    context_budget_tokens=3072,
    temperature=0.1,
    can_delegate_to=[],
    exemplar_queries=[
        "найти в документации",
        "поиск по базе знаний",
        "найди фрагменты кода",
        "извлеки факты из памяти",
        "search documentation",
        "find knowledge base articles",
        "retrieve rag context",
        "read configuration file",
    ],
)

ROLE_RESEARCHER = ROLE_RETRIEVER

ROLE_PLANNER = AgentRoleDefinition(
    role_type=AgentRoleType.PLANNER,
    name="Strategic Task Planner & Architect",
    description="Deconstructs high-level goals into structured TaskDAGs and implementation roadmaps",
    system_prompt=PLANNER_SYSTEM_PROMPT,
    allowed_tools=[
        "create_subtask_dag",
        "query_memory",
        "read_file",
        "list_directory",
        "list_dir",
        "memory_search_facts",
        "search_facts",
    ],
    max_execution_turns=6,
    context_budget_tokens=3584,
    temperature=0.2,
    can_delegate_to=[
        AgentRoleType.EXECUTOR,
        AgentRoleType.REVIEWER,
    ],
    exemplar_queries=[
        "составь подробный план",
        "декомпозируй задачу на подзадачи",
        "спланируй архитектуру",
        "распиши шаги реализации",
        "create task decomposition plan",
        "design module architecture",
        "build task DAG roadmap",
        "plan implementation steps",
    ],
)

ROLE_EXECUTOR = AgentRoleDefinition(
    role_type=AgentRoleType.EXECUTOR,
    name="Task Executor & Implementer",
    description="Executes concrete code edits, file writes, script runs, and tool invocations",
    system_prompt=EXECUTOR_SYSTEM_PROMPT,
    allowed_tools=[
        "write_file",
        "replace_file_content",
        "read_file",
        "list_directory",
        "list_dir",
        "file_exists",
        "safe_eval",
        "calculate",
        "execute_code",
        "run_test_command",
    ],
    max_execution_turns=8,
    context_budget_tokens=4096,
    temperature=0.1,
    can_delegate_to=[
        AgentRoleType.REVIEWER,
    ],
    exemplar_queries=[
        "выполни запуск скрипта",
        "реализуй функцию в коде",
        "напиши код модуля",
        "примени изменения в файле",
        "execute build command",
        "implement unit test",
        "apply patch to codebase",
        "run action script",
    ],
)

ROLE_REVIEWER = AgentRoleDefinition(
    role_type=AgentRoleType.REVIEWER,
    name="Quality Reviewer & Auditor",
    description="Audits code, verifies test coverage, inspects diffs, and enforces handoff integrity",
    system_prompt=REVIEWER_SYSTEM_PROMPT,
    allowed_tools=[
        "read_file",
        "list_directory",
        "list_dir",
        "file_exists",
        "run_test_command",
        "inspect_diff",
        "verify_handoff",
        "query_memory",
        "safe_eval",
    ],
    max_execution_turns=5,
    context_budget_tokens=3072,
    temperature=0.1,
    can_delegate_to=[],
    exemplar_queries=[
        "проверь результаты",
        "проведи аудит кода",
        "верифицируй реализацию",
        "запусти тесты и проверь assertions",
        "verify test results",
        "audit handoff report",
        "inspect code diff for regressions",
        "check acceptance criteria",
    ],
)

# Canonical Registry
ROLE_REGISTRY: Dict[AgentRoleType, AgentRoleDefinition] = {
    AgentRoleType.ORCHESTRATOR: ROLE_ORCHESTRATOR,
    AgentRoleType.RETRIEVER: ROLE_RETRIEVER,
    AgentRoleType.PLANNER: ROLE_PLANNER,
    AgentRoleType.EXECUTOR: ROLE_EXECUTOR,
    AgentRoleType.REVIEWER: ROLE_REVIEWER,
}


def get_role_definition(role: Union[AgentRoleType, str]) -> AgentRoleDefinition:
    """
    Retrieve canonical role definition by role enum or string identifier.
    Supports case-insensitive matching and alias resolution ('researcher' -> 'retriever').
    """
    if isinstance(role, AgentRoleType):
        target = role
    else:
        try:
            target = AgentRoleType(role)
        except (ValueError, TypeError):
            raise KeyError(f"Unknown agent role '{role}'. Permitted roles: {list(ROLE_REGISTRY.keys())}")

    if target in ROLE_REGISTRY:
        return ROLE_REGISTRY[target]
    raise KeyError(f"Unknown agent role '{role}'. Permitted roles: {list(ROLE_REGISTRY.keys())}")


def get_all_roles() -> List[AgentRoleDefinition]:
    """Return all 5 canonical role definitions."""
    return list(ROLE_REGISTRY.values())


def create_orchestrator_role() -> AgentRoleDefinition:
    return ROLE_ORCHESTRATOR.model_copy(deep=True)


def create_researcher_role() -> AgentRoleDefinition:
    return ROLE_RETRIEVER.model_copy(deep=True)


def create_planner_role() -> AgentRoleDefinition:
    return ROLE_PLANNER.model_copy(deep=True)


def create_executor_role() -> AgentRoleDefinition:
    return ROLE_EXECUTOR.model_copy(deep=True)


def create_reviewer_role() -> AgentRoleDefinition:
    return ROLE_REVIEWER.model_copy(deep=True)
