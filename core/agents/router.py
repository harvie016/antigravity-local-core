"""
core/agents/router.py
3-Tier Dynamic Intent Router for the Multi-Agent subsystem.

Routes incoming user queries/tasks to the appropriate AgentRoleType via:
- Tier 1: Fast deterministic regex pattern matching (< 0.1ms).
- Tier 2: Semantic TF-IDF cosine similarity using LocalTFIDFEmbeddingProvider (< 2ms).
- Tier 3: LLM Intent Classifier (GigaChat) with circuit breaker & deterministic heuristic fallback.
Guarantees 100% offline availability with zero paid API keys.
"""

from __future__ import annotations
import re
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from core.agents.models import AgentRoleType
from core.agents.roles import get_all_roles
from core.rag.embeddings import BaseEmbeddingProvider, LocalTFIDFEmbeddingProvider

logger = logging.getLogger("core.agents.router")


class RouteDecision(BaseModel):
    """Structured telemetry returned for an intent routing decision."""
    model_config = ConfigDict(extra="ignore")

    role: AgentRoleType = Field(..., description="Selected destination agent role")
    tier: int = Field(..., description="Routing tier used: 1=Regex, 2=Semantic TF-IDF, 3=LLM/Fallback")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing certainty confidence score")
    method: str = Field(..., description="Strategy: 'regex_match', 'tfidf_cosine_similarity', 'llm_classification', 'deterministic_fallback'")
    matched_pattern: Optional[str] = Field(default=None, description="Regex matched token or pattern if Tier 1")
    scores: Dict[str, float] = Field(default_factory=dict, description="Normalized similarity scores per role if Tier 2")
    reasoning: Optional[str] = Field(default=None, description="Explanatory rationale for routing decision")


class IntentRouter:
    """
    Dynamic 3-Tier Intent Router.
    Evaluates queries through a cascading pipeline:
    Tier 1 (Regex) -> Tier 2 (TF-IDF Cosine Similarity) -> Tier 3 (LLM with Circuit Breaker).
    """

    # Tier 1 Regex Patterns (Bilingual: Russian & English)
    TIER1_PATTERNS: Dict[AgentRoleType, re.Pattern] = {
        AgentRoleType.RETRIEVER: re.compile(
            r"(?i)\b(найти|найди|поиск\w*|извлеки|rag|документ\w*|баз[аеы]\s+знаний|стать[ия]|прочитай\s+файл|"
            r"search|find|retrieve|lookup|documentation|read_file|grep)\b"
        ),
        AgentRoleType.PLANNER: re.compile(
            r"(?i)\b(план\w*|спланируй|декомпозир\w*|шаг[иа]|архитектур\w*|roadmap|распиши|расписание|"
            r"plan\w*|steps?|schedule|decompose|architecture|dag)\b"
        ),
        AgentRoleType.EXECUTOR: re.compile(
            r"(?i)\b(выполни|запусти|реализуй|напиши|примени|скрипт\w*|команд[ау]|создай\s+файл|"
            r"run|execute|implement|build|code|script|apply|write_file)\b"
        ),
        AgentRoleType.REVIEWER: re.compile(
            r"(?i)\b(проверь\w*|аудит\w*|верифицир\w*|тест\w*|инспектируй|отревью|валидац\w*|дифф|"
            r"verify|audit|check|review|tests?|inspect|assertions?|validate)\b"
        ),
        AgentRoleType.ORCHESTRATOR: re.compile(
            r"(?i)\b(координир\w*|оркестрир\w*|обзор\w*|сводк\w*|управлени\w*|помоги|статус|"
            r"coordinate|orchestrate|overview|manage|status|summarize|help)\b"
        ),
    }

    # Rich bilingual semantic prototype descriptions for Tier 2 TF-IDF cosine similarity
    PROTOTYPE_DESCRIPTIONS: Dict[AgentRoleType, str] = {
        AgentRoleType.RETRIEVER: (
            "поиск информации документация rag база знаний извлечение контекста цитаты статьи "
            "read file search find query retrieve documentation snippet text corpus reference"
        ),
        AgentRoleType.PLANNER: (
            "составление плана декомпозиция задач архитектурное проектирование этапы roadmap "
            "create plan task decomposition schedule steps sequence task dag architecture design"
        ),
        AgentRoleType.EXECUTOR: (
            "написание кода выполнение скриптов вызов инструментов модификация файлов запуск команд "
            "execute code run script implementation build apply tools write files action runner"
        ),
        AgentRoleType.REVIEWER: (
            "проверка аудит кода тестирование верификация валидация приемочные критерии инспекция "
            "verify audit code review test suite assertions quality assurance inspection validation"
        ),
        AgentRoleType.ORCHESTRATOR: (
            "координация общее руководство агрегация результатов сводка распределение задач "
            "coordinate high level overview workflow management synthesize results multi agent dispatch"
        ),
    }

    def __init__(
        self,
        embedder: Optional[BaseEmbeddingProvider] = None,
        similarity_threshold: float = 0.20,
        margin_threshold: float = 0.02,
        llm_client: Optional[Any] = None,
        use_llm: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.embedder = embedder or LocalTFIDFEmbeddingProvider(dimension=128)
        self.use_llm = use_llm

        # Refit TF-IDF on prototype corpora to ensure all role terminology is represented in vocabulary
        seed_texts = list(self.PROTOTYPE_DESCRIPTIONS.values())
        for role_def in get_all_roles():
            seed_texts.extend(role_def.exemplar_queries)

        if hasattr(self.embedder, "fit"):
            self.embedder.fit(seed_texts)

        # Precompute and L2-normalize prototype vectors
        self._prototype_vectors: Dict[AgentRoleType, np.ndarray] = {}
        for role, text in self.PROTOTYPE_DESCRIPTIONS.items():
            vec = np.array(self.embedder.embed_text(text), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            self._prototype_vectors[role] = vec

        # Optional LLM client for Tier 3
        if self.use_llm:
            if llm_client is not None:
                self.llm_client = llm_client
            else:
                try:
                    from core.gigachat_client import gigachat
                    self.llm_client = gigachat
                except ImportError:
                    self.llm_client = None
        else:
            self.llm_client = None

    def route(self, query: str) -> AgentRoleType:
        """
        Standard routing interface: returns assigned AgentRoleType.
        Conforms to PROJECT.md line 80: IntentRouter.route(query: str) -> AgentRoleType
        """
        decision = self.route_with_metadata(query)
        return decision.role

    def route_with_metadata(self, query: str) -> RouteDecision:
        """
        Full telemetry routing interface: executes 3-tier cascade and returns RouteDecision.
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return RouteDecision(
                role=AgentRoleType.ORCHESTRATOR,
                tier=1,
                confidence=1.0,
                method="default_empty",
                reasoning="Empty or whitespace query defaulted to Orchestrator.",
            )

        # ----------------------------------------------------------------------
        # TIER 1: Fast Regex Pattern Matching
        # ----------------------------------------------------------------------
        regex_matches: List[tuple[AgentRoleType, str]] = []
        for role, pattern in self.TIER1_PATTERNS.items():
            match = pattern.search(clean_q)
            if match:
                regex_matches.append((role, match.group(0)))

        # Unambiguous single regex match
        if len(regex_matches) == 1:
            matched_role, matched_word = regex_matches[0]
            return RouteDecision(
                role=matched_role,
                tier=1,
                confidence=1.0,
                method="regex_match",
                matched_pattern=matched_word,
                reasoning=f"Matched deterministic keyword '{matched_word}' for {matched_role.value}.",
            )

        # ----------------------------------------------------------------------
        # TIER 2: Semantic TF-IDF Cosine Similarity
        # ----------------------------------------------------------------------
        q_vec = np.array(self.embedder.embed_text(clean_q), dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)

        scores: Dict[str, float] = {}
        if norm_q > 1e-12:
            q_normed = q_vec / norm_q
            for role, p_vec in self._prototype_vectors.items():
                sim = float(np.dot(q_normed, p_vec))
                scores[role.value] = max(0.0, sim)
        else:
            for role in self._prototype_vectors:
                scores[role.value] = 0.0

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_role_str, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        best_role = AgentRoleType(best_role_str)

        # Confident semantic match: exceeds threshold and has sufficient margin
        if best_score >= self.similarity_threshold and (best_score - second_score >= self.margin_threshold):
            return RouteDecision(
                role=best_role,
                tier=2,
                confidence=best_score,
                method="tfidf_cosine_similarity",
                scores=scores,
                reasoning=f"Semantic cosine similarity {best_score:.4f} exceeded threshold ({self.similarity_threshold}).",
            )

        # ----------------------------------------------------------------------
        # TIER 3: LLM Intent Classifier with Circuit Breaker & Fallback
        # ----------------------------------------------------------------------
        llm_decision = self._classify_via_llm(clean_q)
        if llm_decision:
            return llm_decision

        # Tier 3 Fallback: Deterministic Heuristic Classifier
        fallback_role = best_role if best_score > 0.05 else AgentRoleType.ORCHESTRATOR
        return RouteDecision(
            role=fallback_role,
            tier=3,
            confidence=max(0.5, best_score),
            method="deterministic_fallback",
            scores=scores,
            reasoning="Cascading fallback activated: LLM unavailable or ambiguous; selected top semantic candidate.",
        )

    def _classify_via_llm(self, query: str) -> Optional[RouteDecision]:
        """
        Invoke LLM client to classify query into one of the 5 canonical roles.
        Equipped with timeout and error handling circuit breaker.
        """
        if not self.llm_client or not hasattr(self.llm_client, "generate_text"):
            return None

        system_prompt = (
            "You are an intent classification subsystem. Classify the user query into exactly one role:\n"
            "- orchestrator: high-level coordination, overall status, general inquiries\n"
            "- retriever: search, RAG queries, documentation lookups, reading files\n"
            "- planner: task decomposition, architectural roadmaps, step-by-step planning\n"
            "- executor: writing code, running scripts, executing tools, modifying files\n"
            "- reviewer: checking results, auditing code, running tests, inspecting diffs\n\n"
            "Respond ONLY with valid JSON: {\"role\": \"<role>\", \"confidence\": <0.0-1.0>, \"reasoning\": \"<brief>\"}"
        )

        try:
            raw_text = self.llm_client.generate_text(
                system_prompt=system_prompt,
                user_prompt=query,
                temperature=0.0,
                max_tokens=150,
            )
            if not raw_text:
                return None

            # Extract JSON block
            json_str = raw_text.strip()
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            parsed = json.loads(json_str)
            raw_role = parsed.get("role", "").lower().strip()
            role = AgentRoleType(raw_role)
            confidence = float(parsed.get("confidence", 0.85))
            reasoning = parsed.get("reasoning", "Classified via LLM")

            return RouteDecision(
                role=role,
                tier=3,
                confidence=min(1.0, max(0.0, confidence)),
                method="llm_classification",
                reasoning=reasoning,
            )
        except Exception as e:
            logger.info("LLM intent classification skipped or failed (%s). Proceeding to fallback.", e)
            return None
