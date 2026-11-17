"""
Context injection and prompt augmentation engine for RAG subsystem.
Packs retrieved snippets within strict token budgets, performs greedy deduplication,
and formats verifiable source citations conforming to Karpathy's Zero-Hallucination principle.
"""

from typing import List, Tuple, Optional, Set

from core.rag.models import SearchResult, ContextSnippet


def approximate_tokens(text: str) -> int:
    """Fast approximation of token count (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class ContextInjector:
    """
    Packs retrieved knowledge into LLM-ready context blocks with verifiable citations.
    """

    def __init__(self, default_max_tokens: int = 2048):
        self.default_max_tokens = default_max_tokens

    def format_context(
        self,
        results: List[SearchResult],
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[ContextSnippet]]:
        """
        Greedily pack retrieved SearchResults into a structured context string
        respecting token budget and deduplicating overlapping sections.
        Returns:
            (formatted_context_xml_str, list_of_context_snippets)
        """
        if not results:
            return "", []

        budget = max_tokens if max_tokens is not None else self.default_max_tokens
        if budget <= 0:
            return "", []

        wrapper_tokens = 8
        if budget <= wrapper_tokens:
            return "", []

        packed_blocks: List[str] = []
        snippets: List[ContextSnippet] = []
        seen_contents: Set[str] = set()

        citation_index = 1

        for sres in results:
            chunk = sres.chunk
            content_clean = chunk.content.strip()
            if not content_clean:
                continue

            # Check for duplicate or near-duplicate content
            content_hash = f"{chunk.doc_id}:{chunk.chunk_index}"
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)

            source_name = chunk.doc_id
            section_name = chunk.section_title or "Overview"
            citation_id = f"Source {citation_index}"

            # Format citation header
            citation_header = f"[{citation_id}: {source_name}#{section_name} | Score: {sres.score:.2f}]"
            block = f"{citation_header}\n{content_clean}"

            candidate_blocks = packed_blocks + [block]
            candidate_str = "<context>\n" + "\n\n".join(candidate_blocks) + "\n</context>"

            if approximate_tokens(candidate_str) <= budget:
                packed_blocks.append(block)
                snippets.append(
                    ContextSnippet(
                        source=source_name,
                        section=section_name,
                        content=content_clean,
                        score=sres.score,
                        citation_id=citation_id
                    )
                )
                citation_index += 1
            else:
                # We cannot fit the whole block. Check if space allows a truncated version
                base_str = ("<context>\n" + "\n\n".join(packed_blocks) + "\n\n") if packed_blocks else "<context>\n"
                suffix_str = "\n</context>"
                trunc_tag = "... [truncated]"
                trunc_header = f"{citation_header}\n"
                fixed_len = len(base_str) + len(trunc_header) + len(trunc_tag) + 1 + len(suffix_str)
                max_total_len = budget * 4
                max_content_chars = max_total_len - fixed_len

                if max_content_chars >= 40:
                    truncated_content = content_clean[:max_content_chars].rstrip() + " " + trunc_tag
                    trunc_block = f"{citation_header}\n{truncated_content}"
                    cand_trunc_str = "<context>\n" + "\n\n".join(packed_blocks + [trunc_block]) + "\n</context>"
                    if approximate_tokens(cand_trunc_str) <= budget:
                        packed_blocks.append(trunc_block)
                        snippets.append(
                            ContextSnippet(
                                source=source_name,
                                section=section_name,
                                content=truncated_content,
                                score=sres.score,
                                citation_id=citation_id
                            )
                        )
                break

        if not packed_blocks:
            return "", []

        context_str = "<context>\n" + "\n\n".join(packed_blocks) + "\n</context>"
        return context_str, snippets

    def build_augmented_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        context_str: str
    ) -> str:
        """
        Combine system instructions, verifiable context, and user prompt
        with explicit instructions to cite sources and reject hallucinations.
        """
        guardrail_instruction = (
            "\n\n[RAG GROUNDING MANDATE]\n"
            "1. Answer the user prompt strictly based on the verifiable information provided in <context> above.\n"
            "2. Cite your sources in the response using the exact citation tags provided (e.g. [Source 1], [Source 2]).\n"
            "3. If the provided context does not contain sufficient facts to answer the question, state that clearly "
            "without speculating, inventing, or hallucinating facts.\n"
        )

        parts = []
        if system_prompt and system_prompt.strip():
            parts.append(system_prompt.strip())

        if context_str and context_str.strip():
            parts.append(context_str.strip())
            parts.append(guardrail_instruction.strip())

        parts.append(f"User Query:\n{user_prompt.strip()}")

        return "\n\n".join(parts)
