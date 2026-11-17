"""
Multi-format document chunkers for RAG ingestion.
Provides header-aware MarkdownChunker, AST/syntax CodeChunker, structured JSONChunker,
and RecursiveTextChunker with exact character offset and section tracking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import re
import ast
import json
import uuid

from core.rag.models import DocumentMetadata, DocumentChunk


def estimate_token_count(text: str) -> int:
    """Approximate token count (rule of thumb: ~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Split input text into a list of DocumentChunks."""
        pass


class RecursiveTextChunker(BaseChunker):
    """
    Splits text recursively using hierarchical separators:
    double newline, newline, sentence boundaries, space, character.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        doc_id = metadata.source
        spans = self._split_spans(text, self.SEPARATORS)

        chunks: List[DocumentChunk] = []
        for i, (start_char, end_char) in enumerate(spans):
            piece_str = text[start_char:end_char].strip()
            if not piece_str:
                continue

            actual_start = text.find(piece_str, start_char)
            if actual_start == -1:
                actual_start = start_char
            actual_end = actual_start + len(piece_str)

            chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=piece_str,
                    chunk_index=i,
                    start_char=actual_start,
                    end_char=actual_end,
                    section_title=metadata.title,
                    token_count=estimate_token_count(piece_str),
                    metadata=dict(metadata.extra),
                )
            )

        # Re-index chunks sequentially
        for idx, ch in enumerate(chunks):
            ch.chunk_index = idx

        return chunks

    def _split_spans(self, text: str, separators: List[str]) -> List[Tuple[int, int]]:
        """Split text into (start, end) spans directly indexed in source string."""
        if len(text) <= self.chunk_size:
            return [(0, len(text))]

        # Choose the first separator present in text
        sep = ""
        remaining_seps: List[str] = []
        for i, s in enumerate(separators):
            if s == "":
                sep = ""
                remaining_seps = []
                break
            if s in text:
                sep = s
                remaining_seps = separators[i + 1:]
                break

        if sep == "":
            # Hard character split if no separator found
            step = max(1, self.chunk_size - self.chunk_overlap)
            spans = []
            for i in range(0, len(text), step):
                spans.append((i, min(len(text), i + self.chunk_size)))
                if i + self.chunk_size >= len(text):
                    break
            return spans

        # Split text into parts and find exact start/end offsets
        parts = text.split(sep)
        pieces: List[Tuple[int, int]] = []
        search_pos = 0
        for p in parts:
            if not p:
                search_pos += len(sep)
                continue
            pos = text.find(p, search_pos)
            if pos == -1:
                pos = search_pos
            p_end = pos + len(p)
            pieces.append((pos, p_end))
            search_pos = p_end + len(sep)

        if not pieces:
            return [(0, len(text))]

        # If any single piece exceeds chunk_size, recursively subdivide it
        refined_pieces: List[Tuple[int, int]] = []
        for p_start, p_end in pieces:
            piece_text = text[p_start:p_end]
            if len(piece_text) > self.chunk_size and remaining_seps:
                sub_spans = self._split_spans(piece_text, remaining_seps)
                for sub_s, sub_e in sub_spans:
                    refined_pieces.append((p_start + sub_s, p_start + sub_e))
            else:
                refined_pieces.append((p_start, p_end))

        # Greedily group refined pieces into chunks respecting chunk_size and chunk_overlap
        chunks: List[Tuple[int, int]] = []
        curr: List[Tuple[int, int]] = []

        for p_start, p_end in refined_pieces:
            if not curr:
                curr.append((p_start, p_end))
            else:
                candidate_len = p_end - curr[0][0]
                if candidate_len <= self.chunk_size:
                    curr.append((p_start, p_end))
                else:
                    chunks.append((curr[0][0], curr[-1][1]))
                    while curr and ((curr[-1][1] - curr[0][0]) > self.chunk_overlap or (p_end - curr[0][0]) > self.chunk_size):
                        curr.pop(0)
                    curr.append((p_start, p_end))

        if curr:
            chunks.append((curr[0][0], curr[-1][1]))

        return chunks


class MarkdownChunker(BaseChunker):
    """
    Header-aware Markdown chunker that tracks section breadcrumbs
    (# Title > ## Section > ### Sub-section) and preserves structural hierarchy.
    """

    HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        doc_id = metadata.source
        sub_chunker = RecursiveTextChunker(self.chunk_size, self.chunk_overlap)

        # Identify spans inside code block fences (```)
        code_fence_ranges: List[Tuple[int, int]] = []
        in_code_fence = False
        fence_start = -1
        pos = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith("```"):
                if not in_code_fence:
                    in_code_fence = True
                    fence_start = pos
                else:
                    in_code_fence = False
                    code_fence_ranges.append((fence_start, pos + len(line)))
            pos += len(line)
        if in_code_fence:
            code_fence_ranges.append((fence_start, len(text)))

        def is_inside_code_fence(char_pos: int) -> bool:
            for c_start, c_end in code_fence_ranges:
                if c_start <= char_pos < c_end:
                    return True
            return False

        # Find all headers outside code fences
        header_matches = [
            m for m in self.HEADER_REGEX.finditer(text)
            if not is_inside_code_fence(m.start())
        ]
        if not header_matches:
            # Fall back to recursive text chunking
            return sub_chunker.chunk(text, metadata)

        sections: List[Dict[str, Any]] = []
        
        # Leading text before first header
        if header_matches[0].start() > 0:
            preamble = text[:header_matches[0].start()].strip()
            if preamble:
                sections.append({
                    "breadcrumb": metadata.title or "Preamble",
                    "content": preamble,
                    "start": 0,
                    "end": header_matches[0].start()
                })

        breadcrumb_stack: List[Tuple[int, str]] = []

        for i, match in enumerate(header_matches):
            hashes = match.group(1)
            title = match.group(2).strip()
            level = len(hashes)
            header_start = match.start()

            # Determine end of this section
            if i + 1 < len(header_matches):
                section_end = header_matches[i + 1].start()
            else:
                section_end = len(text)

            section_body = text[header_start:section_end].strip()

            # Update breadcrumb stack
            while breadcrumb_stack and breadcrumb_stack[-1][0] >= level:
                breadcrumb_stack.pop()
            breadcrumb_stack.append((level, title))

            full_breadcrumb = " > ".join(t for _, t in breadcrumb_stack)
            if metadata.title and not full_breadcrumb.startswith(metadata.title):
                full_breadcrumb = f"{metadata.title} > {full_breadcrumb}"

            sections.append({
                "breadcrumb": full_breadcrumb,
                "content": section_body,
                "start": header_start,
                "end": section_end
            })

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_text = sec["content"]
            sec_breadcrumb = sec["breadcrumb"]
            sec_start = sec["start"]

            if len(sec_text) <= self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=sec_text,
                        chunk_index=chunk_idx,
                        start_char=sec_start,
                        end_char=sec_start + len(sec_text),
                        section_title=sec_breadcrumb,
                        token_count=estimate_token_count(sec_text),
                        metadata=dict(metadata.extra),
                    )
                )
                chunk_idx += 1
            else:
                # Sub-divide large section
                sec_meta = metadata.model_copy(update={"title": sec_breadcrumb})
                sub_chunks = sub_chunker.chunk(sec_text, sec_meta)
                for sc in sub_chunks:
                    sc.section_title = sec_breadcrumb
                    sc.chunk_index = chunk_idx
                    sc.start_char += sec_start
                    sc.end_char += sec_start
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks


class CodeChunker(BaseChunker):
    """
    Code chunker parsing Python AST for class and function blocks,
    falling back to regex block identification for non-Python or syntax-invalid files.
    """

    FALLBACK_CODE_PATTERN = re.compile(
        r"(?m)^(?:(?:class|def|async\s+def)\s+([a-zA-Z_][a-zA-Z0-9_]*)|function\s+([a-zA-Z_][a-zA-Z0-9_]*))"
    )

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        sub_chunker = RecursiveTextChunker(self.chunk_size, self.chunk_overlap)

        # Attempt Python AST parsing if python file or default
        is_python = metadata.file_type.lower() in ("py", "python") or metadata.source.endswith(".py")
        if is_python:
            try:
                tree = ast.parse(text)
                return self._chunk_python_ast(tree, text, metadata, sub_chunker)
            except SyntaxError:
                # Fall back to regex/block chunking
                pass

        return self._chunk_generic_code(text, metadata, sub_chunker)

    def _chunk_python_ast(
        self,
        tree: ast.AST,
        text: str,
        metadata: DocumentMetadata,
        sub_chunker: RecursiveTextChunker
    ) -> List[DocumentChunk]:
        lines = text.splitlines(keepends=True)
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        # Collect top-level definitions
        blocks: List[Tuple[str, int, int]] = [] # (symbol_name, start_line, end_line)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                prefix = "class" if isinstance(node, ast.ClassDef) else "def"
                symbol = f"{prefix} {node.name}"
                blocks.append((symbol, node.lineno - 1, end_lineno))

        if not blocks:
            # No functions or classes found, use recursive text chunker
            return sub_chunker.chunk(text, metadata)

        # Handle header code (imports, global constants) before first block
        first_start_line = blocks[0][1]
        if first_start_line > 0:
            preamble = "".join(lines[:first_start_line]).strip()
            if preamble:
                if len(preamble) > self.chunk_size:
                    preamble_meta = metadata.model_copy(update={"title": f"{metadata.source}#module_header"})
                    sub_chunks = sub_chunker.chunk(preamble, preamble_meta)
                    for sc in sub_chunks:
                        sc.section_title = f"{metadata.source}#module_header"
                        sc.chunk_index = chunk_idx
                        chunks.append(sc)
                        chunk_idx += 1
                else:
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            doc_id=metadata.source,
                            content=preamble,
                            chunk_index=chunk_idx,
                            start_char=0,
                            end_char=len(preamble),
                            section_title=f"{metadata.source}#module_header",
                            token_count=estimate_token_count(preamble),
                            metadata=dict(metadata.extra),
                        )
                    )
                    chunk_idx += 1

        for symbol, start_l, end_l in blocks:
            block_code = "".join(lines[start_l:end_l]).strip()
            if not block_code:
                continue

            start_char = sum(len(line) for line in lines[:start_l])
            end_char = start_char + len(block_code)

            if len(block_code) <= self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        doc_id=metadata.source,
                        content=block_code,
                        chunk_index=chunk_idx,
                        start_char=start_char,
                        end_char=end_char,
                        section_title=symbol,
                        token_count=estimate_token_count(block_code),
                        metadata=dict(metadata.extra),
                    )
                )
                chunk_idx += 1
            else:
                sec_meta = metadata.model_copy(update={"title": symbol})
                sub_chunks = sub_chunker.chunk(block_code, sec_meta)
                for sc in sub_chunks:
                    sc.section_title = symbol
                    sc.chunk_index = chunk_idx
                    sc.start_char += start_char
                    sc.end_char += start_char
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks

    def _chunk_generic_code(
        self,
        text: str,
        metadata: DocumentMetadata,
        sub_chunker: RecursiveTextChunker
    ) -> List[DocumentChunk]:
        """Regex-based code block splitting for non-Python or unparseable code."""
        matches = list(self.FALLBACK_CODE_PATTERN.finditer(text))
        if not matches:
            return sub_chunker.chunk(text, metadata)

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for i, match in enumerate(matches):
            symbol = match.group(0).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block_text = text[start:end].strip()

            if len(block_text) <= self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        doc_id=metadata.source,
                        content=block_text,
                        chunk_index=chunk_idx,
                        start_char=start,
                        end_char=end,
                        section_title=symbol,
                        token_count=estimate_token_count(block_text),
                        metadata=dict(metadata.extra),
                    )
                )
                chunk_idx += 1
            else:
                sec_meta = metadata.model_copy(update={"title": symbol})
                sub_chunks = sub_chunker.chunk(block_text, sec_meta)
                for sc in sub_chunks:
                    sc.section_title = symbol
                    sc.chunk_index = chunk_idx
                    sc.start_char += start
                    sc.end_char += start
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks


class JSONChunker(BaseChunker):
    """
    Structured JSON chunker converting dictionaries or array elements into
    clean key-path blocks while preserving semantic relationships.
    """

    def chunk(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        sub_chunker = RecursiveTextChunker(self.chunk_size, self.chunk_overlap)
        try:
            data = json.loads(text)
        except Exception:
            # Fall back to recursive text chunking if JSON is invalid
            return sub_chunker.chunk(text, metadata)

        chunks: List[DocumentChunk] = []
        doc_id = metadata.source
        chunk_idx = 0

        if isinstance(data, list):
            # Chunk by array items or batches of items
            current_batch: List[str] = []
            current_len = 0

            for idx, item in enumerate(data):
                item_str = json.dumps(item, ensure_ascii=False, indent=2)
                if len(item_str) > self.chunk_size:
                    if current_batch:
                        batch_content = "\n".join(current_batch)
                        chunks.append(
                            DocumentChunk(
                                id=str(uuid.uuid4()),
                                doc_id=doc_id,
                                content=batch_content,
                                chunk_index=chunk_idx,
                                start_char=0,
                                end_char=len(batch_content),
                                section_title=f"{metadata.source}#items",
                                token_count=estimate_token_count(batch_content),
                                metadata=dict(metadata.extra),
                            )
                        )
                        chunk_idx += 1
                        current_batch = []
                        current_len = 0

                    item_meta = metadata.model_copy(update={"title": f"{metadata.source}#items"})
                    sub_chunks = sub_chunker.chunk(item_str, item_meta)
                    for sc in sub_chunks:
                        sc.section_title = f"{metadata.source}#items"
                        sc.chunk_index = chunk_idx
                        chunks.append(sc)
                        chunk_idx += 1
                elif current_len + len(item_str) > self.chunk_size and current_batch:
                    batch_content = "\n".join(current_batch)
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            content=batch_content,
                            chunk_index=chunk_idx,
                            start_char=0,
                            end_char=len(batch_content),
                            section_title=f"{metadata.source}#items",
                            token_count=estimate_token_count(batch_content),
                            metadata=dict(metadata.extra),
                        )
                    )
                    chunk_idx += 1
                    current_batch = [item_str]
                    current_len = len(item_str)
                else:
                    current_batch.append(item_str)
                    current_len += len(item_str)

            if current_batch:
                batch_content = "\n".join(current_batch)
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=batch_content,
                        chunk_index=chunk_idx,
                        start_char=0,
                        end_char=len(batch_content),
                        section_title=f"{metadata.source}#items",
                        token_count=estimate_token_count(batch_content),
                        metadata=dict(metadata.extra),
                    )
                )
        elif isinstance(data, dict):
            # Flatten top-level keys into structured blocks
            for key, val in data.items():
                val_str = json.dumps(val, ensure_ascii=False, indent=2)
                block_content = f'"{key}": {val_str}'
                section_title = f"{metadata.source}#{key}"

                if len(block_content) <= self.chunk_size:
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            doc_id=doc_id,
                            content=block_content,
                            chunk_index=chunk_idx,
                            start_char=0,
                            end_char=len(block_content),
                            section_title=section_title,
                            token_count=estimate_token_count(block_content),
                            metadata=dict(metadata.extra),
                        )
                    )
                    chunk_idx += 1
                else:
                    sec_meta = metadata.model_copy(update={"title": section_title})
                    sub_chunks = sub_chunker.chunk(block_content, sec_meta)
                    for sc in sub_chunks:
                        sc.section_title = section_title
                        sc.chunk_index = chunk_idx
                        chunks.append(sc)
                        chunk_idx += 1
        else:
            # Primitive JSON value
            return sub_chunker.chunk(text, metadata)

        return chunks


def get_chunker_for_file(
    file_path_or_ext: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80
) -> BaseChunker:
    """
    Factory function returning the optimal chunker based on file extension.
    """
    ext = file_path_or_ext.lower().split(".")[-1] if "." in file_path_or_ext else file_path_or_ext.lower()
    
    if ext in ("md", "markdown"):
        return MarkdownChunker(chunk_size, chunk_overlap)
    elif ext in ("py", "python", "js", "ts", "jsx", "tsx", "sh", "sql", "html", "css"):
        return CodeChunker(chunk_size, chunk_overlap)
    elif ext in ("json", "jsonl"):
        return JSONChunker(chunk_size, chunk_overlap)
    else:
        return RecursiveTextChunker(chunk_size, chunk_overlap)
