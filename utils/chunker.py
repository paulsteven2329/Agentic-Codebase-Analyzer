from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import AnalyzerSettings


SECTION_PATTERNS = [
    r"(?m)^(class\s+\w+.*:)",
    r"(?m)^(def\s+\w+\s*\(.*\):)",
    r"(?m)^(async\s+def\s+\w+\s*\(.*\):)",
    r"(?m)^(export\s+(?:async\s+)?function\s+\w+\s*\(.*\)\s*\{?)",
    r"(?m)^(function\s+\w+\s*\(.*\)\s*\{?)",
    r"(?m)^(const\s+\w+\s*=\s*(?:async\s*)?\(?.*\)?\s*=>\s*\{?)",
    r"(?m)^(interface\s+\w+.*\{)",
    r"(?m)^(type\s+\w+\s*=)",
    r"(?m)^(struct\s+\w+.*\{)",
    r"(?m)^(impl\s+\w+.*\{)",
    r"(?m)^(pub\s+fn\s+\w+\s*\(.*\))",
    r"(?m)^(fn\s+\w+\s*\(.*\))",
]


def _split_code_sections(text: str) -> list[str]:
    matches: list[int] = []
    for pattern in SECTION_PATTERNS:
        matches.extend(match.start() for match in re.finditer(pattern, text))

    boundaries = sorted(set(index for index in matches if index > 0))
    if not boundaries:
        return [text]

    sections: list[str] = []
    start = 0
    for boundary in boundaries:
        chunk = text[start:boundary].strip()
        if chunk:
            sections.append(chunk)
        start = boundary

    final_chunk = text[start:].strip()
    if final_chunk:
        sections.append(final_chunk)
    return sections or [text]


def _build_documents(
    file_path: str,
    relative_path: str,
    content: str,
    settings: AnalyzerSettings,
) -> Iterable[Document]:
    code_sections = _split_code_sections(content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    for section_index, section in enumerate(code_sections):
        chunks = splitter.split_text(section)
        for chunk_index, chunk in enumerate(chunks):
            yield Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "relative_path": relative_path,
                    "file_name": Path(relative_path).name,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                },
            )


def chunk_file_content(
    file_path: str,
    relative_path: str,
    content: str,
    settings: AnalyzerSettings,
) -> list[Document]:
    return list(_build_documents(file_path, relative_path, content, settings))
