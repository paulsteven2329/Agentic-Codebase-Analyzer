from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class AnalyzerSettings:
    allowed_extensions: tuple[str, ...] = (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".md",
        ".sql",
        ".sh",
        ".dockerfile",
    )
    ignored_directories: tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".next",
        ".nuxt",
        ".turbo",
        "coverage",
        ".cache",
    )
    ignored_file_names: tuple[str, ...] = (
        ".DS_Store",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "poetry.lock",
    )
    max_file_bytes: int = 1_000_000
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retriever_k: int = 6
    per_file_retrieval_limit: int = 4
    file_context_char_limit: int = 4000
    final_summary_char_limit: int = 24000
    max_summary_files: int = 12
    use_llm_summary: bool = False
    llm_model: str = "mistral"
    embedding_model: str = "nomic-embed-text"
    vector_store_backend: str = "faiss"
    ollama_base_url: str = "http://localhost:11434"
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    chroma_collection_name: str = "codebase_chunks"
    ui_color: str = "cyan"

    @classmethod
    def from_overrides(
        cls,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        vector_store_backend: str | None = None,
        max_file_bytes: int | None = None,
        use_llm_summary: bool | None = None,
    ) -> "AnalyzerSettings":
        settings = cls()
        if llm_model:
            settings.llm_model = llm_model
        if embedding_model:
            settings.embedding_model = embedding_model
        if vector_store_backend:
            settings.vector_store_backend = vector_store_backend
        if max_file_bytes:
            settings.max_file_bytes = max_file_bytes
        if use_llm_summary is not None:
            settings.use_llm_summary = use_llm_summary
        return settings

    def is_allowed_extension(self, path: Path) -> bool:
        suffixes = path.suffixes
        if path.name.lower() == "dockerfile":
            return True
        return any(suffix.lower() in self.allowed_extensions for suffix in suffixes[-2:] or suffixes)

    def should_ignore_dir(self, directory_name: str) -> bool:
        return directory_name in set(self.ignored_directories)

    def should_ignore_file(self, file_name: str) -> bool:
        return file_name in set(self.ignored_file_names)

    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def normalize_ignore_dirs(self, extra_dirs: Iterable[str] | None = None) -> set[str]:
        ignored = set(self.ignored_directories)
        if extra_dirs:
            ignored.update(extra_dirs)
        return ignored
