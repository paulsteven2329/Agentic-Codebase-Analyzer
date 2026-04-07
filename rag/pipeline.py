from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import AnalyzerSettings
from rag.embeddings import build_embeddings
from utils.chunker import chunk_file_content


LOGGER = logging.getLogger(__name__)


class CodebaseRAGPipeline:
    def __init__(self, settings: AnalyzerSettings) -> None:
        self.settings = settings
        self.embeddings = build_embeddings(settings)

    def load_and_chunk(self, file_records: list[dict[str, Any]]) -> tuple[list[Document], list[dict[str, Any]]]:
        documents: list[Document] = []
        file_payloads: list[dict[str, Any]] = []

        for record in file_records:
            file_path = Path(record["path"])
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                LOGGER.warning("Unable to read %s: %s", file_path, exc)
                continue

            file_payloads.append(
                {
                    "path": record["path"],
                    "relative_path": record["relative_path"],
                    "content": content,
                    "extension": record["extension"],
                    "size_bytes": record["size_bytes"],
                }
            )
            documents.extend(
                chunk_file_content(
                    file_path=record["path"],
                    relative_path=record["relative_path"],
                    content=content,
                    settings=self.settings,
                )
            )

        return documents, file_payloads

    def build_or_load_vector_store(
        self,
        folder_path: str,
        documents: list[Document],
        file_records: list[dict[str, Any]],
    ) -> Any:
        cache_key = self._build_cache_key(folder_path, file_records)
        cache_root = self.settings.ensure_cache_dir() / cache_key
        metadata_path = cache_root / "metadata.json"
        faiss_dir = cache_root / "faiss"
        chroma_dir = cache_root / "chroma"

        if cache_root.exists() and metadata_path.exists():
            if self.settings.vector_store_backend.lower() == "faiss" and faiss_dir.exists():
                LOGGER.info("Loading cached vector store from %s", cache_root)
                return self._load_vector_store(cache_root)
            if self.settings.vector_store_backend.lower() == "chroma" and chroma_dir.exists():
                LOGGER.info("Loading cached vector store from %s", cache_root)
                return self._load_vector_store(cache_root)

        if not documents:
            raise ValueError("Cannot build a vector store without documents.")

        cache_root.mkdir(parents=True, exist_ok=True)
        vector_store = self._create_vector_store(documents, cache_root)
        metadata_path.write_text(
            json.dumps(
                {
                    "folder_path": str(Path(folder_path).resolve()),
                    "file_count": len(file_records),
                    "vector_backend": self.settings.vector_store_backend,
                    "embedding_model": self.settings.embedding_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return vector_store

    def retrieve_context_per_file(self, vector_store: Any, file_payloads: list[dict[str, Any]]) -> dict[str, list[str]]:
        contexts: dict[str, list[str]] = defaultdict(list)
        for payload in file_payloads:
            contexts[payload["relative_path"]] = self._build_fast_file_context(payload)
        return dict(contexts)

    def _build_fast_file_context(self, payload: dict[str, Any]) -> list[str]:
        content = payload["content"]
        limit = self.settings.file_context_char_limit
        if len(content) <= limit:
            return [content]

        half = max(limit // 2, 1)
        return [content[:half], content[-half:]]

    def _build_cache_key(self, folder_path: str, file_records: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        digest.update(str(Path(folder_path).resolve()).encode("utf-8"))
        digest.update(self.settings.embedding_model.encode("utf-8"))
        digest.update(self.settings.vector_store_backend.encode("utf-8"))
        for record in file_records:
            digest.update(record["relative_path"].encode("utf-8"))
            digest.update(str(record["size_bytes"]).encode("utf-8"))
        return digest.hexdigest()[:16]

    def _create_vector_store(self, documents: list[Document], cache_root: Path) -> Any:
        backend = self.settings.vector_store_backend.lower()
        if backend == "chroma":
            store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(cache_root / "chroma"),
                collection_name=self.settings.chroma_collection_name,
            )
            return store

        store = FAISS.from_documents(documents, self.embeddings)
        store.save_local(str(cache_root / "faiss"))
        return store

    def _load_vector_store(self, cache_root: Path) -> Any:
        backend = self.settings.vector_store_backend.lower()
        if backend == "chroma":
            return Chroma(
                persist_directory=str(cache_root / "chroma"),
                embedding_function=self.embeddings,
                collection_name=self.settings.chroma_collection_name,
            )

        return FAISS.load_local(
            str(cache_root / "faiss"),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
