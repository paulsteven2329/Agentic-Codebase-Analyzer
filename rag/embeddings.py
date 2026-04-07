from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from config.settings import AnalyzerSettings


def build_embeddings(settings: AnalyzerSettings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
