from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import AnalyzerSettings
from rag.pipeline import CodebaseRAGPipeline
from utils.file_scanner import discover_code_files


LOGGER = logging.getLogger(__name__)


class AnalyzerState(TypedDict, total=False):
    folder_path: str
    settings: AnalyzerSettings
    progress: list[str]
    project_name: str
    file_records: list[dict[str, Any]]
    skipped_large_files: list[str]
    chunk_documents: list[Any]
    file_payloads: list[dict[str, Any]]
    rag_pipeline: CodebaseRAGPipeline
    vector_store: Any
    retrieved_contexts: dict[str, list[str]]
    file_summaries: list[dict[str, Any]]
    project_summary: dict[str, Any]
    final_summary: str


def input_node(state: AnalyzerState) -> AnalyzerState:
    folder_path = str(Path(state["folder_path"]).expanduser().resolve())
    project_name = Path(folder_path).name
    progress = list(state.get("progress", []))
    progress.append(f"Input received for {folder_path}")
    LOGGER.info("Starting analysis for %s", folder_path)
    return {
        **state,
        "folder_path": folder_path,
        "project_name": project_name,
        "progress": progress,
    }


def file_discovery_node(state: AnalyzerState) -> AnalyzerState:
    settings = state["settings"]
    result = discover_code_files(state["folder_path"], settings)
    progress = list(state.get("progress", []))
    progress.append(f"Discovered {len(result['files'])} candidate code files")
    LOGGER.info("Discovered %s files", len(result["files"]))
    return {
        **state,
        "file_records": result["files"],
        "skipped_large_files": result["skipped_large_files"],
        "progress": progress,
    }


def file_processing_node(state: AnalyzerState) -> AnalyzerState:
    file_records = state.get("file_records", [])
    pipeline = CodebaseRAGPipeline(state["settings"])
    documents, file_payloads = pipeline.load_and_chunk(file_records)
    progress = list(state.get("progress", []))
    progress.append(f"Loaded {len(file_payloads)} files into {len(documents)} chunks")
    LOGGER.info("Prepared %s chunks from %s files", len(documents), len(file_payloads))
    return {
        **state,
        "chunk_documents": documents,
        "file_payloads": file_payloads,
        "rag_pipeline": pipeline,
        "progress": progress,
    }


def embedding_node(state: AnalyzerState) -> AnalyzerState:
    pipeline: CodebaseRAGPipeline = state["rag_pipeline"]
    progress = list(state.get("progress", []))
    if not state.get("chunk_documents"):
        progress.append("No documents available for embedding; skipping vector store creation")
        LOGGER.info("Skipping vector store creation because no documents were produced")
        return {
            **state,
            "vector_store": None,
            "progress": progress,
        }

    vector_store = pipeline.build_or_load_vector_store(
        folder_path=state["folder_path"],
        documents=state.get("chunk_documents", []),
        file_records=state.get("file_records", []),
    )
    progress.append("Embeddings ready and vector store available")
    LOGGER.info("Vector store is ready")
    return {
        **state,
        "vector_store": vector_store,
        "progress": progress,
    }


def retrieval_node(state: AnalyzerState) -> AnalyzerState:
    pipeline: CodebaseRAGPipeline = state["rag_pipeline"]
    progress = list(state.get("progress", []))
    if state.get("vector_store") is None:
        progress.append("No vector store available; retrieval skipped")
        LOGGER.info("Skipping retrieval because vector store is unavailable")
        return {
            **state,
            "retrieved_contexts": {},
            "progress": progress,
        }

    retrieved_contexts = pipeline.retrieve_context_per_file(
        state["vector_store"], state.get("file_payloads", [])
    )
    progress.append(f"Retrieved focused context for {len(retrieved_contexts)} files")
    LOGGER.info("Retrieved file contexts")
    return {
        **state,
        "retrieved_contexts": retrieved_contexts,
        "progress": progress,
    }


def analysis_node(state: AnalyzerState) -> AnalyzerState:
    if not state.get("file_payloads"):
        progress = list(state.get("progress", []))
        progress.append("No files available for file-level analysis")
        LOGGER.info("Skipping file analysis because no files were loaded")
        return {
            **state,
            "file_summaries": [],
            "progress": progress,
        }

    file_summaries: list[dict[str, Any]] = []

    for payload in state.get("file_payloads", []):
        relative_path = payload["relative_path"]
        retrieved_chunks = state.get("retrieved_contexts", {}).get(relative_path, [])
        file_summaries.append(_build_fast_file_summary(payload, retrieved_chunks))

    progress = list(state.get("progress", []))
    progress.append(f"Generated file-level analysis for {len(file_summaries)} files")
    LOGGER.info("Completed file analysis")
    return {
        **state,
        "file_summaries": file_summaries,
        "progress": progress,
    }


def synthesis_node(state: AnalyzerState) -> AnalyzerState:
    if not state.get("file_summaries"):
        skipped = state.get("skipped_large_files", [])
        summary = "\n".join(
            [
                "[PROJECT OVERVIEW]",
                f"- Name: {state.get('project_name', 'unknown')}",
                "- Type: Unknown",
                "- Purpose: No analyzable code files were found in the target folder.",
                "",
                "[ARCHITECTURE]",
                "- No architecture could be inferred because no supported source files were processed.",
                "",
                "[FOLDER STRUCTURE]",
                f"- Root: {state.get('folder_path', 'unknown')}",
                "",
                "[FILE SUMMARIES]",
                "- No important files were available for summarization.",
                "",
                "[DEPENDENCIES]",
                "- Internal dependencies: none detected",
                "- External dependencies: none detected",
                "",
                "[EXECUTION FLOW]",
                "- Input path validated",
                "- Directory scanned for supported source files",
                "- No files passed the analyzer filters",
                "",
                "[KEY INSIGHTS]",
                "- Add supported code or configuration files to generate a richer summary.",
                f"- Skipped large files: {', '.join(skipped) if skipped else 'none'}",
            ]
        )
        progress = list(state.get("progress", []))
        progress.append("Synthesized fallback summary for an empty or unsupported project")
        LOGGER.info("Generated fallback synthesis summary")
        return {
            **state,
            "final_summary": summary,
            "progress": progress,
        }

    settings = state["settings"]
    progress = list(state.get("progress", []))
    if not settings.use_llm_summary:
        final_summary = _build_local_project_summary(state)
        progress.append("Synthesized fast local project summary")
        LOGGER.info("Completed local project synthesis")
        return {
            **state,
            "final_summary": final_summary,
            "progress": progress,
        }

    try:
        synthesis_prompt = _build_project_synthesis_prompt(state)
        llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a senior software architect. Produce a terminal-friendly project summary "
                        "using the exact requested section headers."
                    )
                ),
                HumanMessage(content=synthesis_prompt),
            ]
        )
        final_summary = response.content
        progress.append("Synthesized project-level summary with local LLM")
        LOGGER.info("Completed project synthesis with local LLM")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        LOGGER.warning("LLM synthesis failed, falling back to local summary: %s", exc)
        final_summary = _build_local_project_summary(state)
        progress.append("Synthesized fallback project summary locally")

    return {
        **state,
        "final_summary": final_summary,
        "progress": progress,
    }


def output_node(state: AnalyzerState) -> AnalyzerState:
    progress = list(state.get("progress", []))
    progress.append("Output ready")
    return {
        **state,
        "progress": progress,
    }


def _build_fast_file_summary(payload: dict[str, Any], retrieved_chunks: list[str]) -> dict[str, Any]:
    content = payload["content"]
    relative_path = payload["relative_path"]
    dependencies = _extract_dependencies(content)
    functions, classes = _extract_symbols(content)
    patterns = _detect_patterns(content, relative_path)

    return {
        "file_name": Path(relative_path).name,
        "relative_path": relative_path,
        "purpose": _infer_file_purpose(relative_path, content, functions, classes),
        "key_logic": _build_key_logic(functions, classes, dependencies, retrieved_chunks),
        "dependencies": dependencies[:10],
        "patterns": patterns,
        "role_in_system": _infer_role(relative_path, content),
    }


def _extract_dependencies(content: str) -> list[str]:
    patterns = [
        r"(?m)^\s*from\s+([a-zA-Z0-9_\.]+)\s+import",
        r'(?m)^\s*const\s+.+?=\s+require\(["\'](.+?)["\']\)',
        r'(?m)^\s*import\s+.+?\s+from\s+["\'](.+?)["\']',
        r'(?m)^\s*import\s+["\'](.+?)["\']',
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(1) for match in re.finditer(pattern, content))
    return _dedupe(found)


def _extract_symbols(content: str) -> tuple[list[str], list[str]]:
    function_patterns = [
        r"(?m)^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        r"(?m)^\s*async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        r"(?m)^\s*function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        r"(?m)^\s*export\s+(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        r"(?m)^\s*(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(?.*\)?\s*=>",
    ]
    class_patterns = [
        r"(?m)^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"(?m)^\s*interface\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ]
    functions: list[str] = []
    classes: list[str] = []
    for pattern in function_patterns:
        functions.extend(match.group(1) for match in re.finditer(pattern, content))
    for pattern in class_patterns:
        classes.extend(match.group(1) for match in re.finditer(pattern, content))
    return _dedupe(functions), _dedupe(classes)


def _build_key_logic(
    functions: list[str],
    classes: list[str],
    dependencies: list[str],
    retrieved_chunks: list[str],
) -> list[str]:
    logic: list[str] = []
    if classes:
        logic.append(f"Defines core types/components such as {', '.join(classes[:3])}.")
    if functions:
        logic.append(f"Implements callable logic including {', '.join(functions[:4])}.")
    if dependencies:
        logic.append(f"Interacts with modules like {', '.join(dependencies[:4])}.")
    if retrieved_chunks:
        logic.append("Includes focused code excerpts for quick summarization.")
    if not logic:
        logic.append("Primarily holds configuration, markup, or declarative project data.")
    return logic[:4]


def _infer_file_purpose(
    relative_path: str,
    content: str,
    functions: list[str],
    classes: list[str],
) -> str:
    lower_path = relative_path.lower()
    if "readme" in lower_path:
        return "Project documentation and usage guidance."
    if lower_path.endswith((".json", ".yaml", ".yml", ".toml", ".ini")):
        return "Configuration or metadata used by the project."
    if "test" in lower_path:
        return "Test coverage for project behavior."
    if lower_path.endswith(".html"):
        return "HTML structure for a web interface or template."
    if lower_path.endswith((".css", ".scss")):
        return "Styling definitions for the user interface."
    if classes or functions:
        return "Source module that implements application logic."
    if "docker" in lower_path:
        return "Container or deployment configuration."
    if "package" in lower_path or "requirements" in lower_path:
        return "Dependency manifest for the project."
    if len(content.strip()) < 120:
        return "Small helper or metadata file."
    return "Project file contributing to application structure or behavior."


def _infer_role(relative_path: str, content: str) -> str:
    lower_path = relative_path.lower()
    if "/api" in lower_path or "controller" in lower_path or "route" in lower_path:
        return "API or request-handling layer"
    if "/components/" in lower_path or lower_path.endswith((".tsx", ".jsx")):
        return "User interface layer"
    if "/config/" in lower_path or lower_path.endswith((".json", ".yaml", ".yml", ".toml")):
        return "Configuration layer"
    if "model" in lower_path or "schema" in lower_path:
        return "Data model layer"
    if "service" in lower_path:
        return "Service or business logic layer"
    if "main." in lower_path or "app." in lower_path:
        return "Application entrypoint or orchestration layer"
    if "import " in content or "from " in content:
        return "Application support module"
    return "Supporting project asset"


def _detect_patterns(content: str, relative_path: str) -> list[str]:
    patterns: list[str] = []
    lower_path = relative_path.lower()
    if "class " in content:
        patterns.append("Object-oriented structure")
    if "async " in content or "await " in content:
        patterns.append("Asynchronous flow")
    if "interface " in content or "type " in content:
        patterns.append("Typed contract definitions")
    if "config" in lower_path or lower_path.endswith((".json", ".yaml", ".yml", ".toml")):
        patterns.append("Configuration-driven behavior")
    if "controller" in lower_path or "route" in lower_path:
        patterns.append("Layered request handling")
    return _dedupe(patterns)[:4]


def _dedupe(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def _build_project_synthesis_prompt(state: AnalyzerState) -> str:
    selected_files = _select_important_file_summaries(
        state.get("file_summaries", []),
        max_items=state["settings"].max_summary_files,
    )
    project_snapshot = {
        "project_name": state.get("project_name"),
        "folder_path": state.get("folder_path"),
        "file_count": len(state.get("file_records", [])),
        "selected_file_count": len(selected_files),
    }
    detailed_snapshot = {
        "skipped_large_files": state.get("skipped_large_files", []),
        "file_summaries": selected_files,
    }
    trimmed_detail = json.dumps(detailed_snapshot, indent=2)[: state["settings"].final_summary_char_limit]
    return f"""
Create a clean terminal-friendly summary of this project with exactly these headers:
[PROJECT OVERVIEW]
[ARCHITECTURE]
[FOLDER STRUCTURE]
[FILE SUMMARIES]
[DEPENDENCIES]
[EXECUTION FLOW]
[KEY INSIGHTS]

Requirements:
- Use concise bullets under each section.
- Mention when information is inferred.
- Summarize the most important files rather than dumping everything.
- Note skipped large files if any.
- Include the analyzer's own execution flow as well as the inferred project runtime flow when available.

Project overview seed:
{json.dumps(project_snapshot, indent=2)}

Detailed file analysis:
{trimmed_detail}
""".strip()


def _select_important_file_summaries(file_summaries: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    def score(summary: dict[str, Any]) -> tuple[int, int]:
        path = str(summary.get("relative_path", "")).lower()
        role = str(summary.get("role_in_system", "")).lower()
        priority = 0
        if "entrypoint" in role or path.endswith(("main.py", "app.py", "index.js", "index.ts")):
            priority += 5
        if any(token in path for token in ("service", "controller", "route", "api", "model", "schema")):
            priority += 4
        if any(token in path for token in ("config", "package.json", "requirements.txt", "docker-compose", "readme")):
            priority += 3
        priority += min(len(summary.get("dependencies", [])), 3)
        priority += min(len(summary.get("key_logic", [])), 3)
        return (priority, -len(path))

    ranked = sorted(file_summaries, key=score, reverse=True)
    return ranked[:max_items]


def _build_local_project_summary(state: AnalyzerState) -> str:
    file_records = state.get("file_records", [])
    file_summaries = state.get("file_summaries", [])
    selected_files = _select_important_file_summaries(
        file_summaries,
        max_items=state["settings"].max_summary_files,
    )
    detected_type = _infer_project_type(file_records)
    architecture = _infer_architecture(file_summaries)
    important_dirs = _summarize_directories(file_records)
    dependencies = _summarize_dependencies(file_summaries)
    execution_flow = _build_execution_flow(file_summaries)
    insights = _build_key_insights(state, file_summaries, detected_type)

    lines: list[str] = []
    lines.extend(
        _diagram_section(
            "PROJECT OVERVIEW",
            [
                f"Name: {state.get('project_name', 'unknown')}",
                f"Type: {detected_type}",
                f"Purpose: {_infer_project_purpose(selected_files)}",
            ],
        )
    )
    lines.append("")
    lines.extend(
        _diagram_section(
            "ARCHITECTURE",
            [
                f"Style: {architecture}",
                f"Key modules: {_format_key_modules(selected_files)}",
            ],
        )
    )
    lines.append("")
    lines.extend(_diagram_section("FOLDER STRUCTURE", important_dirs))
    lines.append("")
    lines.extend(_diagram_section("FILE SUMMARIES", [_format_file_summary(summary) for summary in selected_files]))
    lines.append("")
    lines.extend(_diagram_section("DEPENDENCIES", dependencies))
    lines.append("")
    lines.extend(_diagram_section("EXECUTION FLOW", execution_flow))
    lines.append("")
    lines.extend(_diagram_section("KEY INSIGHTS", insights))
    return "\n".join(lines)


def _infer_project_type(file_records: list[dict[str, Any]]) -> str:
    paths = [str(record.get("relative_path", "")).lower() for record in file_records]
    if any(path.endswith(("package.json", ".tsx", ".jsx")) for path in paths):
        return "Web application"
    if any(path.endswith((".py", "requirements.txt")) for path in paths):
        return "Python application or service"
    if any(path.endswith((".java", ".kt")) for path in paths):
        return "JVM application"
    if any(path.endswith((".go",)) for path in paths):
        return "Go service or CLI"
    return "Software project"


def _infer_architecture(file_summaries: list[dict[str, Any]]) -> str:
    roles = [str(summary.get("role_in_system", "")).lower() for summary in file_summaries]
    if any("user interface" in role for role in roles) and any("api" in role or "service" in role for role in roles):
        return "Layered application with separate frontend and backend responsibilities."
    if any("api" in role for role in roles):
        return "Service-oriented backend organized around routes and controllers."
    if any("configuration" in role for role in roles):
        return "Configuration-driven codebase with modular support files."
    return "Modular codebase organized around source files and shared utilities."


def _summarize_directories(file_records: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for record in file_records:
        path = Path(str(record.get("relative_path", ".")))
        top_level = path.parts[0] if len(path.parts) > 1 else "."
        counts[top_level] = counts.get(top_level, 0) + 1
    entries = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not entries:
        return ["Root contains no supported source files."]
    return [f"{name}: {count} analyzed files" for name, count in entries[:8]]


def _summarize_dependencies(file_summaries: list[dict[str, Any]]) -> list[str]:
    internal: set[str] = set()
    external: set[str] = set()
    for summary in file_summaries:
        for dependency in summary.get("dependencies", []):
            if dependency.startswith((".", "/")):
                internal.add(dependency)
            else:
                external.add(dependency)
    lines: list[str] = []
    lines.append(
        f"Internal modules: {', '.join(sorted(internal)[:8]) if internal else 'No explicit internal imports detected'}"
    )
    lines.append(
        f"External libraries: {', '.join(sorted(external)[:10]) if external else 'No explicit external libraries detected'}"
    )
    return lines


def _build_execution_flow(file_summaries: list[dict[str, Any]]) -> list[str]:
    flow = [
        "The analyzer validates the target path and scans for supported source files.",
        "Relevant files are loaded, chunked, and matched with the local cache when available.",
        "Local heuristics extract file roles, symbols, dependencies, and structural cues.",
        "The final report is assembled from the highest-value files and project-wide signals.",
    ]
    entrypoints = [
        summary.get("relative_path")
        for summary in file_summaries
        if "entrypoint" in str(summary.get("role_in_system", "")).lower()
    ]
    if entrypoints:
        flow.append(f"Inferred application entrypoint: {', '.join(entrypoints[:4])}.")
    return flow


def _build_key_insights(
    state: AnalyzerState,
    file_summaries: list[dict[str, Any]],
    detected_type: str,
) -> list[str]:
    skipped = state.get("skipped_large_files", [])
    config_files = sum(
        1 for summary in file_summaries if "configuration" in str(summary.get("role_in_system", "")).lower()
    )
    insights = [
        f"Detected project type: {detected_type}.",
        f"Analysis covered {len(file_summaries)} files using local structure-aware heuristics.",
        f"Configuration-oriented files detected: {config_files}.",
    ]
    if skipped:
        insights.append(f"Skipped oversized files: {', '.join(skipped[:5])}.")
    insights.append("Fast local synthesis is enabled by default to keep terminal output responsive.")
    return insights


def _infer_project_purpose(selected_files: list[dict[str, Any]]) -> str:
    if not selected_files:
        return "Unable to infer because no significant files were analyzed."
    joined_paths = " ".join(str(item.get("relative_path", "")).lower() for item in selected_files)
    if "habit" in joined_paths and "auth" in joined_paths:
        return "A web-based habit tracking system with authentication and user account flows."
    if "auth" in joined_paths and "profile" in joined_paths:
        return "A user-focused web application with authentication, profile management, and protected flows."
    primary = selected_files[0]
    return f"Inferred from {primary.get('relative_path')}: {primary.get('purpose', 'core project logic')}."


def _format_key_modules(selected_files: list[dict[str, Any]]) -> str:
    if not selected_files:
        return "No major modules detected"
    return ", ".join(str(item.get("relative_path")) for item in selected_files[:6])


def _format_file_summary(summary: dict[str, Any]) -> str:
    path = str(summary.get("relative_path", "unknown"))
    role = str(summary.get("role_in_system", "Supporting module"))
    purpose = str(summary.get("purpose", "No purpose inferred")).rstrip(".")
    dependencies = summary.get("dependencies", [])
    logic = summary.get("key_logic", [])

    details: list[str] = [purpose]
    if logic:
        details.append(_sentence_case(_compress_logic(logic[0])))
    if dependencies:
        details.append(f"Key dependencies: {', '.join(dependencies[:3])}")
    return f"{path} [{role}]: {'; '.join(details)}."


def _diagram_section(title: str, items: list[str]) -> list[str]:
    width = max(len(title) + 2, 24)
    top = f"+{'-' * width}+"
    header = f"| {title.ljust(width - 2)} |"
    lines = [top, header, top]
    if not items:
        lines.append("  (no items)")
        return lines
    for index, item in enumerate(items):
        branch = "└─" if index == len(items) - 1 else "├─"
        wrapped = _wrap_text(item, width=95)
        for wrapped_index, chunk in enumerate(wrapped):
            prefix = branch if wrapped_index == 0 else "│ "
            lines.append(f"{prefix} {chunk}")
    return lines


def _wrap_text(text: str, width: int = 95) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _compress_logic(text: str) -> str:
    replacements = {
        "Defines core types/components such as ": "Defines ",
        "Implements callable logic including ": "Implements ",
        "Interacts with modules like ": "Uses ",
        "Includes focused code excerpts for quick summarization.": "Contains representative logic for summarization",
        "Primarily holds configuration, markup, or declarative project data.": "Primarily contains configuration or declarative project data",
    }
    for source, target in replacements.items():
        if text.startswith(source):
            return text.replace(source, target, 1).rstrip(".")
    return text.rstrip(".")


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
