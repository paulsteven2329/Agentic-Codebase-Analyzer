from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import AnalyzerSettings
from graph.workflow import build_workflow
from utils.cli import CliReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a codebase with a LangGraph + RAG workflow powered by local Ollama models."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Absolute or relative path to the project folder to analyze.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the Ollama chat model (default: from settings).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the Ollama embedding model (default: from settings).",
    )
    parser.add_argument(
        "--vector-store",
        choices=("faiss", "chroma"),
        default=None,
        help="Vector store backend to use (default: from settings).",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=None,
        help="Maximum file size to read before skipping files.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--use-llm-summary",
        action="store_true",
        help="Use Ollama to generate the final project summary. Slower, but can be more polished.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    target_path = Path(args.path).expanduser().resolve()
    if not target_path.exists():
        reporter = CliReporter()
        reporter.error(f"Path does not exist: {target_path}")
        return 1
    if not target_path.is_dir():
        reporter = CliReporter()
        reporter.error(f"Path is not a directory: {target_path}")
        return 1

    settings = AnalyzerSettings.from_overrides(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        vector_store_backend=args.vector_store,
        max_file_bytes=args.max_file_bytes,
        use_llm_summary=args.use_llm_summary,
    )
    reporter = CliReporter(accent=settings.ui_color)
    reporter.print_banner(str(target_path))

    app = build_workflow(settings)
    try:
        result: dict = {}
        initial_state = {
            "folder_path": str(target_path),
            "settings": settings,
            "progress": [],
        }
        with reporter.live_status("Starting analysis"):
                for event in app.stream(initial_state, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        if isinstance(node_state, dict):
                            result.update(node_state)
                    reporter.update_status(_status_message_for_node(node_name, result))
    except KeyboardInterrupt:
        reporter.error("Analysis interrupted by user.")
        return 130
    except Exception as exc:
        reporter.error(f"Analyzer failed: {exc}")
        return 3

    summary = result.get("final_summary")
    if not summary:
        reporter.error("No summary could be generated.")
        return 2

    progress = result.get("progress", [])
    if progress:
        reporter.info(f"Workflow completed in {len(progress)} steps")

    reporter.print_summary(summary)
    return 0


def _status_message_for_node(node_name: str, state: dict) -> str:
    progress = state.get("progress", [])
    fallback_messages = {
        "input": "Preparing analysis target",
        "discover_files": "Scanning project files",
        "process_files": "Loading and chunking source files",
        "embed_documents": "Building or loading vector cache",
        "retrieve_context": "Preparing retrieval context",
        "analyze_files": "Analyzing file roles and dependencies",
        "synthesize_summary": "Synthesizing final project summary with local LLM",
        "output": "Finalizing terminal output",
    }
    return progress[-1] if progress else fallback_messages.get(node_name, "Analyzing codebase")


if __name__ == "__main__":
    raise SystemExit(main())
