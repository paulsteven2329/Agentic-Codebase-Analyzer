# Agentic Codebase Analyzer

Agentic Codebase Analyzer is a terminal-first Python application that scans a project folder, chunks source code, stores it in a local vector database, and uses a LangGraph workflow plus local Ollama models to produce an intelligent codebase summary.

## Features

- Recursive codebase scanning with configurable ignore rules
- LangGraph workflow with explicit agent-style processing stages
- RAG pipeline using LangChain splitters, Ollama embeddings, and FAISS or Chroma
- Fast local file analysis plus project-level synthesis using a local Ollama chat model
- Terminal-friendly structured output
- Rich CLI banner and live spinner-based progress feedback
- Cacheable vector store for faster repeated runs
- Large-file skipping to avoid loading oversized artifacts into context

## Project Structure

```text
agentic-codebase-analyzer/
├── main.py
├── graph/
│   ├── workflow.py
│   └── nodes.py
├── rag/
│   ├── pipeline.py
│   └── embeddings.py
├── utils/
│   ├── file_scanner.py
│   └── chunker.py
├── config/
│   └── settings.py
├── requirements.txt
└── README.md
```

## Architecture

The system follows a LangGraph workflow with these nodes:

1. Input node validates and normalizes the target folder path.
2. File discovery node scans the directory tree and filters relevant source files.
3. File processing node loads content and chunks files into RAG-ready documents.
4. Embedding node builds or loads a cached vector store.
5. Retrieval node prepares fast file-focused context from the loaded source content.
6. Analysis node extracts dependencies, symbols, patterns, and file roles locally.
7. Synthesis node converts file-level understanding into a project-wide summary.
8. Output node returns terminal-friendly text to the CLI.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed locally
- At least one local chat model and one embedding model pulled into Ollama

Recommended models:

- Chat model: `mistral` or `llama3`
- Embedding model: `nomic-embed-text`

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Install Ollama for your operating system, then start the Ollama service.

Example on Linux/macOS:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If the installer creates a system service, you do not need to run `ollama serve` manually.

### 4. Pull local models

In a second terminal:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

You can use `llama3` instead of `mistral` if you prefer.

## Usage

Run the analyzer against any local project folder:

```bash
python main.py --path /path/to/project
```

By default, the final summary is generated locally for speed. If you want a slower Ollama-written final synthesis, add `--use-llm-summary`.

Optional flags:

```bash
python main.py \
  --path /path/to/project \
  --use-llm-summary \
  --llm-model mistral \
  --embedding-model nomic-embed-text \
  --vector-store faiss \
  --max-file-bytes 1000000 \
  --log-level INFO
```

## Example Output

```text
[PROJECT OVERVIEW]
- Name: sample-app
- Type: Web application
- Purpose: An inferred React + API project for task tracking

[ARCHITECTURE]
- Monorepo with frontend and backend layers
- Retrieval-based analysis pipeline used by this analyzer
```

## Notes on RAG Behavior

- Files are filtered by extension and common build directories are ignored.
- Files larger than the configured byte limit are skipped and reported.
- Chunking uses code-aware section splitting where possible, then falls back to recursive character splitting.
- File-level analysis is optimized to avoid one LLM call per file, which improves medium-size repo performance significantly.
- Vector stores are cached under `.cache/` using a folder-and-file signature.

## Troubleshooting

- If Ollama is not running, the application will fail when embeddings or LLM calls begin.
- If your model names differ, pass them with `--llm-model` and `--embedding-model`.
- If `faiss-cpu` is difficult to install on your platform, switch to `--vector-store chroma`.

## Future Improvements

- AST-level chunking for richer language-aware summaries
- Concurrent per-file analysis for large repositories
- Incremental cache invalidation based on file hashes
- Optional JSON output mode for downstream automation
