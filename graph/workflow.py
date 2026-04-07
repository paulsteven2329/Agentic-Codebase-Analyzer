from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.nodes import (
    AnalyzerState,
    analysis_node,
    embedding_node,
    file_discovery_node,
    file_processing_node,
    input_node,
    output_node,
    retrieval_node,
    synthesis_node,
)


def build_workflow(_settings: object):
    graph = StateGraph(AnalyzerState)

    graph.add_node("input", input_node)
    graph.add_node("discover_files", file_discovery_node)
    graph.add_node("process_files", file_processing_node)
    graph.add_node("embed_documents", embedding_node)
    graph.add_node("retrieve_context", retrieval_node)
    graph.add_node("analyze_files", analysis_node)
    graph.add_node("synthesize_summary", synthesis_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "discover_files")
    graph.add_edge("discover_files", "process_files")
    graph.add_edge("process_files", "embed_documents")
    graph.add_edge("embed_documents", "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_files")
    graph.add_edge("analyze_files", "synthesize_summary")
    graph.add_edge("synthesize_summary", "output")
    graph.add_edge("output", END)

    return graph.compile()
