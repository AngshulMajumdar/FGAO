"""FGAO: Feedback-Greedy Agentic Orchestration.

A training-free numerical controller for tool-using pipelines and RAG.

Layers
------
FGAOController   step-wise controller: reset / should_stop / select_action / update
Workflow         declare goals + tools; FGAO decides the order (invoke, stream, batch, async)
FGAORAG          ready-made RAG orchestrator with pluggable retrievers, LLM and checks
"""
from .controller import ControllerStopped, FGAOConfig, FGAOController
from .estimator import EffectStatistics, blend
from .types import Action, ActionSpec, Condition, Decision, Observation, Scores, Semantics, Status, deficiency
from .workflow import CompiledWorkflow, Interrupt, Run, Step, Tool, ToolResult, Workflow, WorkflowResult
from .rag import (
    BM25Retriever, CachedLLM, Costs, DenseRetriever, Document, FGAORAG, FunctionLLM, FunctionRetriever, Priors,
    RAGResult, Source, Weights, default_sources,
)

__version__ = "1.0.0"

__all__ = [
    "FGAOConfig", "FGAOController", "ControllerStopped", "EffectStatistics", "blend",
    "Action", "ActionSpec", "Condition", "Decision", "Observation", "Scores", "Semantics", "Status", "deficiency",
    "Workflow", "CompiledWorkflow", "Tool", "ToolResult", "Step", "WorkflowResult", "Run", "Interrupt",
    "FGAORAG", "RAGResult", "Source", "Costs", "Priors", "Weights", "default_sources",
    "Document", "BM25Retriever", "DenseRetriever", "FunctionRetriever", "FunctionLLM", "CachedLLM",
    "__version__",
]
