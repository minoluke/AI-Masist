"""
UniTest - Draft Processing Pipeline

Standalone implementation of AI-Scientist-v2's draft processing workflow.
Implements the 6-phase pipeline from code generation to VLM analysis.

Usage:
    from uniTest import DraftProcessor

    processor = DraftProcessor(
        task_desc="Your research idea...",
        evaluation_metrics=["accuracy", "loss"],
        cfg=config
    )

    node = processor.process_draft(working_dir="./working")
"""

from .draft_processor import DraftProcessor
from .core import Node, MetricValue, WorstMetricValue, ExecutionResult
from .phases import (
    CodeGenerator,
    ResultEvaluator,
    MetricsExtractor,
    PlotGenerator,
    VLMAnalyzer,
)

__version__ = "0.1.0"
__all__ = [
    "DraftProcessor",
    "Node",
    "MetricValue",
    "WorstMetricValue",
    "ExecutionResult",
    "CodeGenerator",
    "ResultEvaluator",
    "MetricsExtractor",
    "PlotGenerator",
    "VLMAnalyzer",
]
