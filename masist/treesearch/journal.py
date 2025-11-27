"""
Journal and Node - ノードとその集合を管理するクラス
AI-Scientist-v2のjournal.pyに準拠
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Any, List, Dict
import copy
import os
import json

from dataclasses_json import DataClassJsonMixin
from .utils.execution_result import ExecutionResult
from .utils.metric import MetricValue, WorstMetricValue

from rich import print

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Simplified helper functions
def trim_long_string(s: str, max_len: int = 1000) -> str:
    """Trim long strings for display"""
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def query(*args, **kwargs):
    """Placeholder for query function - not used in basic version"""
    raise NotImplementedError("query function not implemented in simplified version")


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    plan: str = field(default="", kw_only=True)  # type: ignore
    overall_plan: str = field(default="", kw_only=True)  # type: ignore
    code: str = field(default="", kw_only=True)  # type: ignore
    plot_code: str = field(default=None, kw_only=True)  # type: ignore
    plot_plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    exp_results_dir: str = field(default=None, kw_only=True)  # type: ignore

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- parsing info ----
    parse_metrics_plan: str = field(default="", kw_only=True)
    parse_metrics_code: str = field(default="", kw_only=True)
    parse_term_out: list[str] = field(default=None, kw_only=True)
    parse_exc_type: str | None = field(default=None, kw_only=True)
    parse_exc_info: dict | None = field(default=None, kw_only=True)
    parse_exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- plot execution info ----
    plot_term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    plot_exec_time: float = field(default=None, kw_only=True)  # type: ignore
    plot_exc_type: str | None = field(default=None, kw_only=True)
    plot_exc_info: dict | None = field(default=None, kw_only=True)
    plot_exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    is_buggy_plots: bool = field(default=None, kw_only=True)

    # ---- plotting ----
    plot_data: dict = field(default_factory=dict, kw_only=True)
    plots_generated: bool = field(default=False, kw_only=True)
    plots: List[str] = field(default_factory=list)
    plot_paths: List[str] = field(default_factory=list)

    # ---- VLM feedback ----
    plot_analyses: List[str] = field(default_factory=list)
    vlm_feedback_summary: List[str] = field(default_factory=list)
    datasets_successfully_tested: List[str] = field(default_factory=list)

    # ---- execution time feedback ----
    exec_time_feedback: str = field(default="", kw_only=True)

    # ---- ablation study ----
    ablation_name: str = field(default=None, kw_only=True)

    # ---- hyperparam tuning ----
    hyperparam_name: str = field(default=None, kw_only=True)

    # ---- seed node ----
    is_seed_node: bool = field(default=False, kw_only=True)
    is_seed_agg_node: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.children, list):
            self.children = set(self.children)
        if self.parent is not None and not isinstance(self.parent, str):
            self.parent.children.add(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k not in ("parent", "children"):
                setattr(result, k, copy.deepcopy(v, memo))

        result.parent = self.parent
        result.children = set()

        return result

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        if hasattr(self, "id"):
            state["id"] = self.id
        return state

    def __setstate__(self, state):
        """Set state during unpickling"""
        self.__dict__.update(state)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def absorb_plot_exec_result(self, plot_exec_result: ExecutionResult):
        """Absorb the result of executing the plotting code from this node."""
        self.plot_term_out = plot_exec_result.term_out
        self.plot_exec_time = plot_exec_result.exec_time
        self.plot_exc_type = plot_exec_result.exc_type
        self.plot_exc_info = plot_exec_result.exc_info
        self.plot_exc_stack = plot_exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            "code": self.code,
            "plan": self.plan,
            "overall_plan": (
                self.overall_plan if hasattr(self, "overall_plan") else None
            ),
            "plot_code": self.plot_code,
            "plot_plan": self.plot_plan,
            "step": self.step,
            "id": self.id,
            "ctime": self.ctime,
            "_term_out": self._term_out,
            "parse_metrics_plan": self.parse_metrics_plan,
            "parse_metrics_code": self.parse_metrics_code,
            "parse_term_out": self.parse_term_out,
            "parse_exc_type": self.parse_exc_type,
            "parse_exc_info": self.parse_exc_info,
            "parse_exc_stack": self.parse_exc_stack,
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exc_info": self.exc_info,
            "exc_stack": self.exc_stack,
            "analysis": self.analysis,
            "exp_results_dir": (
                str(Path(self.exp_results_dir).resolve().relative_to(os.getcwd()))
                if self.exp_results_dir
                else None
            ),
            "metric": {
                "value": self.metric.value if self.metric else None,
                "maximize": self.metric.maximize if self.metric else None,
                "name": self.metric.name if hasattr(self.metric, "name") else None,
                "description": (
                    self.metric.description
                    if hasattr(self.metric, "description")
                    else None
                ),
            },
            "is_buggy": self.is_buggy,
            "is_buggy_plots": self.is_buggy_plots,
            "parent_id": None if self.parent is None else self.parent.id,
            "children": [child.id for child in self.children] if self.children else [],
            "plot_data": self.plot_data,
            "plots_generated": self.plots_generated,
            "plots": self.plots,
            "plot_paths": (
                [
                    str(Path(p).resolve().relative_to(os.getcwd()))
                    for p in self.plot_paths
                ]
                if self.plot_paths
                else []
            ),
            "plot_analyses": [
                {
                    **analysis,
                    "plot_path": (
                        str(
                            Path(analysis["plot_path"])
                            .resolve()
                            .relative_to(os.getcwd())
                        )
                        if analysis.get("plot_path")
                        else None
                    ),
                }
                for analysis in self.plot_analyses
            ],
            "vlm_feedback_summary": self.vlm_feedback_summary,
            "datasets_successfully_tested": self.datasets_successfully_tested,
            "ablation_name": self.ablation_name,
            "hyperparam_name": self.hyperparam_name,
            "is_seed_node": self.is_seed_node,
            "is_seed_agg_node": self.is_seed_agg_node,
            "exec_time_feedback": self.exec_time_feedback,
        }

    @classmethod
    def from_dict(cls, data: Dict, journal: Optional[Any] = None) -> "Node":
        """Create a Node from a dictionary, optionally linking to journal for relationships"""
        parent_id = data.pop("parent_id", None)
        children = data.pop("children", [])

        metric_data = data.pop("metric", None)
        if metric_data:
            if isinstance(metric_data, dict):
                data["metric"] = MetricValue(
                    value=metric_data["value"],
                    maximize=metric_data["maximize"],
                    name=metric_data["name"],
                    description=metric_data["description"],
                )
            else:
                data["metric"] = (
                    WorstMetricValue()
                    if data.get("is_buggy")
                    else MetricValue(metric_data)
                )

        node = cls(**data)

        if journal is not None and parent_id:
            parent = journal.get_node_by_id(parent_id)
            if parent:
                node.parent = parent
                parent.children.add(node)

        return node


class Journal:
    """A collection of nodes representing the solution tree."""

    def __init__(self):
        self.nodes: List[Node] = []

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)
        buggy_status = "buggy" if node.is_buggy else "good"
        logger.info(f"Added node {node.id[:8]} to journal (step {node.step}, {buggy_status})")

    @property
    def draft_nodes(self) -> List[Node]:
        """Return a list of nodes representing initial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        """
        Return a list of nodes that are not considered buggy.
        Good nodes: is_buggy=False AND is_buggy_plots=False
        """
        good = [
            n for n in self.nodes
            if n.is_buggy is False and n.is_buggy_plots is False
        ]
        return good

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_metric_history(self) -> List[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes if n.metric is not None]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """
        Return the best solution found so far.

        Args:
            only_good: If True, only consider good nodes (is_buggy=False)

        Returns:
            Best node based on metric comparison, or None if no valid nodes
        """
        if only_good:
            candidates = self.good_nodes
            if not candidates:
                logger.warning("No good nodes found in journal")
                return None
        else:
            candidates = self.nodes

        if len(candidates) == 0:
            return None

        if len(candidates) == 1:
            return candidates[0]

        try:
            best = max(candidates, key=lambda n: n.metric if n.metric else float('-inf'))
            logger.info(f"Selected node {best.id} as best (metric: {best.metric})")
            return best
        except Exception as e:
            logger.error(f"Error selecting best node: {e}")
            return candidates[0]

    def generate_summary(self, include_code: bool = False) -> str:
        """
        Generate a summary of the research progress.

        Args:
            include_code: If True, include code snippets in summary

        Returns:
            Summary string for LLM context
        """
        if not self.nodes:
            return "No experiments conducted yet."

        summary_parts = []

        if self.good_nodes:
            summary_parts.append("=== Successful Experiments ===")
            for node in self.good_nodes:
                summary_parts.append(f"Node {node.id}:")
                summary_parts.append(f"  Plan: {node.plan}")
                if node.metric:
                    summary_parts.append(f"  Metric: {node.metric}")
                if node.analysis:
                    summary_parts.append(f"  Analysis: {node.analysis}")
                if include_code:
                    summary_parts.append(f"  Code: {node.code[:200]}...")
                summary_parts.append("")

        if self.buggy_nodes:
            summary_parts.append("=== Failed Experiments ===")
            for node in self.buggy_nodes[:5]:
                summary_parts.append(f"Node {node.id}:")
                if node.plan:
                    summary_parts.append(f"  Plan: {node.plan}")
                if node.exc_type:
                    summary_parts.append(f"  Error: {node.exc_type}")
                if node.analysis:
                    summary_parts.append(f"  Analysis: {node.analysis}")
                summary_parts.append("")

        summary_parts.append(f"Total nodes: {len(self.nodes)}, Good nodes: {len(self.good_nodes)}, Buggy nodes: {len(self.buggy_nodes)}")

        return "\n".join(summary_parts)

    def to_dict(self) -> dict:
        """Convert journal to a JSON-serializable dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes]
        }
