"""
Journal - ノードの集合を管理するクラス
Stage 1: Initial Implementation用
"""
from __future__ import annotations
from typing import List, Optional
import logging

from .node import Node
from .metric import MetricValue

logger = logging.getLogger(__name__)


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
        logger.info(f"Added node {node.id} (step {node.step}) to journal")

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
        logger.info(f"Found {len(good)} good nodes out of {len(self.nodes)} total")
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

        # Find node with best metric (highest value)
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

        # Summary of good nodes
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
                    summary_parts.append(f"  Code: {node.code[:200]}...")  # First 200 chars
                summary_parts.append("")

        # Summary of buggy nodes
        if self.buggy_nodes:
            summary_parts.append("=== Failed Experiments ===")
            for node in self.buggy_nodes[:5]:  # Limit to 5 most recent failures
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
