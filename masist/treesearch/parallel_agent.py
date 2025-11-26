"""
ParallelAgent - 並列ノード処理エージェント
AI-Scientist-v2/ai_scientist/treesearch/parallel_agent.py から移植
Stage 1 (Initial Implementation) 専用版
"""
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import List, Optional, Any
import random
import logging

from .journal import Node, Journal
from .node_processor import process_node_wrapper

logger = logging.getLogger(__name__)


class ParallelAgent:
    """
    Parallel agent for Stage 1: Initial Implementation.
    Manages parallel node processing using ProcessPoolExecutor.
    """

    def __init__(
        self,
        task_desc: str,
        cfg: Any,
        journal: Journal,
        evaluation_metrics: list = None,
    ):
        """
        Initialize ParallelAgent.

        Args:
            task_desc: Research task description
            cfg: Configuration object
            journal: Journal instance for storing nodes
            evaluation_metrics: List of evaluation metrics
        """
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.evaluation_metrics = evaluation_metrics or []

        # Worker configuration
        self.num_workers = cfg.agent.num_workers
        self.timeout = cfg.exec.timeout * 10  # Node processing timeout

        # Create process pool
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._is_shutdown = False

        logger.info(f"ParallelAgent initialized with {self.num_workers} workers")

    def _get_leaves(self, node: Node) -> List[Node]:
        """Get all leaf nodes in the tree rooted at node."""
        if not node.children:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    def _select_parallel_nodes(self) -> List[Optional[Node]]:
        """
        Select N nodes to process in parallel.
        Balancing between tree exploration and exploitation.

        Returns:
            List of nodes to process (None means new draft)
        """
        nodes_to_process = []
        processed_trees = set()
        search_cfg = self.cfg.agent.search

        print(f"[cyan]Selecting nodes... num_workers: {self.num_workers}[/cyan]")

        while len(nodes_to_process) < self.num_workers:
            # === Initial drafting phase (creating root nodes) ===
            print(
                f"Checking draft nodes... "
                f"journal.draft_nodes: {len(self.journal.draft_nodes)}, "
                f"num_drafts: {search_cfg.num_drafts}"
            )
            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                nodes_to_process.append(None)  # None means new draft
                continue

            # === Get viable trees (trees with at least one non-buggy leaf) ===
            viable_trees = [
                root
                for root in self.journal.draft_nodes
                if not all(leaf.is_buggy for leaf in self._get_leaves(root))
            ]

            # === Debugging phase (with some probability) ===
            if random.random() < search_cfg.debug_prob:
                print("Checking debuggable nodes...")
                try:
                    debuggable_nodes = [
                        n
                        for n in self.journal.buggy_nodes
                        if (
                            isinstance(n, Node)
                            and n.is_leaf
                            and n.debug_depth <= search_cfg.max_debug_depth
                        )
                    ]

                    if debuggable_nodes:
                        print(f"Found {len(debuggable_nodes)} debuggable nodes")
                        node = random.choice(debuggable_nodes)

                        # Get tree root
                        tree_root = node
                        while tree_root.parent:
                            tree_root = tree_root.parent

                        tree_id = id(tree_root)
                        if tree_id not in processed_trees or len(processed_trees) >= len(viable_trees):
                            nodes_to_process.append(node)
                            processed_trees.add(tree_id)
                            continue
                except Exception as e:
                    print(f"Error getting debuggable nodes: {e}")

            # === Improvement phase ===
            print("Checking good nodes for improvement...")
            good_nodes = self.journal.good_nodes
            if not good_nodes:
                nodes_to_process.append(None)  # Back to drafting
                continue

            # Get best node from unprocessed tree if possible
            best_node = self.journal.get_best_node(only_good=True)
            if best_node is None:
                nodes_to_process.append(None)
                continue

            # Get tree root
            tree_root = best_node
            while tree_root.parent:
                tree_root = tree_root.parent

            tree_id = id(tree_root)
            if tree_id not in processed_trees or len(processed_trees) >= len(viable_trees):
                nodes_to_process.append(best_node)
                processed_trees.add(tree_id)
                continue

            # If we can't use best node (tree already processed), try next best nodes
            for node in sorted(good_nodes, key=lambda n: n.metric if n.metric else float('-inf'), reverse=True):
                tree_root = node
                while tree_root.parent:
                    tree_root = tree_root.parent
                tree_id = id(tree_root)
                if tree_id not in processed_trees or len(processed_trees) >= len(viable_trees):
                    nodes_to_process.append(node)
                    processed_trees.add(tree_id)
                    break

        return nodes_to_process

    def step(self):
        """
        Execute one iteration of parallel node processing.
        Selects nodes, processes them in parallel, and adds results to journal.
        """
        print("=" * 60)
        print("ParallelAgent.step() - Selecting nodes to process")
        print("=" * 60)

        nodes_to_process = self._select_parallel_nodes()
        print(f"Selected nodes: {[n.id if n else 'Draft' for n in nodes_to_process]}")

        # Convert nodes to dicts for pickling
        node_data_list = []
        for node in nodes_to_process:
            if node:
                try:
                    node_data = node.to_dict()
                    node_data_list.append(node_data)
                except Exception as e:
                    logger.error(f"Error preparing node {node.id}: {str(e)}")
                    raise
            else:
                node_data_list.append(None)  # None means new draft

        # Generate memory summary for context
        memory_summary = self.journal.generate_summary(include_code=False)

        # Submit tasks to process pool
        print(f"Submitting {len(node_data_list)} tasks to process pool")
        futures = []
        for node_data in node_data_list:
            future = self.executor.submit(
                process_node_wrapper,
                node_data,
                self.task_desc,
                self.cfg,
                self.evaluation_metrics,
                memory_summary,
            )
            futures.append(future)

        # Collect results
        print("Waiting for results...")
        for i, future in enumerate(futures):
            try:
                result_data = future.result(timeout=self.timeout)

                # Recreate node from result dict
                result_node = Node.from_dict(result_data, self.journal)

                # Add node to journal
                self.journal.append(result_node)
                print(f"Added result node {result_node.id} to journal")

                # Log metrics
                if result_node.metric:
                    print(f"  Metric: {result_node.metric}")
                if result_node.is_buggy:
                    print(f"  [red]Node is buggy[/red]")

            except TimeoutError:
                logger.error(f"Worker {i} timed out")
                print(f"[red]Worker {i} timed out[/red]")
            except Exception as e:
                logger.error(f"Error processing node: {str(e)}")
                print(f"[red]Error processing node: {str(e)}[/red]")
                import traceback
                traceback.print_exc()

        print(f"Step complete. Journal now has {len(self.journal)} nodes")
        print(f"  Good nodes: {len(self.journal.good_nodes)}")
        print(f"  Buggy nodes: {len(self.journal.buggy_nodes)}")

    def run(self, max_steps: int = 20) -> bool:
        """
        Run Stage 1 until success or max_steps reached.

        Args:
            max_steps: Maximum number of iterations

        Returns:
            True if at least one good node was found, False otherwise
        """
        print("=" * 60)
        print(f"ParallelAgent.run() - Starting Stage 1 (max_steps={max_steps})")
        print("=" * 60)

        for step_num in range(max_steps):
            print(f"\n[cyan]Step {step_num + 1}/{max_steps}[/cyan]")

            self.step()

            # Check success condition
            if len(self.journal.good_nodes) > 0:
                print(f"\n[green]SUCCESS! Found {len(self.journal.good_nodes)} good node(s)[/green]")
                best_node = self.journal.get_best_node(only_good=True)
                if best_node:
                    print(f"Best node: {best_node.id}")
                    print(f"  Metric: {best_node.metric}")
                return True

        print(f"\n[yellow]Max steps reached. No good nodes found.[/yellow]")
        return False

    def cleanup(self):
        """Cleanup resources."""
        if not self._is_shutdown:
            self.executor.shutdown(wait=True)
            self._is_shutdown = True
            logger.info("ParallelAgent executor shutdown")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
