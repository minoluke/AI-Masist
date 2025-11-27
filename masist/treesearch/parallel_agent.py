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

        logger.debug(f"Selecting nodes... num_workers: {self.num_workers}")

        while len(nodes_to_process) < self.num_workers:
            # === Initial drafting phase (creating root nodes) ===
            logger.debug(
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
                logger.debug("Checking debuggable nodes...")
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
                        logger.debug(f"Found {len(debuggable_nodes)} debuggable nodes")
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
                    logger.debug(f"Error getting debuggable nodes: {e}")

            # === Improvement phase ===
            logger.debug("Checking good nodes for improvement...")
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
        logger.debug("=" * 60)
        logger.debug("ParallelAgent.step() - Selecting nodes to process")
        logger.debug("=" * 60)

        nodes_to_process = self._select_parallel_nodes()
        logger.debug(f"Selected nodes: {[n.id if n else 'Draft' for n in nodes_to_process]}")

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
        logger.debug(f"Submitting {len(node_data_list)} tasks to process pool")
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
        logger.debug("Waiting for results...")
        for i, future in enumerate(futures):
            try:
                result_data = future.result(timeout=self.timeout)

                # Recreate node from result dict
                result_node = Node.from_dict(result_data, self.journal)

                # Add node to journal
                self.journal.append(result_node)
                logger.debug(f"Added result node {result_node.id} to journal")

                # Log metrics
                if result_node.metric:
                    logger.debug(f"  Metric: {result_node.metric}")
                if result_node.is_buggy:
                    logger.debug(f"  Node is buggy")

            except TimeoutError:
                logger.error(f"Worker {i} timed out")
            except Exception as e:
                logger.error(f"Error processing node: {str(e)}")
                import traceback
                traceback.print_exc()

        logger.debug(f"Step complete. Journal now has {len(self.journal)} nodes")
        logger.debug(f"  Good nodes: {len(self.journal.good_nodes)}")
        logger.debug(f"  Buggy nodes: {len(self.journal.buggy_nodes)}")

    def run(self, max_steps: int = 20) -> bool:
        """
        Run Stage 1 until success or max_steps reached.

        Args:
            max_steps: Maximum number of iterations

        Returns:
            True if at least one good node was found, False otherwise
        """
        logger.info("=" * 60)
        logger.info(f"ParallelAgent.run() - Starting Stage 1 (max_steps={max_steps})")
        logger.info("=" * 60)

        for step_num in range(max_steps):
            logger.info(f"Step {step_num + 1}/{max_steps}")

            self.step()

            # Check success condition
            if len(self.journal.good_nodes) > 0:
                logger.info(f"SUCCESS! Found {len(self.journal.good_nodes)} good node(s)")
                best_node = self.journal.get_best_node(only_good=True)
                if best_node:
                    logger.info(f"Best node: {best_node.id}")
                    logger.info(f"  Metric: {best_node.metric}")

                    # Run multi-seed evaluation if enabled
                    if hasattr(self.cfg.agent, 'multi_seed_eval') and self.cfg.agent.multi_seed_eval.enabled:
                        logger.info("Running multi-seed evaluation...")
                        seed_nodes = self._run_multi_seed_evaluation(best_node)
                        if seed_nodes:
                            logger.info(f"Multi-seed evaluation complete. {len(seed_nodes)} seed nodes created.")
                            self._run_plot_aggregation(best_node, seed_nodes)

                return True

        logger.warning("Max steps reached. No good nodes found.")
        return False

    def _run_multi_seed_evaluation(self, node: Node) -> List[Node]:
        """
        Run multiple seeds of the same node to get statistical metrics.
        Returns a list of nodes with different random seeds.
        """
        from .node_processor import process_node_wrapper

        num_seeds = self.cfg.agent.multi_seed_eval.num_seeds
        node_code = node.code

        logger.info(f"Starting multi-seed evaluation with {num_seeds} seeds...")

        # Submit parallel jobs for different seeds
        seed_nodes = []
        futures = []

        for seed in range(num_seeds):
            # Create seed-injected code
            seed_code = (
                f"# Set random seed\n"
                f"import random\n"
                f"import numpy as np\n"
                f"\n"
                f"seed = {seed}\n"
                f"random.seed(seed)\n"
                f"np.random.seed(seed)\n"
                f"\n"
            ) + node_code

            # Create node data with seed-injected code
            node_data = node.to_dict()
            node_data["code"] = seed_code

            logger.debug(f"Submitting seed {seed} evaluation...")
            futures.append(
                self.executor.submit(
                    process_node_wrapper,
                    node_data,
                    self.task_desc,
                    self.cfg,
                    self.evaluation_metrics,
                    "",  # memory_summary
                )
            )

        # Collect results
        for i, future in enumerate(futures):
            try:
                result_data = future.result(timeout=self.timeout)
                result_node = Node.from_dict(result_data, self.journal)
                result_node.is_seed_node = True
                result_node.parent = node
                self.journal.append(result_node)
                seed_nodes.append(result_node)
                logger.debug(f"Seed {i} evaluation complete: node {result_node.id}")
            except Exception as e:
                logger.error(f"Error in multi-seed evaluation seed {i}: {str(e)}")

        return seed_nodes

    def _run_plot_aggregation(self, node: Node, seed_nodes: List[Node]) -> Optional[Node]:
        """
        Generate an aggregation node for seed evaluation results.
        Creates aggregated plots with error bars.
        """
        from .interpreter import Interpreter
        from pathlib import Path
        import os

        if not seed_nodes:
            logger.warning("No seed nodes to aggregate")
            return None

        try:
            logger.debug("Generating aggregation plots...")

            # Set MASIST_ROOT environment variable for path resolution
            os.environ["MASIST_ROOT"] = os.getcwd()

            # Generate aggregation plotting code
            agg_code = self._generate_aggregation_code(node, seed_nodes)

            # Create interpreter for aggregation
            process_interpreter = Interpreter(
                working_dir=self.cfg.workspace_dir,
                timeout=self.cfg.exec.timeout,
                format_tb_ipython=self.cfg.exec.format_tb_ipython,
                agent_file_name=self.cfg.exec.agent_file_name,
            )

            try:
                working_dir = os.path.join(self.cfg.workspace_dir, "working")
                os.makedirs(working_dir, exist_ok=True)

                # Execute aggregation code
                exec_result = process_interpreter.run(agg_code, True)
                process_interpreter.cleanup_session()

                # Create aggregation node
                agg_node = Node(
                    code=agg_code,
                    plan="Seed evaluation aggregation",
                    parent=node,
                    is_seed_agg_node=True,
                )
                agg_node.absorb_exec_result(exec_result)

                # Collect generated plots
                plots_dir = Path(working_dir)
                if plots_dir.exists():
                    base_dir = Path(self.cfg.workspace_dir).parent
                    run_name = Path(self.cfg.workspace_dir).name
                    exp_results_dir = (
                        base_dir / "logs" / run_name / "experiment_results"
                        / f"seed_aggregation_{agg_node.id}"
                    )
                    exp_results_dir.mkdir(parents=True, exist_ok=True)
                    agg_node.exp_results_dir = str(exp_results_dir)

                    # Save aggregation code
                    with open(exp_results_dir / "aggregation_code.py", "w") as f:
                        f.write(agg_code)

                    # Move generated plots
                    for plot_file in plots_dir.glob("*.png"):
                        final_path = exp_results_dir / plot_file.name
                        plot_file.resolve().rename(final_path)
                        agg_node.plots.append(str(final_path))
                        agg_node.plot_paths.append(str(final_path.absolute()))

                agg_node.is_buggy = False
                self.journal.append(agg_node)
                logger.info(f"Aggregation complete: node {agg_node.id}")
                return agg_node

            finally:
                if process_interpreter:
                    process_interpreter.cleanup_session()

        except Exception as e:
            logger.error(f"Error in plot aggregation: {str(e)}")
            return None

    def _generate_aggregation_code(self, parent_node: Node, seed_nodes: List[Node]) -> str:
        """Generate code to aggregate metrics from seed evaluations."""
        # Collect experiment data paths from seed nodes (relative paths)
        data_paths = []
        for sn in seed_nodes:
            if sn.exp_results_dir:
                data_paths.append(sn.exp_results_dir)

        code = f'''
import os
import numpy as np
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Seed evaluation data paths (relative to MASIST_ROOT)
data_paths = {data_paths}

# Resolve paths using MASIST_ROOT environment variable
masist_root = os.getenv("MASIST_ROOT", os.getcwd())

def extract_metrics(exp_data):
    """Extract metrics from experiment_data (supports both old and new format)."""
    # New format: scenarios dict with per-scenario metrics
    if 'scenarios' in exp_data and isinstance(exp_data['scenarios'], dict):
        scenario_metrics = {{}}
        for scenario_name, scenario_data in exp_data['scenarios'].items():
            if isinstance(scenario_data, dict) and 'metrics' in scenario_data:
                scenario_metrics[scenario_name] = scenario_data['metrics']
        # Also check for top-level aggregated metrics
        if 'metrics' in exp_data:
            scenario_metrics['_aggregated'] = exp_data['metrics']
        return scenario_metrics
    # Old format: direct metrics dict
    elif 'metrics' in exp_data:
        return {{'_default': exp_data['metrics']}}
    return {{}}

# Collect metrics from each seed (organized by scenario)
all_scenario_metrics = {{}}  # scenario_name -> list of metrics dicts

for path in data_paths:
    # Resolve relative path using MASIST_ROOT
    full_path = os.path.join(masist_root, path) if not os.path.isabs(path) else path
    npz_path = os.path.join(full_path, 'experiment_data.npz')
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            exp_data = data['experiment_data'].item()
            scenario_metrics = extract_metrics(exp_data)
            for scenario_name, metrics in scenario_metrics.items():
                if scenario_name not in all_scenario_metrics:
                    all_scenario_metrics[scenario_name] = []
                all_scenario_metrics[scenario_name].append(metrics)
        except Exception as e:
            print(f"Error loading {{npz_path}}: {{e}}")

print(f"Loaded metrics from {{len(data_paths)}} seeds")
print(f"Scenarios found: {{list(all_scenario_metrics.keys())}}")

# Aggregate metrics per scenario
aggregated_by_scenario = {{}}

for scenario_name, metrics_list in all_scenario_metrics.items():
    if not metrics_list:
        continue

    # Get all metric keys for this scenario
    all_keys = set()
    for m in metrics_list:
        if isinstance(m, dict):
            all_keys.update(m.keys())

    # Calculate mean and std for each metric
    aggregated = {{}}
    for key in all_keys:
        values = []
        for m in metrics_list:
            if isinstance(m, dict) and key in m:
                val = m[key]
                if isinstance(val, (int, float)):
                    values.append(val)
        if values:
            aggregated[key] = {{
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values,
                'n': len(values)
            }}

    if aggregated:
        aggregated_by_scenario[scenario_name] = aggregated
        print(f"\\n[{{scenario_name}}]")
        for key, stats in aggregated.items():
            print(f"  {{key}}: mean={{stats['mean']:.4f}}, std={{stats['std']:.4f}} (n={{stats['n']}})")

# Create aggregation plots (one per scenario or combined)
if aggregated_by_scenario:
    # Skip internal aggregated metrics for plotting
    plot_scenarios = {{k: v for k, v in aggregated_by_scenario.items() if not k.startswith('_')}}

    if len(plot_scenarios) == 1:
        # Single scenario: simple bar plot
        scenario_name, aggregated = list(plot_scenarios.items())[0]
        plt.figure(figsize=(10, 6))
        keys = list(aggregated.keys())
        means = [aggregated[k]['mean'] for k in keys]
        stds = [aggregated[k]['std'] for k in keys]

        x = np.arange(len(keys))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xticks(x, keys, rotation=45, ha='right')
        plt.ylabel('Value')
        plt.title(f'Aggregated Metrics - {{scenario_name}} (Mean ± Std)')
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, 'aggregated_metrics.png'), dpi=150)
        plt.close()
        print("\\nSaved aggregated_metrics.png")

    elif len(plot_scenarios) > 1:
        # Multiple scenarios: grouped bar plot for comparison
        all_metric_keys = set()
        for agg in plot_scenarios.values():
            all_metric_keys.update(agg.keys())
        all_metric_keys = sorted(all_metric_keys)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_metric_keys))
        width = 0.8 / len(plot_scenarios)

        for i, (scenario_name, aggregated) in enumerate(plot_scenarios.items()):
            means = [aggregated.get(k, {{}}).get('mean', 0) for k in all_metric_keys]
            stds = [aggregated.get(k, {{}}).get('std', 0) for k in all_metric_keys]
            offset = (i - len(plot_scenarios)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=scenario_name, capsize=3, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(all_metric_keys, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Aggregated Metrics by Scenario (Mean ± Std)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, 'aggregated_metrics_comparison.png'), dpi=150)
        plt.close()
        print("\\nSaved aggregated_metrics_comparison.png")

# Save aggregated data
np.savez_compressed(
    os.path.join(working_dir, 'aggregated_data.npz'),
    aggregated_by_scenario=np.array(aggregated_by_scenario, dtype=object),
    num_seeds={len(seed_nodes)}
)
print("Saved aggregated_data.npz")
'''
        return code

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
