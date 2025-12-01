"""
ParallelAgent - 並列ノード処理エージェント
AI-Scientist-v2/ai_scientist/treesearch/parallel_agent.py から移植
Stage 1 (Initial Implementation) 専用版
"""
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import List, Optional, Any, Tuple
import random
import logging

from .journal import Node, Journal
from .node_processor import process_node_wrapper
from .backend import query
from .utils.response import extract_code, extract_text_up_to_code

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
        stage_name: str = None,
        best_stage1_node: Node = None,
        best_stage2_node: Node = None,
        best_stage3_node: Node = None,
    ):
        """
        Initialize ParallelAgent.

        Args:
            task_desc: Research task description
            cfg: Configuration object
            journal: Journal instance for storing nodes
            evaluation_metrics: List of evaluation metrics
            stage_name: Name of the current stage
            best_stage1_node: Best node from stage 1 (for stage 2 initialization)
            best_stage2_node: Best node from stage 2 (for stage 3 initialization)
            best_stage3_node: Best node from stage 3 (for stage 4 initialization)
        """
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.evaluation_metrics = evaluation_metrics or []
        self.stage_name = stage_name
        self.best_stage1_node = best_stage1_node
        self.best_stage2_node = best_stage2_node
        self.best_stage3_node = best_stage3_node
        self.data_preview = None

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
            best_node = self.journal.get_best_node(
                only_good=True, cfg=self.cfg, task_desc=self.task_desc
            )
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
                best_node = self.journal.get_best_node(
                    only_good=True, cfg=self.cfg, task_desc=self.task_desc
                )
                if best_node:
                    logger.info(f"Best node: {best_node.id}")
                    logger.info(f"  Metric: {best_node.metric}")

                    # Run multi-seed evaluation if enabled (num_seeds > 0)
                    if hasattr(self.cfg.agent, 'multi_seed_eval') and self.cfg.agent.multi_seed_eval.get("num_seeds", 0) > 0:
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
        Run multiple evaluations of the same node to get statistical metrics.
        Returns a list of nodes that executed the same code.

        Note: Since MAS simulations use LLM-based agents, the results will vary
        even with the same code due to LLM non-determinism.
        """
        from .node_processor import process_node_wrapper

        num_seeds = self.cfg.agent.multi_seed_eval.get("num_seeds", 3)

        logger.info(f"Starting multi-seed evaluation with {num_seeds} runs...")

        # Submit parallel jobs
        seed_nodes = []
        futures = []

        for i in range(num_seeds):
            # Use parent node's code directly (seed_eval=True skips Phase 1)
            node_data = node.to_dict()

            logger.debug(f"Submitting evaluation run {i}...")
            futures.append(
                self.executor.submit(
                    process_node_wrapper,
                    node_data,
                    self.task_desc,
                    self.cfg,
                    self.evaluation_metrics,
                    "",  # memory_summary
                    True,  # seed_eval=True: skip Phase 1, use parent's code
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
                logger.debug(f"Evaluation run {i} complete: node {result_node.id}")
            except Exception as e:
                logger.error(f"Error in multi-seed evaluation run {i}: {str(e)}")

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

    @property
    def _prompt_resp_fmt(self):
        """Response format for aggregation code generation"""
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> Tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                return nl_text, code

            logger.warning("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )

        logger.error("Final plan + code extraction attempt failed")
        return "", completion_text  # type: ignore

    def _generate_aggregation_code(self, parent_node: Node, seed_nodes: List[Node]) -> str:
        """
        Generate aggregated plots from multi-seed evaluation results using LLM.
        The LLM references each seed's plot_code and generates new code that
        aggregates the results with mean values and standard error bars.

        Args:
            parent_node: The original node that was evaluated
            seed_nodes: List of nodes from seed evaluation

        Returns:
            str: The plotting code for aggregated results
        """
        # Build prompt guideline
        prompt_guideline = [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "ONLY plot data that exists in experiment_data.npz - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles while also specifying the scenario being visualized.",
            "Make sure to use descriptive names for figures when saving e.g. always include the scenario name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Make sure to add legend for standard error bars and means if applicable",
        ]

        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data_path_list = # Make sure to use the correct experiment data path that's provided in the Experiment Data Path section
                    all_experiment_data = []
                    for experiment_data_path in experiment_data_path_list:
                        npz_path = os.path.join(os.getenv("MASIST_ROOT", os.getcwd()), experiment_data_path, 'experiment_data.npz')
                        if os.path.exists(npz_path):
                            data = np.load(npz_path, allow_pickle=True)
                            experiment_data = data['experiment_data'].item()
                            all_experiment_data.append(experiment_data)
                except Exception as e:
                    print(f'Error loading experiment data: {e}')

                try:
                    # First plot with error bars
                    plt.figure()
                    # ... aggregate data across seeds and plot with mean ± std ...
                    plt.savefig(os.path.join(working_dir, '[plot_name_1].png'))
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {e}")
                    plt.close()

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig(os.path.join(working_dir, '[plot_name_2].png'))
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {e}")
                    plt.close()
            """,
        ]

        # Build plotting prompt
        plotting_prompt = {
            "Introduction": (
                "You are an expert in data visualization and plotting. "
                "You are given a set of evaluation results and the code that was used to plot them. "
                "Your task is to write a new plotting code that aggregate the results "
                "e.g. for example, by adding mean values and standard error bars to the plots."
            ),
            "Instructions": {},
        }

        plotting_prompt["Instructions"] |= self._prompt_resp_fmt
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }

        # Add reference plotting codes from seed nodes
        plot_code_refs = []
        exp_data_paths = []
        for i, sn in enumerate(seed_nodes):
            if sn.plot_code:
                plot_code_refs.append(f"plotting code {i+1}:\n{sn.plot_code}\n")
            if sn.exp_results_dir:
                exp_data_paths.append(sn.exp_results_dir)

        plotting_prompt["Instructions"] |= {
            "Plotting code reference": "\n".join(plot_code_refs) if plot_code_refs else "No reference code available",
            "Experiment Data Path": "\n".join(exp_data_paths) if exp_data_paths else "No data paths available",
        }

        # Query LLM for aggregation code
        plan, code = self.plan_and_code_query(plotting_prompt)

        logger.info(f"Generated aggregation plan:\n{plan}")
        logger.debug(f"Generated aggregation code:\n{code}")

        # Ensure the code starts with imports
        if code and not code.strip().startswith("import"):
            code = "import matplotlib.pyplot as plt\nimport numpy as np\nimport os\n\n" + code

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
