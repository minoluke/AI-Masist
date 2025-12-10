"""
Worker処理関数 - 単一ノードの全フェーズ処理
AI-Scientist-v2/parallel_agent.py::_process_node_wrapper()から移植
"""
import os
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def _generate_exec_time_feedback(exec_time: float, timeout: int) -> str:
    """
    Generate feedback about execution time.

    Args:
        exec_time: Actual execution time in seconds
        timeout: Configured timeout in seconds

    Returns:
        Feedback string or empty string if execution time is normal
    """
    if exec_time is None:
        return ""

    # Too fast (likely incomplete execution)
    if exec_time < 5:
        return (
            "実行時間が非常に短い（5秒未満）です。"
            "シミュレーションが正常に完了したか確認してください。"
            "データの保存やログ出力が適切に行われているか確認してください。"
        )

    # Near timeout (>80% of timeout)
    if exec_time > timeout * 0.8:
        return (
            f"実行時間がタイムアウト({timeout}秒)に近づいています（{exec_time:.1f}秒）。"
            "max_round や max_turns を減らすか、処理を最適化することを検討してください。"
        )

    # Very long (>50% of timeout)
    if exec_time > timeout * 0.5:
        return (
            f"実行時間が比較的長い（{exec_time:.1f}秒）です。"
            "必要に応じて処理の効率化を検討してください。"
        )

    return ""


def process_node_wrapper(
    node_data: Optional[Dict],
    task_desc: str,
    cfg: Any,
    evaluation_metrics: list = None,
    memory_summary: Optional[str] = None,
    seed_eval: bool = False,
) -> Dict:
    """
    Wrapper function that processes a single node through all phases.

    Args:
        node_data: Parent node dict (None for draft nodes)
        task_desc: Research task description
        cfg: Configuration object
        evaluation_metrics: List of evaluation metrics
        memory_summary: Optional memory summary for context
        seed_eval: If True, skip Phase 1 and use parent node's code directly
                   (for multi-seed evaluation)

    Returns:
        Dict representation of the processed node
    """
    from .journal import Node
    from .interpreter import Interpreter
    from .code_generator import CodeGenerator
    from .metrics_extractor import MetricsExtractor
    from .result_evaluator import ResultEvaluator
    from .plot_generator import PlotGenerator
    from .vlm_analyzer import VLMAnalyzer
    from .utils.metric import WorstMetricValue

    logger.debug("Starting process_node_wrapper")

    # Create process-specific workspace
    process_id = multiprocessing.current_process().name
    workspace = os.path.join(cfg.workspace_dir, f"process_{process_id}")
    os.makedirs(workspace, exist_ok=True)
    logger.debug(f"Process {process_id} using workspace: {workspace}")

    # Create process-specific working directory
    working_dir = os.path.join(workspace, "working")
    os.makedirs(working_dir, exist_ok=True)

    # Create interpreter instance for worker process
    logger.debug("Creating Interpreter")
    process_interpreter = Interpreter(
        working_dir=workspace,
        timeout=cfg.exec.timeout,
        format_tb_ipython=cfg.exec.format_tb_ipython,
        agent_file_name=cfg.exec.agent_file_name,
    )

    try:
        # Recreate parent node from node_data if provided
        if node_data:
            parent_node = Node.from_dict(node_data, journal=None)
            logger.debug(f"Recreated parent node: {parent_node.id}")
        else:
            parent_node = None
            logger.debug("No parent node - creating draft")

        # ========== Phase 1: Code Generation ==========
        if evaluation_metrics is None:
            evaluation_metrics = []

        if seed_eval and parent_node is not None:
            # Seed evaluation: skip code generation, use parent node's code directly
            # Also reuse parse_metrics_code and plot_code from parent (AI-Scientist-v2 pattern)
            logger.debug("Phase 1: Skipped (seed_eval mode)")
            child_node = Node(
                plan="Seed evaluation node",
                code=parent_node.code,
                plot_code=parent_node.plot_code,
                parse_metrics_code=parent_node.parse_metrics_code,
                parse_metrics_plan=parent_node.parse_metrics_plan,
                parent=parent_node,
                is_seed_node=True,
            )
        else:
            # Normal flow: generate new code
            logger.debug("Phase 1: Code Generation")
            code_generator = CodeGenerator(task_desc, evaluation_metrics, cfg, memory_summary)

            if parent_node is None:
                # Draft node
                logger.debug("Generating draft node")
                child_node = code_generator.generate()
            elif parent_node.is_buggy:
                # Debug node
                logger.debug(f"Generating debug node for {parent_node.id}")
                child_node = code_generator.generate_debug(parent_node)
            else:
                # Improve node
                logger.debug(f"Generating improve node for {parent_node.id}")
                child_node = code_generator.generate_improve(parent_node)

            # Set parent if provided
            if parent_node:
                child_node.parent = parent_node

        # Short node ID for logging
        node_id_short = child_node.id[:8]

        # ========== Phase 2: Code Execution ==========
        logger.info(f"[Node {node_id_short}] Phase 2: Code Execution")
        exec_result = process_interpreter.run(child_node.code, True)
        child_node.absorb_exec_result(exec_result)
        process_interpreter.cleanup_session()

        # Log execution exception if any
        if exec_result.exc_type:
            logger.warning(f"[Node {node_id_short}] Phase 2 exception: {exec_result.exc_type}")
            if exec_result.exc_info:
                logger.debug(f"[Node {node_id_short}] Exception details: {exec_result.exc_info}")

        # Generate execution time feedback
        child_node.exec_time_feedback = _generate_exec_time_feedback(
            exec_result.exec_time, cfg.exec.timeout
        )
        if child_node.exec_time_feedback:
            logger.warning(f"[Node {node_id_short}] Exec time feedback: {child_node.exec_time_feedback}")

        # ========== Phase 3: Result Evaluation ==========
        logger.info(f"[Node {node_id_short}] Phase 3: Result Evaluation")
        evaluator = ResultEvaluator(task_desc, cfg)
        evaluator.evaluate(child_node, exec_result, working_dir)

        # ========== Phase 4: Metrics Extraction ==========
        # Note: AI-Scientist-v2 does NOT skip this phase even if is_buggy=True
        # Metrics may still be extractable even if the simulation had logical bugs
        logger.info(f"[Node {node_id_short}] Phase 4: Metrics Extraction")
        metrics_extractor = MetricsExtractor(cfg, process_interpreter)

        # Check for saved data files
        data_files = [f for f in os.listdir(working_dir) if f.endswith(".npy") or f.endswith(".npz")]
        if not data_files:
            logger.warning(f"[Node {node_id_short}] Marked as buggy (Phase 4: no .npy/.npz files)")
            child_node.metric = WorstMetricValue()
            child_node.is_buggy = True
        else:
            try:
                metrics_extractor.extract(child_node, working_dir)
            except Exception as e:
                logger.warning(f"[Node {node_id_short}] Marked as buggy (Phase 4: metrics extraction error)")
                logger.error(f"Error extracting metrics: {str(e)}")
                child_node.metric = WorstMetricValue()
                child_node.is_buggy = True
                child_node.parse_exc_type = str(e)

        # ========== Phase 5: Plot Generation ==========
        # Only generate plots if experiment was successful
        if not child_node.is_buggy:
            logger.info(f"[Node {node_id_short}] Phase 5: Plot Generation")
            plot_generator = PlotGenerator(cfg, process_interpreter)

            try:
                # Create experiment results directory
                exp_results_dir = (
                    Path(cfg.log_dir)
                    / "experiment_results"
                    / f"experiment_{child_node.id}_proc_{os.getpid()}"
                )
                child_node.exp_results_dir = str(exp_results_dir)
                exp_results_dir.mkdir(parents=True, exist_ok=True)

                # Generate and execute plots
                plot_generator.generate_and_execute(
                    child_node,
                    working_dir,
                    plot_code_from_prev_stage=None
                )

                # Save experiment code and data
                exp_code_path = exp_results_dir / "experiment_code.py"
                with open(exp_code_path, "w") as f:
                    f.write(child_node.code)
                logger.debug(f"Saved experiment code to {exp_code_path}")

                # Move experiment data files (.npy and .npz)
                plots_dir = Path(working_dir)
                for exp_data_file in plots_dir.glob("*.np[yz]"):
                    exp_data_path = exp_results_dir / exp_data_file.name
                    exp_data_file.resolve().rename(exp_data_path)
                    logger.debug(f"Saved experiment data to {exp_data_path}")

                # Move plot files (.png) and update plots with web-friendly paths
                # (AI-Scientist-v2 compatible: plots should contain relative paths for HTML visualization)
                child_node.plots = []  # Reset to update with correct paths
                child_node.plot_paths = []  # Reset to update with correct paths
                for plot_file in plots_dir.glob("*.png"):
                    final_path = exp_results_dir / plot_file.name
                    plot_file.resolve().rename(final_path)

                    # Create a web-friendly relative path for HTML visualization
                    # HTML is at: logs/0-run/stage_X_.../tree_plot.html
                    # Images are at: logs/0-run/experiment_results/experiment_XXX/file.png
                    web_path = f"../experiment_results/experiment_{child_node.id}_proc_{os.getpid()}/{plot_file.name}"
                    child_node.plots.append(web_path)  # For visualization
                    child_node.plot_paths.append(str(final_path.absolute()))  # For programmatic access
                    logger.debug(f"Moved plot to {final_path}, web_path: {web_path}")

            except Exception as e:
                logger.warning(f"[Node {node_id_short}] Plot generation failed (Phase 5)")
                logger.error(f"[Node {node_id_short}] Error: {str(e)}")

        # ========== Phase 6: VLM Analysis ==========
        if not child_node.is_buggy and child_node.plots:
            logger.info(f"[Node {node_id_short}] Phase 6: VLM Analysis")
            vlm_analyzer = VLMAnalyzer(task_desc, cfg)

            try:
                vlm_analyzer.analyze(child_node)
                logger.debug(f"Generated VLM analysis for node {child_node.id}")
            except Exception as e:
                logger.warning(f"[Node {node_id_short}] VLM analysis failed (Phase 6)")
                logger.error(f"[Node {node_id_short}] Error: {str(e)}")

        # ========== Return Result ==========
        logger.debug("Converting result to dict")
        result_data = child_node.to_dict()
        logger.debug(f"Result data keys: {result_data.keys()}")
        logger.debug(f"Result data size: {len(str(result_data))} chars")
        logger.debug("Returning result")
        return result_data

    except Exception as e:
        logger.error(f"Worker process error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        if 'process_interpreter' in locals() and process_interpreter:
            process_interpreter.cleanup_session()
