"""
Worker処理関数 - 単一ノードの全フェーズ処理
AI-Scientist-v2/parallel_agent.py::_process_node_wrapper()から移植
"""
import os
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def process_node_wrapper(
    node_data: Optional[Dict],
    task_desc: str,
    cfg: Any,
    evaluation_metrics: list = None,
    memory_summary: Optional[str] = None,
) -> Dict:
    """
    Wrapper function that processes a single node through all phases.

    Args:
        node_data: Parent node dict (None for draft nodes)
        task_desc: Research task description
        cfg: Configuration object
        evaluation_metrics: List of evaluation metrics
        memory_summary: Optional memory summary for context

    Returns:
        Dict representation of the processed node
    """
    from ..core.node import Node
    from ..executor.interpreter import Interpreter
    from ..phases.code_generator import CodeGenerator
    from ..phases.metrics_extractor import MetricsExtractor
    from ..phases.result_evaluator import ResultEvaluator
    from ..phases.plot_generator import PlotGenerator
    from ..phases.vlm_analyzer import VLMAnalyzer
    from ..core.metric import WorstMetricValue

    print("Starting process_node_wrapper")

    # Create process-specific workspace
    process_id = multiprocessing.current_process().name
    workspace = os.path.join(cfg.workspace_dir, f"process_{process_id}")
    os.makedirs(workspace, exist_ok=True)
    print(f"Process {process_id} using workspace: {workspace}")

    # Create process-specific working directory
    working_dir = os.path.join(workspace, "working")
    os.makedirs(working_dir, exist_ok=True)

    # Create interpreter instance for worker process
    print("Creating Interpreter")
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
            print(f"Recreated parent node: {parent_node.id}")
        else:
            parent_node = None
            print("No parent node - creating draft")

        # ========== Phase 1: Code Generation ==========
        print("Phase 1: Code Generation")
        if evaluation_metrics is None:
            evaluation_metrics = []
        code_generator = CodeGenerator(task_desc, evaluation_metrics, cfg, memory_summary)

        if parent_node is None:
            # Draft node
            print("Generating draft node")
            child_node = code_generator.generate()
        elif parent_node.is_buggy:
            # Debug node - TODO: implement in Week 2
            print(f"Debug node requested for {parent_node.id}, but not yet implemented")
            raise NotImplementedError("Debug nodes will be implemented in Week 2 (Task 4)")
        else:
            # Improve node - TODO: implement in Week 2
            print(f"Improve node requested for {parent_node.id}, but not yet implemented")
            raise NotImplementedError("Improve nodes will be implemented in Week 2 (Task 4)")

        # Set parent if provided
        if parent_node:
            child_node.parent = parent_node

        # ========== Phase 2: Code Execution ==========
        print("Phase 2: Code Execution")
        exec_result = process_interpreter.run(child_node.code, True)
        child_node.absorb_exec_result(exec_result)
        process_interpreter.cleanup_session()

        # ========== Phase 4: Result Evaluation ==========
        print("Phase 4: Result Evaluation")
        evaluator = ResultEvaluator(task_desc, cfg)
        evaluator.evaluate(child_node, exec_result, working_dir)

        # ========== Phase 3: Metrics Extraction ==========
        print("Phase 3: Metrics Extraction")
        metrics_extractor = MetricsExtractor(cfg, process_interpreter)

        # Check for saved data files
        data_files = [f for f in os.listdir(working_dir) if f.endswith(".npy")]
        if not data_files:
            logger.warning("No .npy files found. Setting metric to WorstMetricValue")
            child_node.metric = WorstMetricValue()
            child_node.is_buggy = True
        else:
            try:
                metrics_extractor.extract(child_node, working_dir)
            except Exception as e:
                logger.error(f"Error extracting metrics: {str(e)}")
                child_node.metric = WorstMetricValue()
                child_node.is_buggy = True
                child_node.parse_exc_type = str(e)

        # ========== Phase 5: Plot Generation ==========
        # Only generate plots if experiment was successful
        if not child_node.is_buggy:
            print("Phase 5: Plot Generation")
            plot_generator = PlotGenerator(cfg, process_interpreter)

            try:
                # Create experiment results directory
                base_dir = Path(cfg.workspace_dir).parent
                run_name = Path(cfg.workspace_dir).name
                exp_results_dir = (
                    base_dir
                    / "logs"
                    / run_name
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
                logger.info(f"Saved experiment code to {exp_code_path}")

                # Move experiment data files
                plots_dir = Path(working_dir)
                for exp_data_file in plots_dir.glob("*.npy"):
                    exp_data_path = exp_results_dir / exp_data_file.name
                    exp_data_file.resolve().rename(exp_data_path)
                    logger.info(f"Saved experiment data to {exp_data_path}")

            except Exception as e:
                logger.error(f"Error generating plots: {str(e)}")

        # ========== Phase 6: VLM Analysis ==========
        if not child_node.is_buggy and child_node.plots:
            print("Phase 6: VLM Analysis")
            vlm_analyzer = VLMAnalyzer(task_desc, cfg)

            try:
                vlm_analyzer.analyze(child_node)
                logger.info(f"Generated VLM analysis for node {child_node.id}")
            except Exception as e:
                logger.error(f"Error analyzing plots: {str(e)}")

        # ========== Return Result ==========
        print("Converting result to dict")
        result_data = child_node.to_dict()
        print(f"Result data keys: {result_data.keys()}")
        print(f"Result data size: {len(str(result_data))} chars")
        print("Returning result")
        return result_data

    except Exception as e:
        print(f"Worker process error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'process_interpreter' in locals() and process_interpreter:
            process_interpreter.cleanup_session()
