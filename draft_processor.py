"""
Draft Processor - Orchestrates the 6-phase draft implementation pipeline
Phase 1: Code Generation
Phase 2: Execution
Phase 3: Evaluation
Phase 4: Metrics Extraction
Phase 5: Plot Generation
Phase 6: VLM Analysis
"""
import logging
import os
from pathlib import Path
from typing import Any

from .core.node import Node
from .core.execution_result import ExecutionResult
from .executor.interpreter import Interpreter
from .phases import (
    CodeGenerator,
    ResultEvaluator,
    MetricsExtractor,
    PlotGenerator,
    VLMAnalyzer,
)

logger = logging.getLogger(__name__)


class DraftProcessor:
    """
    Orchestrates the complete draft processing pipeline
    Implements the 6-phase workflow from code generation to VLM analysis
    """

    def __init__(self, task_desc: str, evaluation_metrics: list, cfg: Any):
        """
        Initialize the draft processor

        Args:
            task_desc: Research idea description
            evaluation_metrics: List of metrics to evaluate
            cfg: Configuration object with agent settings
        """
        self.task_desc = task_desc
        self.evaluation_metrics = evaluation_metrics
        self.cfg = cfg

        # Initialize interpreter for code execution
        self.interpreter = Interpreter(
            timeout=cfg.exec.timeout,
            num_gpus=getattr(cfg.exec, "num_gpus", 0),
        )

        # Initialize phase processors
        self.code_generator = CodeGenerator(
            task_desc=task_desc,
            evaluation_metrics=evaluation_metrics,
            cfg=cfg,
            memory_summary=None,  # Can be set later if needed
        )

        self.result_evaluator = ResultEvaluator(task_desc=task_desc, cfg=cfg)

        self.metrics_extractor = MetricsExtractor(cfg=cfg, interpreter=self.interpreter)

        self.plot_generator = PlotGenerator(cfg=cfg, interpreter=self.interpreter)

        self.vlm_analyzer = VLMAnalyzer(task_desc=task_desc, cfg=cfg)

    def process_draft(self, working_dir: str = None) -> Node:
        """
        Execute the complete 6-phase draft processing pipeline

        Args:
            working_dir: Working directory for code execution

        Returns:
            Node with all results populated
        """
        # Setup working directory
        if working_dir is None:
            working_dir = os.path.join(os.getcwd(), "working")

        working_path = Path(working_dir)
        working_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting Draft Processing Pipeline")
        logger.info("=" * 80)

        # Phase 1: Code Generation
        logger.info("Phase 1/6: Generating code...")
        node = self._phase1_code_generation()
        print(f"\n[green]✓ Phase 1 Complete: Generated {len(node.code)} chars of code[/green]\n")

        # Phase 2: Execution
        logger.info("Phase 2/6: Executing code...")
        exec_result = self._phase2_execution(node, working_dir)
        print(f"[green]✓ Phase 2 Complete: Execution time {exec_result.exec_time:.2f}s[/green]\n")

        # Phase 3: Evaluation
        logger.info("Phase 3/6: Evaluating results...")
        self._phase3_evaluation(node, exec_result)
        print(f"[green]✓ Phase 3 Complete: is_buggy={node.is_buggy}[/green]\n")

        # Phase 4: Metrics Extraction
        logger.info("Phase 4/6: Extracting metrics...")
        self._phase4_metrics_extraction(node, working_dir)
        print(f"[green]✓ Phase 4 Complete: metric={node.metric}[/green]\n")

        # Phase 5: Plot Generation
        logger.info("Phase 5/6: Generating plots...")
        self._phase5_plot_generation(node, working_dir)
        print(f"[green]✓ Phase 5 Complete: {len(node.plots)} plots generated[/green]\n")

        # Phase 6: VLM Analysis
        logger.info("Phase 6/6: Analyzing plots with VLM...")
        self._phase6_vlm_analysis(node)
        print(f"[green]✓ Phase 6 Complete: VLM analysis done[/green]\n")

        logger.info("=" * 80)
        logger.info("Draft Processing Pipeline Complete")
        logger.info("=" * 80)

        return node

    def _phase1_code_generation(self) -> Node:
        """Phase 1: Generate code from research idea"""
        node = self.code_generator.generate()
        logger.info(f"Generated code: {len(node.code)} characters")
        logger.info(f"Generated plan: {node.plan[:100]}...")
        return node

    def _phase2_execution(self, node: Node, working_dir: str) -> ExecutionResult:
        """Phase 2: Execute the generated code"""
        # Inject working directory into code
        code_with_working_dir = self._inject_working_dir(node.code, working_dir)

        # Execute code
        exec_result = self.interpreter.run(code_with_working_dir, capture_output=True)
        self.interpreter.cleanup_session()

        logger.info(f"Execution completed in {exec_result.exec_time:.2f}s")
        if exec_result.exc_type:
            logger.error(f"Execution error: {exec_result.exc_type}")

        return exec_result

    def _phase3_evaluation(self, node: Node, exec_result: ExecutionResult):
        """Phase 3: Evaluate execution results"""
        self.result_evaluator.evaluate(node, exec_result)
        logger.info(f"Evaluation complete: is_buggy={node.is_buggy}")
        if node.is_buggy:
            logger.warning(f"Bug detected: {node.analysis}")

    def _phase4_metrics_extraction(self, node: Node, working_dir: str):
        """Phase 4: Extract metrics from experiment data"""
        self.metrics_extractor.extract(node, working_dir)
        logger.info(f"Metrics extraction complete: {node.metric}")

    def _phase5_plot_generation(self, node: Node, working_dir: str):
        """Phase 5: Generate and execute plotting code"""
        self.plot_generator.generate_and_execute(node, working_dir)
        logger.info(f"Generated {len(node.plots) if node.plots else 0} plots")

    def _phase6_vlm_analysis(self, node: Node):
        """Phase 6: Analyze plots with Vision-Language Model"""
        if node.plots:
            self.vlm_analyzer.analyze(node)
            logger.info(f"VLM analysis complete: {len(node.plot_analyses)} analyses")
        else:
            logger.warning("No plots to analyze, skipping VLM phase")

    def _inject_working_dir(self, code: str, working_dir: str) -> str:
        """
        Inject working directory setup into code if not present
        """
        working_dir_setup = f"""import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
"""

        # Check if working_dir is already defined in code
        if "working_dir" in code and "os.path.join" in code:
            return code

        # Otherwise, prepend it
        return working_dir_setup + "\n" + code

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "interpreter"):
            self.interpreter.cleanup_session()
