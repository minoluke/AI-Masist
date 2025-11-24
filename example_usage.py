"""
Example usage of the Draft Processing Pipeline

This script demonstrates how to use the DraftProcessor to generate,
execute, and analyze a machine learning experiment from a research idea.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from draft_processor import DraftProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class SimpleConfig:
    """Simple configuration class for testing"""

    @dataclass
    class ExecConfig:
        timeout: int = 300  # 5 minutes
        num_gpus: int = 0

    @dataclass
    class AgentConfig:
        @dataclass
        class CodeConfig:
            model: str = "gpt-4o"
            temp: float = 0.7

        @dataclass
        class FeedbackConfig:
            model: str = "gpt-4o"
            temp: float = 0.3

        @dataclass
        class VLMFeedbackConfig:
            model: str = "gpt-4o"
            temp: float = 0.3

        code: CodeConfig = CodeConfig()
        feedback: FeedbackConfig = FeedbackConfig()
        vlm_feedback: VLMFeedbackConfig = VLMFeedbackConfig()
        k_fold_validation: int = 1
        data_preview: bool = False

    @dataclass
    class ExperimentConfig:
        num_syn_datasets: int = 2

    exec: ExecConfig = ExecConfig()
    agent: AgentConfig = AgentConfig()
    experiment: ExperimentConfig = ExperimentConfig()


def main():
    """
    Main example: Process a simple research idea through the draft pipeline
    """
    # Define research task
    task_desc = """
    Implement a neural network classifier for the MNIST dataset.
    Use a simple feedforward network with 2 hidden layers.
    Train for 10 epochs and evaluate on test set.
    Track training and validation accuracy and loss.
    """

    # Define evaluation metrics
    evaluation_metrics = [
        "training accuracy",
        "validation accuracy",
        "test accuracy",
        "training loss",
        "validation loss",
    ]

    # Create configuration
    cfg = SimpleConfig()

    # Create draft processor
    processor = DraftProcessor(
        task_desc=task_desc, evaluation_metrics=evaluation_metrics, cfg=cfg
    )

    # Setup working directory
    working_dir = Path("./working_example")
    working_dir.mkdir(exist_ok=True)

    try:
        # Run the complete pipeline
        logger.info("Starting draft processing pipeline...")
        node = processor.process_draft(working_dir=str(working_dir))

        # Print results
        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)
        print(f"\nPlan:\n{node.plan}\n")
        print(f"Code Length: {len(node.code)} characters")
        print(f"Execution Time: {node.exec_time:.2f}s")
        print(f"Is Buggy: {node.is_buggy}")
        print(f"Metric: {node.metric}")
        print(f"Plots Generated: {len(node.plots) if node.plots else 0}")

        if node.analysis:
            print(f"\nAnalysis:\n{node.analysis}")

        if hasattr(node, "vlm_feedback_summary") and node.vlm_feedback_summary:
            print(f"\nVLM Feedback:\n{node.vlm_feedback_summary}")

        print("\n" + "=" * 80)

        # Save results
        results_file = working_dir / "results.txt"
        with open(results_file, "w") as f:
            f.write("Draft Processing Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Task: {task_desc}\n\n")
            f.write(f"Plan:\n{node.plan}\n\n")
            f.write(f"Is Buggy: {node.is_buggy}\n")
            f.write(f"Metric: {node.metric}\n\n")
            if node.analysis:
                f.write(f"Analysis:\n{node.analysis}\n\n")
            if hasattr(node, "vlm_feedback_summary"):
                f.write(f"VLM Feedback:\n{node.vlm_feedback_summary}\n")

        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        processor.cleanup()


if __name__ == "__main__":
    main()
