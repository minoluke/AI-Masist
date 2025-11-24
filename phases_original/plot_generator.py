"""
Plot Generator for Phase 5
Generates and executes plotting code for experiment visualizations
"""
import logging
from typing import Any, Tuple
from pathlib import Path

from ..core.node import Node
from ..llm.backend import query
from ..utils.response import extract_code, extract_text_up_to_code

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate and execute plotting code for experiment results"""

    def __init__(self, cfg: Any, interpreter: Any):
        self.cfg = cfg
        self.interpreter = interpreter

    def generate_and_execute(
        self, node: Node, working_dir: str, plot_code_from_prev_stage: str = None
    ) -> None:
        """
        Generate plotting code and execute it with retry mechanism
        Combines _generate_plotting_code() and plot execution logic
        """
        # Step 1: Generate plotting code
        plotting_code = self._generate_plotting_code(
            node, working_dir, plot_code_from_prev_stage
        )

        # Step 2: Execute plotting code with retry (max 3 attempts)
        retry_count = 0
        while True:
            plot_exec_result = self.interpreter.run(plotting_code, True)
            self.interpreter.cleanup_session()

            # Store execution result in node
            node.plot_exec_result = plot_exec_result
            node.plot_term_out = plot_exec_result.term_out
            node.plot_exc_type = plot_exec_result.exc_type
            node.plot_exc_info = plot_exec_result.exc_info
            node.plot_exc_stack = plot_exec_result.exc_stack

            # Check if failed and retry if needed
            if node.plot_exc_type and retry_count < 3:
                logger.warning(
                    f"Plotting code failed with exception: {node.plot_exc_type}"
                )
                logger.warning(f"Plotting code output: {node.plot_term_out}")
                retry_count += 1
                continue
            else:
                break

        logger.info(f"Plotting execution completed for node {node.id}")

        # Step 3: Track generated plots
        self._track_generated_plots(node, working_dir)

    def _generate_plotting_code(
        self, node: Node, working_dir: str, plot_code_from_prev_stage: str = None
    ) -> str:
        """Generate code for plotting experiment results using LLM"""
        prompt_guideline = [
            "AVAILABLE DATA: ",
            "Experiment Data: experiment_data.npy",
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Use the following experiment code to infer the data to plot: " + node.code,
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
        ]

        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
                except Exception as e:
                    print(f'Error loading experiment data: {e}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {e}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {e}")
                    plt.close()
            """,
        ]

        # Build plotting prompt
        plotting_prompt = {
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= self._prompt_resp_fmt
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }

        # Get plotting code from LLM
        plan, code = self.plan_and_code_query(plotting_prompt)

        # Ensure the code starts with imports
        if not code.strip().startswith("import"):
            code = "import matplotlib.pyplot as plt\nimport numpy as np\n\n" + code

        node.plot_code = code
        node.plot_plan = plan

        return code

    def _track_generated_plots(self, node: Node, working_dir: str):
        """Track generated plot files and save to node"""
        plots_dir = Path(working_dir)
        if not plots_dir.exists():
            logger.warning(f"Working directory {working_dir} does not exist")
            return

        # Find all PNG files in working directory
        plot_files = list(plots_dir.glob("*.png"))
        if not plot_files:
            logger.warning("No plot files found in working directory")
            return

        logger.info(f"Found {len(plot_files)} plot files")

        # Save plots to node
        node.plots = []
        node.plot_paths = []
        for plot_file in plot_files:
            # Store absolute path for programmatic access
            node.plot_paths.append(str(plot_file.absolute()))
            # Store relative path for web visualization
            node.plots.append(str(plot_file.name))
            logger.info(f"Tracked plot: {plot_file.name}")

    @property
    def _prompt_resp_fmt(self):
        """Response format for plotting code generation"""
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for plotting. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. "
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
