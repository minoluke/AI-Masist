"""
Plot Generator for Phase 5
Generates and executes plotting code for experiment visualizations
"""
import logging
from typing import Any, Tuple
from pathlib import Path

from .journal import Node
from .backend import query
from .utils.response import extract_code, extract_text_up_to_code

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
            "Data files (.npy, .npz, .csv) are located in the working directory",
            "You must dynamically discover and load these files - do NOT hardcode filenames",
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "First, find all data files in working_dir using os.listdir()",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "For multi-agent simulations, visualize scenario comparisons, metric distributions, and temporal patterns",
            "Include time-series plots (e.g., turns, message counts) if available",
            "ONLY plot data that exists in the discovered files - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles while also specifying the scenario being visualized.",
            "Make sure to use descriptive names for figures when saving e.g. always include the scenario name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. multiple runs), make sure to plot only aggregated results or at most 5 representative samples.",
            "Use the following experiment code to infer the data to plot: " + node.code,
        ]

        prompt_guideline += [
            "experiment_data structure: ",
            """
                # experiment_data.npz には 'scenarios' dict が含まれます。
                # 以下の2パターンがあります：

                # ========================================
                # パターン1: 複数条件がある場合（条件比較実験）
                # ========================================
                experiment_data = {
                    'scenarios': {
                        'CONDITION_A': {
                            'messages': [...],     # 全メッセージログ（ターン順）
                            'events': [...],       # 重要イベント（合意成立、衝突など）
                            'metrics': {...},      # この条件の評価指標
                            'config': {...},       # この条件の設定
                        },
                        'CONDITION_B': { ... },
                        # ... 他の条件
                    },
                    'metrics': {...},              # 全条件の集約メトリクス（オプション）
                }

                # ========================================
                # パターン2: 単一条件の場合
                # ========================================
                experiment_data = {
                    'scenarios': {
                        'default': {               # キーは 'default' となる
                            'messages': [...],
                            'events': [...],
                            'metrics': {...},
                            'config': {...},
                        },
                    },
                    'metrics': {...},
                }
            """,
            "Example data loading and plot saving code: ",
            """
                # Load experiment_data.npz
                npz_path = os.path.join(working_dir, 'experiment_data.npz')
                if os.path.exists(npz_path):
                    data = np.load(npz_path, allow_pickle=True)
                    exp_data = data['experiment_data'].item()  # Extract dict from 0-d array

                    if 'scenarios' in exp_data:
                        scenarios = exp_data['scenarios']
                        is_single_condition = len(scenarios) == 1 and 'default' in scenarios

                        if is_single_condition:
                            # 単一条件の場合: 'default' シナリオのみ
                            scenario_data = scenarios['default']
                            metrics = scenario_data.get('metrics', {})
                            messages = scenario_data.get('messages', [])
                            events = scenario_data.get('events', [])
                            config = scenario_data.get('config', {})

                            # プロット作成（タイトルに 'default' は含めない）
                            try:
                                plt.figure()
                                # ... plotting code using metrics/messages/events ...
                                plt.title('Experiment Results')  # 'default' は不要
                                plt.savefig(os.path.join(working_dir, 'results_plot.png'))
                                plt.close()
                            except Exception as e:
                                print(f"Error creating plot: {e}")
                                plt.close()

                        else:
                            # 複数条件の場合: 各シナリオを比較
                            for scenario_name, scenario_data in scenarios.items():
                                metrics = scenario_data.get('metrics', {})
                                messages = scenario_data.get('messages', [])
                                events = scenario_data.get('events', [])
                                config = scenario_data.get('config', {})

                                # 各シナリオのプロット
                                try:
                                    plt.figure()
                                    # ... plotting code ...
                                    plt.title(f'{scenario_name} - Results')
                                    plt.savefig(os.path.join(working_dir, f'{scenario_name}_plot.png'))
                                    plt.close()
                                except Exception as e:
                                    print(f"Error creating plot for {scenario_name}: {e}")
                                    plt.close()

                            # 条件間比較プロット（オプション）
                            try:
                                plt.figure()
                                # ... comparison plotting code ...
                                plt.title('Scenario Comparison')
                                plt.savefig(os.path.join(working_dir, 'scenario_comparison.png'))
                                plt.close()
                            except Exception as e:
                                print(f"Error creating comparison plot: {e}")
                                plt.close()

                    # Also check top-level metrics if present
                    if 'metrics' in exp_data:
                        aggregated_metrics = exp_data['metrics']
                        # Create aggregated plots if needed
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
