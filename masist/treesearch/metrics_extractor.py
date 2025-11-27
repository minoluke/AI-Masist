"""
Metrics Extractor for Phase 4
3-step metrics extraction: generate parse code -> execute -> structure metrics
"""
import logging
import os
from typing import cast, Any

from .journal import Node
from .utils.metric import MetricValue, WorstMetricValue
from .utils.execution_result import ExecutionResult
from .backend import query
from .utils.function_specs import metric_parse_spec

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """Extract metrics from experiment data using 3-step process"""

    def __init__(self, cfg: Any, interpreter: Any):
        self.cfg = cfg
        self.interpreter = interpreter

    def extract(self, node: Node, working_dir: str):
        """
        Extract metrics from experiment_data.npy
        3 steps: 1) Generate parse code, 2) Execute parse code, 3) Structure metrics
        """
        # Check if data files exist
        data_files = [f for f in os.listdir(working_dir) if f.endswith((".npy", ".npz"))]
        if not data_files:
            logger.warning(
                "No .npy or .npz files found in working directory. Data may not have been saved properly."
            )
            return

        try:
            # Step 1: Generate parse code
            parse_metrics_code, parse_metrics_plan = self._generate_parse_code(node)
            node.parse_metrics_code = parse_metrics_code
            node.parse_metrics_plan = parse_metrics_plan

            # Step 2: Execute parse code
            metrics_exec_result = self.interpreter.run(parse_metrics_code, True)
            self.interpreter.cleanup_session()

            node.parse_term_out = metrics_exec_result.term_out
            node.parse_exc_type = metrics_exec_result.exc_type
            node.parse_exc_info = metrics_exec_result.exc_info
            node.parse_exc_stack = metrics_exec_result.exc_stack

            # Step 3: Structure metrics
            if metrics_exec_result.exc_type is None:
                self._structure_metrics(node, metrics_exec_result)
            else:
                # Failure Pattern 1: Parse code execution error
                logger.error(
                    f"Error executing metrics parsing code: {metrics_exec_result.exc_info}"
                )
                node.metric = WorstMetricValue()
                node.is_buggy = True

        except Exception as e:
            # Failure Pattern 3: Exception catch
            logger.error(f"Error parsing metrics for node {node.id}: {str(e)}")
            node.metric = WorstMetricValue()
            node.is_buggy = True
            node.parse_exc_type = str(e)
            node.parse_exc_info = None
            node.parse_exc_stack = None
            node.parse_term_out = (
                "Error parsing metrics. There was an error in the parsing code: " + str(e)
            )

    def _generate_parse_code(self, node: Node) -> tuple[str, str]:
        """Step 1: Generate metrics parsing code using LLM"""
        parse_metrics_prompt = {
            "Introduction": (
                "You are an AI researcher analyzing experimental results stored in data files. "
                "Write code to load and analyze the metrics from data files in the working directory."
            ),
            "Context": [
                "Original Code: " + node.code,
            ],
            "Instructions": [
                "0. Make sure to get the working directory from os.path.join(os.getcwd(), 'working')",
                "1. Find and load data files (.npy, .npz, or .csv) in the working directory",
                "2. Extract metrics for each dataset/scenario. Make sure to refer to the original code to understand the structure of the data.",
                "3. For .npz files containing nested dictionaries, iterate over the dictionary keys (scenarios) rather than assuming the filename is the scenario name",
                "4. Always print the name of the dataset/scenario before printing the metrics",
                "5. Always print the name of the metric before printing the value by specifying the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
                "6. You only need to print the best or final value for each metric for each dataset",
                "7. DO NOT CREATE ANY PLOTS",
                "Important code structure requirements:",
                "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block. Do not use 'if __name__ == \"__main__\":' at all.",
                "  - All code should be at the global scope or in functions that are called from the global scope",
                "  - The script should execute immediately when run, without requiring any special entry point",
            ],
            "experiment_data structure": [
                "The experiment_data.npz file contains a 'scenarios' dict with the following structure:",
                "",
                "experiment_data = {",
                "    'scenarios': {",
                "        'CONDITION_A': {",
                "            'messages': [...],     # Message logs",
                "            'events': [...],       # Important events",
                "            'metrics': {...},      # Metrics for this scenario",
                "            'config': {...},       # Configuration",
                "        },",
                "        'CONDITION_B': { ... },",
                "        # ... other conditions",
                "    },",
                "    'metrics': {...},              # Aggregated metrics (optional)",
                "}",
                "",
                "For single-condition experiments, the scenario name may be 'default'.",
            ],
            "Example data loading code": [
                """
import os
import numpy as np

working_dir = os.path.join(os.getcwd(), 'working')

# Load experiment_data.npz
npz_path = os.path.join(working_dir, 'experiment_data.npz')
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    exp_data = data['experiment_data'].item()  # Extract dict from 0-d array

    # Check if 'scenarios' key exists (new format)
    if 'scenarios' in exp_data:
        for scenario_name, scenario_data in exp_data['scenarios'].items():
            print(f"Scenario: {scenario_name}")
            if 'metrics' in scenario_data:
                metrics = scenario_data['metrics']
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value}")

    # Also check top-level metrics if present
    if 'metrics' in exp_data:
        print("Aggregated metrics:")
        for metric_name, metric_value in exp_data['metrics'].items():
            print(f"{metric_name}: {metric_value}")
"""
            ],
            "Response format": self._prompt_metricparse_resp_fmt(),
        }

        # Use plan_and_code_query equivalent
        completion_text = query(
            system_message=parse_metrics_prompt,
            user_message=None,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.temp,
        )

        # Extract code
        from .utils.response import extract_code, extract_text_up_to_code

        code = extract_code(completion_text)
        plan = extract_text_up_to_code(completion_text)

        logger.debug(f"Parse metrics plan: {plan}")
        logger.debug(f"Parse metrics code: {code}")

        return code, plan

    def _structure_metrics(self, node: Node, metrics_exec_result: ExecutionResult):
        """Step 3: Structure metrics using LLM function calling"""
        metrics_prompt = {
            "Introduction": "Parse the metrics from the execution output. You only need the final or best value of a metric for each dataset, not the entire list during training.",
            "Execution Output": metrics_exec_result.term_out,
        }

        logger.debug(f"Metrics_exec_result.term_out: {metrics_exec_result.term_out}")
        logger.debug(f"Metrics Parsing Execution Result: {metrics_exec_result}")

        metrics_response = cast(
            dict,
            query(
                system_message=metrics_prompt,
                user_message=None,
                func_spec=metric_parse_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )

        logger.debug(f"Metrics: {metrics_response}")

        if metrics_response["valid_metrics_received"]:
            node.metric = MetricValue(
                value={"metric_names": metrics_response["metric_names"]}
            )
            logger.info(f"Successfully extracted metrics for node {node.id}")
        else:
            # Failure Pattern 2: LLM cannot parse
            node.metric = WorstMetricValue()
            node.is_buggy = True
            logger.error(f"No valid metrics received for node {node.id}")

    def _prompt_metricparse_resp_fmt(self):
        """Response format for metrics parsing"""
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. "
            )
        }
