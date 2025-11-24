"""
Result Evaluator for Phase 3
Evaluates execution results and determines if the code is buggy
"""
import logging
from typing import cast, Any

from ..core.node import Node
from ..core.execution_result import ExecutionResult
from ..llm.backend import query
from ..llm.backend.utils import wrap_code
from ..llm.function_specs import review_func_spec

logger = logging.getLogger(__name__)


class ResultEvaluator:
    """Evaluate execution results using LLM to determine if code is buggy"""

    def __init__(self, task_desc: str, cfg: Any):
        self.task_desc = task_desc
        self.cfg = cfg

    def evaluate(self, node: Node, exec_result: ExecutionResult, workspace: str = None):
        """
        Parse execution result and determine if the node is buggy
        Equivalent to parse_exec_result() from parallel_agent.py
        """
        logger.info(f"Evaluator is parsing execution results for node {node.id}")

        # Absorb execution result into node
        node.absorb_exec_result(exec_result)

        # Build prompt for LLM
        prompt = {
            "Introduction": (
                "You are an experienced AI researcher. "
                "You have written code for your research experiment and now need to evaluate the output of the code execution. "
                "Analyze the execution output, determine if there were any bugs, and provide a summary of the findings. "
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        # Query LLM with function calling
        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )

        # Update node with evaluation results
        node.analysis = response["summary"]
        node.is_buggy = response["is_bug"] or node.exc_type is not None

        print(
            "[red]Checking if response contains metric name and description[/red]",
            flush=True,
        )
        print(response)

        return node
