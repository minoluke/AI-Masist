"""
Result Evaluator for Phase 3
Evaluates execution results and determines if the code is buggy
"""
import logging
import os
from typing import cast, Any

import numpy as np
import pandas as pd

from .journal import Node
from .utils.execution_result import ExecutionResult
from .backend import query
from .backend.utils import wrap_code
from .utils.function_specs import review_func_spec

logger = logging.getLogger(__name__)


class ResultEvaluator:
    """Evaluate execution results using LLM to determine if code is buggy"""

    def __init__(self, task_desc: str, cfg: Any):
        self.task_desc = task_desc
        self.cfg = cfg
        self.ag2_reference = self._load_ag2_reference()

    def _load_ag2_reference(self) -> str:
        """Load AG2 quick reference document"""
        doc_path = os.path.join(os.path.dirname(__file__), "..", "docs", "AG2_QUICK_REFERENCE.md")
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"AG2 reference document not found at {doc_path}")
            return ""

    def evaluate(self, node: Node, exec_result: ExecutionResult, workspace: str = None):
        """
        Parse execution result and determine if the node is buggy
        Equivalent to parse_exec_result() from parallel_agent.py
        """
        logger.info(f"Evaluator is parsing execution results for node {node.id}")

        # Absorb execution result into node
        node.absorb_exec_result(exec_result)

        # Get list of generated output files
        output_files_info = "出力ファイル情報はありません。"
        if workspace and os.path.exists(workspace):
            files = [f for f in os.listdir(workspace) if os.path.isfile(os.path.join(workspace, f))]
            if files:
                file_info_list = []
                for f in files:
                    file_path = os.path.join(workspace, f)
                    file_size = os.path.getsize(file_path)

                    # Read first 10 lines/rows
                    preview = ""
                    try:
                        ext = os.path.splitext(f)[1].lower()
                        if ext == '.csv':
                            df = pd.read_csv(file_path)
                            preview = f"\n    先頭10行:\n{df.head(10).to_string(index=False)}"
                        elif ext == '.npy':
                            data = np.load(file_path, allow_pickle=True)
                            if data.ndim == 2:
                                preview = f"\n    先頭10行:\n{str(data[:10])}"
                            else:
                                preview = f"\n    データ:\n{str(data[:10]) if len(data) > 10 else str(data)}"
                        elif ext in ['.txt', '.log']:
                            with open(file_path, 'r') as file:
                                lines = file.readlines()[:10]
                                preview = f"\n    先頭10行:\n{''.join(lines)}"
                    except Exception as e:
                        preview = f"\n    (読み込みエラー: {str(e)})"

                    file_info_list.append(f"  - {f} ({file_size} bytes){preview}")
                output_files_info = "生成されたファイル一覧:\n" + "\n".join(file_info_list)

        # Build prompt for LLM
        prompt = {
            "Introduction": (
                "あなたは実験コードの実行結果を評価する熟練したAI研究者です。"
                "以下の2つを分析して、実行が成功したかどうかを判断してください："
                " (1) 実行時の出力・ログ、(2) 生成された出力ファイル。"
                "\n\n"
                "【評価基準】\n"
                "- SUCCESS（成功）:\n"
                "  出力ファイルが生成されており、研究目的に合致した有効で空でないデータが含まれている場合。\n"
                "\n"
                "- NORMAL（正常）:\n"
                "  'terminated'、'completed'や、反復上限到達など、"
                "  フレームワーク特有のログメッセージは一般的に正常動作であり、バグとは見なさない。\n"
                "\n"
                "- BUG（バグ）:\n"
                "  Python 例外が発生した場合、または期待される出力ファイルが存在しない／空／破損している場合。\n"
                "\n"
                "- 判断においては、ログメッセージよりも「出力ファイルの内容」を優先すること。\n"
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
            "Output files": output_files_info,
        }

        # AG2リファレンスドキュメントを追加
        if self.ag2_reference:
            prompt["AG2 API Reference"] = self.ag2_reference

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
