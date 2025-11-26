"""
Code Generator for Draft Phase
Extracts and adapts code from parallel_agent.py MinimalAgent class
"""
import os
import random
from typing import Any, Tuple
import humanize

from ..core.node import Node
from ..llm.backend import query
from ..utils.response import extract_code, extract_text_up_to_code, wrap_code


class CodeGenerator:
    """Generate Python code from research ideas using LLM"""

    def __init__(self, task_desc: str, evaluation_metrics: list, cfg: Any, memory_summary: str = None):
        self.task_desc = task_desc
        self.evaluation_metrics = evaluation_metrics
        self.cfg = cfg
        self.memory_summary = memory_summary
        self.data_preview = None
        self.ag2_reference = self._load_ag2_reference()

    def _load_ag2_reference(self) -> str:
        """Load AG2 quick reference document"""
        doc_path = os.path.join(os.path.dirname(__file__), "..", "docs", "AG2_QUICK_REFERENCE.md")
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: AG2 reference document not found at {doc_path}")
            return ""

    def generate(self) -> Node:
        """Generate a draft implementation (equivalent to _draft())"""
        prompt: Any = {
            "Introduction": (
                "あなたは MASist（社会科学系・LLMマルチエージェント実験＋論文自動生成プラットフォーム）の"
                "シミュレーションエンジン開発を担うAI研究者です。"
                "最初のタスクは、以下の Research idea（シミュレーション検討シート）に基づき、"
                "Autogen を利用した LLM マルチエージェントシミュレーションを設計・実装し、"
                "複数シナリオの実行・ログ収集・評価・簡易可視化まで行う堅牢なベースラインを構築することです。"
                "まずは高度な最適化よりも、正しく動作する最小限のパイプライン構築を優先してください。"
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "これは初期ベースライン設計であり、複雑なハイパーパラメータ調整は行わないこと。",
                "Memory セクションの情報と一貫する設計にすること。",
                "シミュレーション検討シート（Research idea）の内容を忠実に反映すること。",
                "EDA（データ探索）は提案しないこと。",
                "必要に応じて複数の合成シナリオ（異なる環境設定や役割構造）を作成してよい。",
                "マルチエージェントフレームワークは必ず Autogen を使用すること。",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # AG2リファレンスドキュメントを追加
        if self.ag2_reference:
            prompt["AG2 API Reference"] = self.ag2_reference

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        print("CodeGenerator: Getting plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("CodeGenerator: Draft complete")
        return Node(plan=plan, code=code)

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "autogen",
            "matplotlib",
            "seaborn",
            "scikit-learn",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": (
                f"以下パッケージ利用可：{pkg_str}。"
                "マルチエージェントフレームワークは autogen を必ず使用すること。"
                "その他、可視化・データ処理用ライブラリは必要に応じて利用可。"
            )
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        # デフォルト値を設定
        num_syn_datasets = 2
        if hasattr(self.cfg, "experiment") and hasattr(self.cfg.experiment, "num_syn_datasets"):
            num_syn_datasets = self.cfg.experiment.num_syn_datasets

        impl_guideline = [
            "【MASist シミュレーション要件】",
            "  - Autogen によるエージェント定義・グループ対話・停止条件を用いた"
            "    マルチエージェントシミュレーションパイプラインを構築すること。",
            "  - エージェントのロール、内部状態、環境状態、終了条件を明示すること。",
            "  - 同じシナリオを複数回試行（異なる乱数シード）できる構造にすること。",
            f"  - 動作確認用に、複数のシナリオ（最低 {num_syn_datasets} 件）で実行し、"
            "    シナリオ間比較を行うこと。",
            "",
            "【プロンプト/入力設計】",
            "  - 検討シートの項目（背景、目的、研究質問、仮説、エージェント、環境、プロトコルなど）を"
            "    そのまま LLM 入力として使える構造に整形すること。",
            "",
            "【experiment_data の構造（修正版：教師あり学習前提を撤廃）】",
            "  experiment_data = {",
            "      'scenario_name': {",
            "          'runs': [",
            "              {",
            "                  'seed': 0,",
            "                  'messages': [...],",
            "                  'metrics': {...},",
            "              },",
            "          ],",
            "          'aggregated_metrics': {...},",
            "      },",
            "  }",
            "",
            "【シミュレーションの記録】",
            "  - 各 run（試行）で記録すべき情報：",
            "        ・全メッセージログ（ターン順）",
            "        ・途中の重要イベント（例：合意成立、衝突）",
            "        ・run の評価指標（self.evaluation_metrics）",
            "  - scenario ごとに aggregated_metrics を計算すること（平均・分散など）。",
            "",
            "【評価要件（train/val/loss を撤廃し MAS 向けに修正済み）】",
            "  - 各 run の終了後に主要メトリクスを print：",
            "       print(f'Run {run_id} (seed={seed}): metrics = {metrics}')",
            "  - 各 scenario 終了後に aggregated_metrics を print。",
            "  - ALL metrics を追跡し、実行後に保存する。",
            "",
            "【保存要件（ログ・メトリクス）※必須※】",
            "  - **重要**: 必ず np.savez_compressed() で experiment_data 全体を保存すること。",
            "  - **JSON形式での保存は禁止**。必ず .npz 形式を使用すること。",
            "  - **ファイル名は必ず 'experiment_data.npz' とすること（タイムスタンプやシナリオ名を含めない）**",
            "  - **シナリオごとに別ファイルを作成しないこと。全シナリオを1つの辞書にまとめて保存する**",
            "  - 保存例:",
            "       np.savez_compressed(f'{working_dir}/experiment_data.npz', experiment_data=np.array(experiment_data, dtype=object))",
            "  - 保存先は working_dir とする。",
            "",
            "【コード構造要件】",
            "  - コードは以下の3行から始めること：",
            "       import os",
            "       working_dir = os.path.join(os.getcwd(), 'working')",
            "       os.makedirs(working_dir, exist_ok=True)",
            "  - **重要**: `if __name__ == \"__main__\":` ブロックは絶対に使用しないこと。",
            "  - すべての実行コードはグローバルスコープに直接記述すること（関数定義の外）。",
            "  - 外部設定ファイルに依存せず、1ファイルで完結すること。",
            f"  - 実行時間は {humanize.naturaldelta(self.cfg.exec.timeout)} 内で収まるよう、"
            "    試行数・最大ターン数を適切に設定すること。",
            "",
            "【Autogen 依存について】",
            "  - Autogen および LLM API キーが環境に存在する前提で、コードは単一ファイルとして成立すべき。",
        ]

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "最初に、提案するシミュレーション設計・実装方針を7〜10文で簡潔に説明し、"
                "その後に Autogen ベースのシミュレーション実装を含む単一の Python コードブロック"
                "（ ```python ... ``` ）を提示してください。"
                "コードは一つのファイルとして完結し、実行可能であること。"
                "自然言語の説明 → 改行 → コードブロックの順で、余分な見出しは不要です。"
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
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    # ========== Week 2: Debug/Improve メソッド ==========

    def generate_debug(self, parent_node: Node) -> Node:
        """
        Generate a bugfix implementation based on parent node's error information.
        Equivalent to MinimalAgent._debug() in AI-Scientist-v2.
        """
        prompt: Any = {
            "Introduction": (
                "あなたは MASist（LLMマルチエージェント実験プラットフォーム）のシミュレーションエンジン開発を担うAI研究者です。"
                "前回のシミュレーションコードにバグがあったため、以下の情報をもとにバグを修正してください。"
                "応答は、まず自然言語で修正方針を簡潔に説明し、その後に修正済みの完全なコードを提示してください。"
            ),
            "Research idea": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out or "", lang=""),
            "Instructions": {},
        }

        # VLMフィードバックがあれば追加
        if parent_node.vlm_feedback_summary:
            prompt["Feedback based on generated plots"] = parent_node.vlm_feedback_summary

        # 実行時間フィードバックがあれば追加
        if hasattr(parent_node, 'exec_time_feedback') and parent_node.exec_time_feedback:
            prompt["Feedback about execution time"] = parent_node.exec_time_feedback

        prompt["Instructions"] |= self._prompt_debug_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix guideline": [
                "前回の実装のバグを特定し、修正方針を3〜5文で簡潔に説明すること。",
                "修正後のコードは完全で実行可能であること（省略不可）。",
                "EDA（データ探索）は提案しないこと。",
                "Autogen を使ったマルチエージェントシミュレーションの構造を維持すること。",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # AG2リファレンスドキュメントを追加
        if self.ag2_reference:
            prompt["AG2 API Reference"] = self.ag2_reference

        print("[yellow]--------------------------------[/yellow]")
        print(f"[yellow]CodeGenerator: Debugging node {parent_node.id}[/yellow]")
        print("[yellow]--------------------------------[/yellow]")

        print("CodeGenerator: Getting debug plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("CodeGenerator: Debug complete")

        return Node(plan=plan, code=code, parent=parent_node)

    def generate_improve(self, parent_node: Node) -> Node:
        """
        Generate an improved implementation based on parent node's results.
        Equivalent to MinimalAgent._improve() in AI-Scientist-v2.
        """
        prompt: Any = {
            "Introduction": (
                "あなたは MASist（LLMマルチエージェント実験プラットフォーム）のシミュレーションエンジン開発を担うAI研究者です。"
                "前回のシミュレーションは正常に動作しましたが、さらなる改善が可能です。"
                "以下の情報をもとに、シミュレーションの品質・精度・効率を向上させてください。"
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Previous solution": {
                "Code": wrap_code(parent_node.code),
                "Plan": parent_node.plan,
            },
            "Instructions": {},
        }

        # メトリクス情報があれば追加
        if parent_node.metric:
            prompt["Previous metrics"] = str(parent_node.metric)

        # VLMフィードバックがあれば追加
        if parent_node.vlm_feedback_summary:
            prompt["Feedback based on generated plots"] = parent_node.vlm_feedback_summary

        # 実行時間フィードバックがあれば追加
        if hasattr(parent_node, 'exec_time_feedback') and parent_node.exec_time_feedback:
            prompt["Feedback about execution time"] = parent_node.exec_time_feedback

        # 分析結果があれば追加
        if parent_node.analysis:
            prompt["Previous analysis"] = parent_node.analysis

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Improvement guideline": [
                "前回の実装を改善する方針を7〜10文で説明すること。",
                "改善の観点：メトリクスの向上、シミュレーションの安定性、コードの効率性など。",
                "改善後のコードは完全で実行可能であること（省略不可）。",
                "Autogen を使ったマルチエージェントシミュレーションの構造を維持すること。",
                "過度な複雑化は避け、段階的な改善を心がけること。",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # AG2リファレンスドキュメントを追加
        if self.ag2_reference:
            prompt["AG2 API Reference"] = self.ag2_reference

        print("[green]--------------------------------[/green]")
        print(f"[green]CodeGenerator: Improving node {parent_node.id}[/green]")
        print("[green]--------------------------------[/green]")

        print("CodeGenerator: Getting improvement plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("CodeGenerator: Improvement complete")

        return Node(plan=plan, code=code, parent=parent_node)

    @property
    def _prompt_debug_resp_fmt(self):
        """Response format for debug prompts"""
        return {
            "Response format": (
                "まず、前回の実装のバグと修正方針を3〜5文で簡潔に説明してください。"
                "その後に、修正済みの完全な Python コードブロック（ ```python ... ``` ）を提示してください。"
                "コードは完全で実行可能であること。省略は不可です。"
                "自然言語の説明 → 改行 → コードブロックの順で、余分な見出しは不要です。"
            )
        }
