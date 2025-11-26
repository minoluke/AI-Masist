import os
import re
import random
import time
import pandas as pd
from dataclasses import dataclass, field
import numpy as np

import autogen
from autogen import ConversableAgent, LLMConfig

# =========================
# 作業ディレクトリ
# =========================
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# =========================
# 環境クラス（AutoGen とは独立）
# =========================
class TokenGameEnvironment:
    def __init__(self, threshold, group_rule, num_agents=4, token_limit=10, total_rounds=20):
        self.threshold = threshold
        self.group_rule = group_rule
        self.num_agents = num_agents
        self.token_limit = token_limit
        self.total_rounds = total_rounds
        self.current_round = 0

    def reset(self):
        self.current_round = 0
        return self.total_rounds, self.token_limit

    def step(self, actions):
        """
        actions: list[int] 各エージェントの拠出
        戻り値: (achieved: bool, total_contribution: int, results: list[int])
        """
        contributions = actions
        total_contribution = sum(contributions)
        achieved = total_contribution >= self.threshold

        # シンプルな効用関数: 残りトークン + (閾値達成ボーナス 1)
        results = [(self.token_limit - c) + (1 if achieved else 0) for c in contributions]
        return achieved, total_contribution, results


# =========================
# LLM エージェント → 拠出額を決めるラッパ
# =========================
@dataclass
class LLMTockenAgentWrapper:
    """AG2 の ConversableAgent を包んで、整数の拠出額だけ取ってくる薄いラッパ。"""
    agent: ConversableAgent
    agent_id: int
    total_initial_tokens: int = 10
    remaining_token: int = 10
    history: list = field(default_factory=list)  # (round, contribution, total_contribution, achieved, remaining_token)

    def decide_contribution(
        self,
        round_num: int,
        rule_active: bool,
        group_rule,
        threshold: int,
        env_rounds: int,
        user_proxy: ConversableAgent,
    ) -> int:
        """
        AG2 を使って LLM に「このラウンドでいくら出すか」を考えさせる。
        """

        # 過去履歴をテキスト化（トークン節約のため直近のみ）
        if self.history:
            last_h = self.history[-1]
            history_text = (
                f"Last round: round={last_h[0]}, you contributed={last_h[1]}, "
                f"total_contribution={last_h[2]}, achieved={last_h[3]}, "
                f"remaining_token={last_h[4]}."
            )
        else:
            history_text = "No previous rounds. This is your first decision."

        # ルールの説明
        if rule_active:
            rule_text = (
                f"A group rule is now in effect. "
                f"The target contribution vector is {group_rule} "
                f"(this means agent i is expected to contribute group_rule[i])."
            )
        else:
            rule_text = "There is no explicit rule yet. You can decide freely."

        # このエージェントに与えるプロンプト
        prompt = f"""
You are Agent {self.agent_id} in a public-goods token game.

- There are {env_rounds} total rounds. This is round {round_num}.
- Threshold for group success each round: {threshold}.
- You start with {self.total_initial_tokens} tokens and cannot exceed your remaining tokens.
- Your current remaining tokens: {self.remaining_token}.
- Other agents' target rule vector (if active): {group_rule}.
- {rule_text}

History:
{history_text}

Game mechanics:
- In each round, you choose a non-negative integer contribution NOT exceeding your remaining tokens.
- The group succeeds in the round if the sum of all contributions >= threshold.
- Your payoff per round is: (10 - contribution) + (1 if group succeeds else 0).

Your task:
- Decide your contribution for THIS round.
- Respond with ONLY a single integer number (no explanation, no text around it).
- The integer must be between 0 and {self.remaining_token}, inclusive.
"""

        # user_proxy → agent への 2 エージェントチャットを 1 ターンだけ走らせる
        chat_result = user_proxy.initiate_chat(
            recipient=self.agent,
            message=prompt,
            max_turns=1,
            silent=True,  # ログをうるさくしないため
        )

        # assistant（self.agent）の最後のメッセージから中身を取り出す
        if chat_result.chat_history:
            last_msg = chat_result.chat_history[-1]
            content = last_msg.get("content", "")
        else:
            content = ""

        # 数字だけ抽出
        match = re.search(r"-?\d+", str(content))
        if match:
            contribution = int(match.group())
        else:
            # うまくパースできなかったら 0〜remaining_token のランダムにフォールバック
            contribution = random.randint(0, self.remaining_token)

        # 範囲クリップ
        contribution = max(0, min(self.remaining_token, contribution))

        return contribution


# =========================
# AG2 / LLM 設定
# =========================
llm_config = LLMConfig(
    model="gpt-4o-mini",  # 必要に応じて変更
    api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.3,
)

# 各プレイヤー用エージェント（LLM）
assistant_agents = [
    ConversableAgent(
        name=f"agent_{i}",
        system_message=(
            "You are a rational but slightly cooperative agent in a repeated public-goods token game. "
            "Follow the instructions carefully and output only an integer contribution each round."
        ),
        llm_config=llm_config,
    )
    for i in range(4)
]

# 「人間側」役のエージェント（実際には LLM は使わない）
user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,          # LLM 呼び出しをしない
    human_input_mode="NEVER",  # 完全自動
)


# =========================
# メトリクス計算
# =========================
def compute_run_metrics(df: pd.DataFrame) -> dict:
    """1つの run（1シード分）の DataFrame から簡単なメトリクスを集計。"""
    # 1ラウンドにつき agent 0 の achieved を代表値とする
    df_round = df[df["agent_id"] == 0]

    success_rate = df_round["achieved"].mean()
    avg_total_contrib = df_round["total_contribution"].mean()

    return {
        "success_rate": float(success_rate),
        "avg_total_contribution": float(avg_total_contrib),
    }


# =========================
# シミュレーション本体（1シナリオ×1シード）
# =========================
def run_simulation(scenario_name, threshold, group_rule, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    env = TokenGameEnvironment(threshold, group_rule, num_agents=4)
    env.reset()

    # AutoGen エージェントを包むラッパを用意
    wrappers = [
        LLMTockenAgentWrapper(agent=assistant_agents[i], agent_id=i, total_initial_tokens=10, remaining_token=10)
        for i in range(4)
    ]

    log_data = []

    for round_num in range(1, env.total_rounds + 1):
        rule_active = round_num > 10  # 11ラウンド目以降でルール適用

        # 各エージェントに拠出額を決めさせる
        contributions = []
        for w in wrappers:
            if w.remaining_token <= 0:
                contribution = 0
            else:
                contribution = w.decide_contribution(
                    round_num=round_num,
                    rule_active=rule_active,
                    group_rule=group_rule,
                    threshold=env.threshold,
                    env_rounds=env.total_rounds,
                    user_proxy=user_proxy,
                )
            contributions.append(contribution)

        # 環境を1ステップ進める
        achieved, total_contribution, results = env.step(contributions)

        # トークン残量と履歴を更新 & ログ出力用に保存
        for i, w in enumerate(wrappers):
            w.remaining_token -= contributions[i]
            w.history.append(
                (
                    round_num,
                    contributions[i],
                    total_contribution,
                    achieved,
                    w.remaining_token,
                )
            )

            log_data.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    "round": round_num,
                    "agent_id": i,
                    "contribution": contributions[i],
                    "total_contribution": total_contribution,
                    "achieved": achieved,
                    "score": results[i],
                    "remaining_token": w.remaining_token,
                    "rule_followed_phase": rule_active,
                }
            )

    df = pd.DataFrame(log_data)

    # ---------- (1) npy保存：シナリオ×シードごとの生ログ ----------
    output_file = os.path.join(working_dir, f"{scenario_name}_seed{seed}.npy")
    np.save(output_file, df.to_numpy())
    print(f"{scenario_name}: wrote results to {output_file}")

    # ---------- (2) experiment_data 用の dict を返す ----------
    return {
        "seed": seed,
        "messages": log_data,          # DataFrame の行データ相当（dict のリスト）
        "metrics": compute_run_metrics(df),
    }


# =========================
# experiment_data 全体の保存
# =========================
def save_experiment_data(experiment_data, working_dir):
    """
    experiment_data 全体を np.savez_compressed() で保存。
    - ファイル名にタイムスタンプを含める
    - 保存先は working_dir
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_data_{timestamp}.npz"
    path = os.path.join(working_dir, filename)

    # 辞書をそのまま保存するため、object 配列に包んで保存
    np.savez_compressed(path, experiment_data=np.array(experiment_data, dtype=object))

    print(f"[Saved] experiment_data → {path}")
    return path


# =========================
# 全シナリオ実行 → experiment_data を構築して保存
# =========================
def run_all_experiments():
    experiment_data = {}

    # シナリオ定義: {シナリオ名: (threshold, group_rule)}
    scenarios = {
        "FAIRSUFF": (20, [5, 5, 5, 5]),
        "FAIRINF":  (20, [5, 5, 6, 6]),
    }

    # 必要に応じて複数 seed に拡張可能
    seeds = [0]

    for scenario_name, (threshold, rule) in scenarios.items():
        runs = []
        for sd in seeds:
            run_result = run_simulation(scenario_name, threshold, rule, seed=sd)
            runs.append(run_result)

        # シナリオ全体の aggregated_metrics（例：平均）
        agg_success = np.mean([r["metrics"]["success_rate"] for r in runs])
        agg_contrib = np.mean([r["metrics"]["avg_total_contribution"] for r in runs])

        experiment_data[scenario_name] = {
            "runs": runs,
            "aggregated_metrics": {
                "mean_success_rate": float(agg_success),
                "mean_avg_total_contribution": float(agg_contrib),
            },
        }

    # experiment_data 全体を 1 ファイルに保存
    save_experiment_data(experiment_data, working_dir)

    return experiment_data



experiment_data = run_all_experiments()
