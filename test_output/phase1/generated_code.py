import os
import logging
import numpy as np
from typing import List
from autogen import ConversableAgent, GroupChat, GroupChatManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM設定（環境変数から）
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
        }
    ],
    "temperature": 0.7,
    "timeout": 60,
}

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# エージェントの作成
def create_agents(num_agents: int) -> List[ConversableAgent]:
    agents = []
    for i in range(num_agents):
        agent = ConversableAgent(
            name=f"Agent_{i}",
            system_message="あなたはしきい値公共財ゲームのプレイヤーです。自分の持つトークンから出したい額を決めてください。",
            description="TPGGのプレイヤー",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agents.append(agent)
    return agents


# メトリクスの計算
def compute_metrics(
    total_contribution: int, threshold: int, contributions: List[int]
) -> dict:
    achieved = total_contribution >= threshold
    return {
        "threshold_achievement_rate": int(achieved),
        "average_contribution": total_contribution / len(contributions),
        "excess_contribution": max(0, total_contribution - threshold),
        "rule_compliance_rate": sum(
            1
            for i, c in enumerate(contributions)
            if c == (threshold // len(contributions))
        )
        / len(contributions),
    }


# シミュレーションの実行
def run_simulation(condition: dict):
    agents = create_agents(4)
    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    contributions_log = []
    for round_number in range(20):
        if round_number < 10:
            contributions = [0] * 4
        else:
            contributions = [condition["R"][i] for i in range(4)]

        chat_result = group_chat.initiate_chat(
            recipient=manager, message=f"ラウンド{round_number + 1}を開始します。"
        )

        total_contribution = sum(contributions)
        contributions_log.append(contributions)
        metrics = compute_metrics(total_contribution, condition["T"], contributions)
        logger.info(f"ラウンド{round_number + 1}: {metrics}")

    return {"contributions": contributions_log, "metrics": metrics, "config": condition}


# 実験設定の定義
experiment_conditions = [
    {"name": "FAIRSUFF", "T": 20, "R": [5, 5, 5, 5]},
    {"name": "FAIRINF", "T": 20, "R": [5, 5, 6, 6]},
    {"name": "UNFAIRSUFF", "T": 22, "R": [5, 5, 6, 6]},
    {"name": "UNFAIRINF", "T": 22, "R": [6, 6, 6, 6]},
    {"name": "CONTROL", "T": 22, "R": None},
]

# 主要メトリクスの保存
experiment_data = {"scenarios": {}}
for condition in experiment_conditions:
    result = run_simulation(condition)
    experiment_data["scenarios"][condition["name"]] = result

# 結果の保存
np.savez_compressed(
    f"{working_dir}/experiment_data.npz", experiment_data=experiment_data
)
print(f"Simulation completed: metrics = {experiment_data}")
