import os
import numpy as np
from autogen import (
    ConversableAgent,
    GroupChat,
    GroupChatManager,
)

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# LLM設定
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.7,
    "timeout": 120,
}

scenarios = {
    "FAIRSUFF": {"T": 20, "rules": (5, 5, 5, 5)},
    "FAIRINF": {"T": 20, "rules": (5, 5, 6, 6)},
    "UNFAIRSUFF": {"T": 22, "rules": (5, 5, 6, 6)},
    "UNFAIRINF": {"T": 22, "rules": (6, 6, 6, 6)},
    "CONTROL": {"T": 22, "rules": ()},
}


def create_agents():
    return [
        ConversableAgent(
            name=f"Agent{i+1}",
            system_message="トークンを選択してください。",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        for i in range(4)
    ]


def run_simulation(scenario_name):
    scenario = scenarios[scenario_name]
    successes = []
    total_messages = []
    for seed in range(3):  # 3回試行
        np.random.seed(seed)
        agents = create_agents()
        group_chat = GroupChat(agents=agents, messages=[], max_round=20)
        manager = GroupChatManager(group_chat, llm_config)

        round_results = []

        for round_number in range(20):
            # 各エージェントがトークンを決定
            contributions = [
                agent.initiate_chat(
                    recipient=manager,
                    message=f"私は{np.random.randint(0, 11)}トークンを出します。",
                )
                for agent in agents
            ]
            total_contrib = sum(
                int(msg["content"].split(" ")[1]) for msg in contributions
            )
            success = total_contrib >= scenario["T"]
            round_results.append(
                (
                    round_number,
                    [int(msg["content"].split(" ")[1]) for msg in contributions],
                    total_contrib,
                    success,
                )
            )

        total_messages.extend(round_results)

        print(
            f"Run {seed} (scenario: {scenario_name}): total_contributions = {[result[2] for result in round_results]}"
        )

    # 結果保存
    np.savez_compressed(
        os.path.join(working_dir, f"experiment_data_{scenario_name}.npz"),
        total_messages=total_messages,
    )


# 各シナリオを実行
for scenario_name in scenarios.keys():
    run_simulation(scenario_name)
