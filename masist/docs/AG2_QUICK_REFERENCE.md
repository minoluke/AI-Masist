# AG2 クイックリファレンス（コード生成用）

このドキュメントはLLMがAG2コードを生成する際に参照するためのものです。

---

## 基本インポートとセットアップ

```python
import os
import json
import logging
from typing import Annotated, Dict, List, Any
from datetime import datetime
import numpy as np

from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
    initiate_chats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM設定（環境変数から）
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
```

---

## エージェント作成

### ConversableAgent（基本）
```python
agent = ConversableAgent(
    name="agent_name",                 # 必須: 一意の名前
    system_message="ロールの説明...",    # 役割・振る舞いの指示
    description="簡潔な説明",            # GroupChatでの選択用
    llm_config=llm_config,
    human_input_mode="NEVER",          # シミュレーションでは "NEVER"
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "終了" in x.get("content", ""),
)
```

### 複数エージェント生成
```python
def create_agents(roles: List[dict]) -> List[ConversableAgent]:
    agents = []
    for i, role in enumerate(roles):
        agent = ConversableAgent(
            name=role["name"],
            system_message=role["system_message"],
            description=role.get("description", ""),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agents.append(agent)
    return agents
```

---

## 対話パターン

### 1. Two-Agent Chat（2者対話）
```python
chat_result = agent_a.initiate_chat(
    recipient=agent_b,
    message="開始メッセージ",
    max_turns=5,
    clear_history=True,
    silent=False,
)
# 結果取得
history = chat_result.chat_history  # List[dict]
summary = chat_result.summary       # str
```

### 2. GroupChat（グループ対話）
```python
group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",  # "auto", "round_robin", "random"
    allow_repeat_speaker=True,
    send_introductions=True,
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: "終了" in x.get("content", ""),
)

chat_result = agent1.initiate_chat(
    recipient=manager,
    message="議論開始",
)
# メッセージ取得
messages = group_chat.messages  # List[dict with 'name', 'content']
```

### 3. Sequential Chat（連続対話）
```python
from autogen import initiate_chats

chat_queue = [
    {"sender": a1, "recipient": a2, "message": "msg1", "max_turns": 3},
    {"sender": a2, "recipient": a3, "message": "msg2", "max_turns": 2},
]
results = initiate_chats(chat_queue)
```

### カスタム発言者選択
```python
def custom_speaker_selection(last_speaker, group_chat):
    import random
    available = [a for a in group_chat.agents if a != last_speaker]
    return random.choice(available)

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method=custom_speaker_selection,
)
```

---

## 終了条件

```python
# メッセージ内容で終了
is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper()

# ターン数で終了
max_turns=10  # initiate_chat
max_round=20  # GroupChat

# 連続返信数で終了
max_consecutive_auto_reply=5
```

---

## ツール（関数）登録

```python
from typing import Annotated
from autogen import register_function

def my_tool(
    param1: Annotated[str, "パラメータ1の説明"],
    param2: Annotated[int, "パラメータ2の説明"]
) -> str:
    """ツールの説明"""
    return f"結果: {param1}, {param2}"

# ツール登録
register_function(
    my_tool,
    caller=caller_agent,    # ツールを呼ぶエージェント
    executor=executor_agent, # ツールを実行するエージェント
    description="ツールの説明",
)
```

---

## ログ収集パターン

### 基本ログクラス
```python
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json

@dataclass
class RunResult:
    seed: int
    messages: List[dict]
    metrics: dict
    termination_reason: str = ""

class SimulationLogger:
    def __init__(self):
        self.runs: List[RunResult] = []

    def log_run(self, seed: int, messages: List[dict], metrics: dict, reason: str = ""):
        self.runs.append(RunResult(seed, messages, metrics, reason))

    def to_dict(self) -> dict:
        return {"runs": [asdict(r) for r in self.runs]}

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
```

---

## 実験データ構造（MASist標準）

```python
experiment_data = {
    'scenario_name': {
        'runs': [
            {
                'seed': 0,
                'messages': [
                    {'round': 0, 'speaker': 'agent1', 'content': '...'},
                    ...
                ],
                'metrics': {
                    'metric1': value,
                    'metric2': value,
                },
            },
            # 他のseed...
        ],
        'aggregated_metrics': {
            'metric1_mean': ...,
            'metric1_std': ...,
        },
    },
    # 他のシナリオ...
}
```

---

## 典型的な実行フロー

```python
def run_simulation(scenario_config: dict, num_seeds: int = 3) -> dict:
    """シミュレーション実行"""
    results = {'runs': [], 'aggregated_metrics': {}}

    for seed in range(num_seeds):
        # エージェント作成
        agents = create_agents(scenario_config['roles'])

        # グループチャット設定
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=scenario_config.get('max_round', 20),
            speaker_selection_method=scenario_config.get('speaker_method', 'round_robin'),
        )
        manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

        # 実行
        chat_result = agents[0].initiate_chat(
            recipient=manager,
            message=scenario_config['initial_message'],
        )

        # メトリクス計算
        metrics = compute_metrics(group_chat.messages, scenario_config)

        # 結果保存
        results['runs'].append({
            'seed': seed,
            'messages': group_chat.messages,
            'metrics': metrics,
        })

        print(f"Run {seed} (seed={seed}): metrics = {metrics}")

    # 集約
    results['aggregated_metrics'] = aggregate_metrics(results['runs'])
    return results

def compute_metrics(messages: List[dict], config: dict) -> dict:
    """メトリクス計算"""
    metrics = {}
    # 例: 発言回数
    for agent_name in [m.get('name') for m in messages if m.get('name')]:
        count_key = f'{agent_name}_messages'
        metrics[count_key] = metrics.get(count_key, 0) + 1
    metrics['total_messages'] = len(messages)
    return metrics

def aggregate_metrics(runs: List[dict]) -> dict:
    """メトリクス集約"""
    import numpy as np
    all_keys = set()
    for run in runs:
        all_keys.update(run['metrics'].keys())

    aggregated = {}
    for key in all_keys:
        values = [run['metrics'].get(key, 0) for run in runs]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    return aggregated
```

---

## 保存と出力

```python
# NumPy保存（MASistではこの形式を使用）
np.savez_compressed(
    f"{working_dir}/experiment_data_{timestamp}.npz",
    experiment_data=experiment_data
)

# 結果出力
for scenario, data in experiment_data.items():
    print(f"\n=== {scenario} ===")
    print(f"Aggregated: {data['aggregated_metrics']}")
```

---

## 注意事項

1. **human_input_mode**: シミュレーションでは必ず `"NEVER"` を設定
2. **終了条件**: 無限ループ防止のため `max_round` または `max_turns` を必ず設定
3. **エージェント名**: GroupChat内で一意である必要がある
4. **LLM設定**: 環境変数 `OPENAI_API_KEY` が設定されている前提
5. **ログ**: 全メッセージを `group_chat.messages` または `chat_result.chat_history` から取得
