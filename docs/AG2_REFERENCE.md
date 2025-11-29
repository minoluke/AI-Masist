# AG2 (AutoGen) リファレンスドキュメント
## 社会科学系マルチエージェントシミュレーション向け

このドキュメントは、シミュレーション検討シートの各項目をAG2で実装するための基本パターンを網羅しています。

---

## 目次

1. [基本セットアップ](#1-基本セットアップ)
2. [エージェント定義](#2-エージェント定義)
3. [対話パターン（プロトコル）](#3-対話パターンプロトコル)
4. [終了条件](#4-終了条件)
5. [ツールと環境](#5-ツールと環境)
6. [ログと結果取得](#6-ログと結果取得)
7. [状態管理](#7-状態管理)
8. [実装パターン集](#8-実装パターン集)

---

## 1. 基本セットアップ

### 1.1 インストールとインポート

```python
# インストール
# pip install ag2[openai]  # または pip install autogen[openai]

# 基本インポート
import os
import logging
from typing import Annotated, Dict, List, Any, Callable
from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
    initiate_chats,
    LLMConfig,
)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### 1.2 LLM設定

```python
# 方法1: 環境変数から直接設定
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    ],
    "temperature": 0.7,
    "timeout": 300,
}

# 方法2: 設定ファイルから読み込み（OAI_CONFIG_LIST）
# OAI_CONFIG_LIST ファイルの例:
# [
#     {
#         "model": "gpt-4o",
#         "api_key": "sk-..."
#     }
# ]
from autogen import LLMConfig
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# 方法3: 辞書形式で直接指定
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai",
        }
    ],
    "temperature": 0.5,
    "max_tokens": 2000,
}
```

---

## 2. エージェント定義

シミュレーション検討シートの「エージェント」セクションに対応

### 2.1 ConversableAgent（基本エージェント）

```python
agent = ConversableAgent(
    name="agent_name",                    # エージェント名（必須）
    system_message="あなたは...",          # ロールと振る舞いの定義
    description="このエージェントは...",   # GroupChatでの選択に使用される説明
    llm_config=llm_config,                # LLM設定
    human_input_mode="NEVER",             # "ALWAYS", "TERMINATE", "NEVER"
    max_consecutive_auto_reply=10,        # 連続自動返信の最大数
    is_termination_msg=lambda x: "終了" in x.get("content", ""),  # 終了条件
    default_auto_reply="続けてください",   # デフォルトの返信
    code_execution_config=False,          # コード実行設定
)
```

### 2.2 AssistantAgent（LLMアシスタント）

```python
assistant = AssistantAgent(
    name="assistant",
    system_message="あなたは有能なアシスタントです。",
    llm_config=llm_config,
    # AssistantAgentはデフォルトで human_input_mode="NEVER"
)
```

### 2.3 UserProxyAgent（ユーザー代理）

```python
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",             # シミュレーションでは通常 "NEVER"
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config=False,          # コード実行を無効化
    default_auto_reply="続けてください",
)
```

### 2.4 複数エージェントの定義例（ゲーム理論シミュレーション）

```python
def create_agents(llm_config: dict, scenario_config: dict) -> List[ConversableAgent]:
    """シナリオ設定に基づいてエージェントを生成"""
    agents = []
    
    for i, role in enumerate(scenario_config["roles"]):
        agent = ConversableAgent(
            name=f"player_{i}",
            system_message=f"""
あなたは{role['name']}です。
役割: {role['description']}
目標: {role['objective']}
制約: {role['constraints']}

以下のルールに従って行動してください：
1. 自分の利益を最大化する
2. 他のプレイヤーの行動を考慮する
3. 明確な意思決定を行い、その理由を説明する
""",
            description=role['description'],
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agents.append(agent)
    
    return agents
```

---

## 3. 対話パターン（プロトコル）

シミュレーション検討シートの「プロトコル」セクションに対応

### 3.1 Two-Agent Chat（2者間対話）

最もシンプルな対話パターン。

```python
# 基本的な2者間対話
chat_result = agent_a.initiate_chat(
    recipient=agent_b,
    message="議論を始めましょう",
    max_turns=5,                          # 最大ターン数
    clear_history=True,                   # 履歴をクリアするか
    silent=False,                         # 出力を表示するか
    summary_method="last_msg",            # 要約方法: "last_msg" or "reflection_with_llm"
)

# 結果の取得
print(f"チャット履歴: {chat_result.chat_history}")
print(f"要約: {chat_result.summary}")
print(f"コスト: {chat_result.cost}")
```

### 3.2 GroupChat（グループチャット）

複数エージェントが参加する対話。社会科学シミュレーションの主要パターン。

```python
# グループチャットの作成
group_chat = GroupChat(
    agents=[agent_a, agent_b, agent_c, moderator],
    messages=[],
    max_round=20,                         # 最大ラウンド数
    speaker_selection_method="auto",      # "auto", "round_robin", "random", "manual"
    allow_repeat_speaker=True,            # 同じ発言者の連続を許可
    send_introductions=True,              # 自己紹介を送信
)

# グループチャットマネージャーの作成
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: "合意に達しました" in x.get("content", ""),
)

# チャットの開始
chat_result = moderator.initiate_chat(
    recipient=manager,
    message="議論を開始します。テーマは...",
)
```

### 3.3 Speaker Selection Methods（発言者選択方法）

```python
# round_robin: 順番に発言
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin",
)

# random: ランダムに選択
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="random",
)

# auto: LLMが次の発言者を決定（デフォルト）
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="auto",
)

# カスタム選択関数
def custom_speaker_selection(
    last_speaker: ConversableAgent,
    group_chat: GroupChat
) -> ConversableAgent:
    """カスタムの発言者選択ロジック"""
    # 例: 前の発言者以外からランダムに選択
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

### 3.4 Sequential Chat（連続チャット）

複数の2者間チャットを順番に実行し、結果を引き継ぐ。

```python
from autogen import initiate_chats

# チャットキューの定義
chat_queue = [
    {
        "sender": researcher,
        "recipient": analyst,
        "message": "データを分析してください",
        "max_turns": 3,
        "summary_method": "reflection_with_llm",
    },
    {
        "sender": analyst,
        "recipient": writer,
        "message": "分析結果をレポートにまとめてください",
        "max_turns": 2,
        "carryover": "前のチャットの要約を参照してください",  # 引き継ぎ情報
    },
]

# 連続チャットの実行
results = initiate_chats(chat_queue)
```

### 3.5 Nested Chat（ネストチャット）

チャット内で別のチャットを起動する。

```python
# ネストチャットの登録
agent.register_nested_chats(
    chat_queue=[
        {
            "recipient": expert_agent,
            "message": "専門家の意見を求めます",
            "max_turns": 2,
            "summary_method": "last_msg",
        }
    ],
    trigger=lambda sender: "専門家に相談" in sender.last_message().get("content", ""),
)
```

---

## 4. 終了条件

シミュレーション検討シートの「終了条件」セクションに対応

### 4.1 メッセージ内容による終了

```python
# is_termination_msg を使用
agent = ConversableAgent(
    name="agent",
    system_message="...",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: any([
        "TERMINATE" in x.get("content", "").upper(),
        "合意" in x.get("content", ""),
        "終了" in x.get("content", ""),
    ]),
)
```

### 4.2 ターン数による終了

```python
# initiate_chat の max_turns
chat_result = agent_a.initiate_chat(
    recipient=agent_b,
    message="開始",
    max_turns=5,  # 5ターンで終了
)

# GroupChat の max_round
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20,  # 20ラウンドで終了
)
```

### 4.3 連続返信数による終了

```python
agent = ConversableAgent(
    name="agent",
    system_message="...",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,  # 3回連続返信で停止
)
```

### 4.4 カスタム終了条件

```python
class SimulationState:
    """シミュレーション状態を管理"""
    def __init__(self):
        self.round_count = 0
        self.consensus_reached = False
        self.max_rounds = 50
    
    def check_termination(self, msg: dict) -> bool:
        """終了条件をチェック"""
        content = msg.get("content", "")
        
        # ラウンド数チェック
        self.round_count += 1
        if self.round_count >= self.max_rounds:
            return True
        
        # 合意チェック
        if "全員合意" in content:
            self.consensus_reached = True
            return True
        
        return False

state = SimulationState()
agent = ConversableAgent(
    name="agent",
    system_message="...",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=state.check_termination,
)
```

---

## 5. ツールと環境

シミュレーション検討シートの「環境」セクションに対応

### 5.1 ツール（関数）の登録

```python
from typing import Annotated
from autogen import register_function

# ツール関数の定義
def get_market_price(
    item: Annotated[str, "商品名"],
    quantity: Annotated[int, "数量"]
) -> str:
    """市場価格を取得する"""
    # シミュレーション用のダミー実装
    prices = {"apple": 100, "banana": 80, "orange": 120}
    price = prices.get(item, 0) * quantity
    return f"{item}の{quantity}個の価格: {price}円"

def submit_bid(
    item: Annotated[str, "商品名"],
    price: Annotated[int, "入札価格"],
    agent_name: Annotated[str, "エージェント名"]
) -> str:
    """入札を提出する"""
    return f"{agent_name}が{item}に{price}円で入札しました"

# エージェントへのツール登録
register_function(
    get_market_price,
    caller=buyer_agent,      # ツールを呼び出すエージェント
    executor=executor_agent,  # ツールを実行するエージェント
    description="市場価格を取得する",
)

register_function(
    submit_bid,
    caller=buyer_agent,
    executor=executor_agent,
    description="入札を提出する",
)
```

### 5.2 function_map を使った登録

```python
# function_map を使用した一括登録
agent.register_function(
    function_map={
        "get_market_price": get_market_price,
        "submit_bid": submit_bid,
        "check_balance": check_balance,
    }
)
```

### 5.3 環境クラスの実装

```python
class SimulationEnvironment:
    """シミュレーション環境を管理するクラス"""
    
    def __init__(self, config: dict):
        self.config = config
        self.state = {
            "round": 0,
            "market_prices": {},
            "agent_balances": {},
            "transaction_history": [],
        }
    
    def get_state(self) -> dict:
        """現在の環境状態を取得"""
        return self.state.copy()
    
    def update_state(self, action: dict) -> dict:
        """アクションに基づいて状態を更新"""
        action_type = action.get("type")
        
        if action_type == "transaction":
            self._process_transaction(action)
        elif action_type == "market_update":
            self._update_market(action)
        
        return self.get_state()
    
    def _process_transaction(self, action: dict):
        """取引を処理"""
        self.state["transaction_history"].append({
            "round": self.state["round"],
            **action
        })
    
    def _update_market(self, action: dict):
        """市場を更新"""
        self.state["market_prices"].update(action.get("prices", {}))
    
    def next_round(self):
        """次のラウンドに進む"""
        self.state["round"] += 1

# 環境をツールとして公開
env = SimulationEnvironment(config={})

def env_get_state() -> str:
    """環境の現在状態を取得"""
    import json
    return json.dumps(env.get_state(), ensure_ascii=False, indent=2)

def env_submit_action(
    agent_name: Annotated[str, "エージェント名"],
    action_type: Annotated[str, "アクション種別"],
    action_data: Annotated[str, "アクションデータ（JSON）"]
) -> str:
    """環境にアクションを送信"""
    import json
    action = {
        "type": action_type,
        "agent": agent_name,
        **json.loads(action_data)
    }
    new_state = env.update_state(action)
    return f"アクション完了。新しい状態: {json.dumps(new_state, ensure_ascii=False)}"
```

---

## 6. ログと結果取得

シミュレーション検討シートの「ログ」セクションに対応

### 6.1 ChatResult の構造

```python
# initiate_chat の戻り値
chat_result = agent_a.initiate_chat(
    recipient=agent_b,
    message="開始",
    max_turns=5,
)

# ChatResult の属性
print(f"チャット履歴: {chat_result.chat_history}")
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

print(f"要約: {chat_result.summary}")
# 最後のメッセージまたはLLMによる要約

print(f"コスト: {chat_result.cost}")
# {"usage_including_cached_inference": {...}, "usage_excluding_cached_inference": {...}}

print(f"人間の入力: {chat_result.human_input}")
# 人間の入力があった場合のリスト
```

### 6.2 メッセージ履歴の取得

```python
# エージェントのチャットメッセージを取得
messages = agent.chat_messages[other_agent]
# [{"role": "...", "content": "..."}, ...]

# GroupChat のメッセージを取得
messages = group_chat.messages
# [{"role": "...", "name": "agent_name", "content": "..."}, ...]
```

### 6.3 構造化ログの実装

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class MessageLog:
    """メッセージログ"""
    timestamp: str
    round: int
    speaker: str
    content: str
    action: Optional[dict] = None

@dataclass
class RunLog:
    """実行ログ"""
    seed: int
    start_time: str
    end_time: str
    messages: List[MessageLog]
    final_state: dict
    metrics: dict
    termination_reason: str

class SimulationLogger:
    """シミュレーションログを管理"""
    
    def __init__(self):
        self.current_run: Optional[RunLog] = None
        self.runs: List[RunLog] = []
        self.message_buffer: List[MessageLog] = []
        self.round = 0
    
    def start_run(self, seed: int):
        """新しい実行を開始"""
        self.message_buffer = []
        self.round = 0
        self.start_time = datetime.now().isoformat()
        self.seed = seed
    
    def log_message(self, speaker: str, content: str, action: dict = None):
        """メッセージをログに記録"""
        log = MessageLog(
            timestamp=datetime.now().isoformat(),
            round=self.round,
            speaker=speaker,
            content=content,
            action=action,
        )
        self.message_buffer.append(log)
    
    def next_round(self):
        """次のラウンドに進む"""
        self.round += 1
    
    def end_run(self, final_state: dict, metrics: dict, termination_reason: str):
        """実行を終了"""
        run_log = RunLog(
            seed=self.seed,
            start_time=self.start_time,
            end_time=datetime.now().isoformat(),
            messages=self.message_buffer.copy(),
            final_state=final_state,
            metrics=metrics,
            termination_reason=termination_reason,
        )
        self.runs.append(run_log)
        self.current_run = run_log
    
    def to_dict(self) -> dict:
        """辞書形式で出力"""
        return {
            "runs": [asdict(run) for run in self.runs]
        }
    
    def save(self, filepath: str):
        """ファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
```

### 6.4 カスタムメッセージハンドラー

```python
def message_handler(
    recipient: ConversableAgent,
    messages: List[dict],
    sender: ConversableAgent,
    config: dict
) -> tuple:
    """メッセージを処理してログに記録"""
    last_message = messages[-1]
    content = last_message.get("content", "")
    
    # ログに記録
    logger.log_message(
        speaker=sender.name,
        content=content,
    )
    
    # False を返すと通常の処理を続行
    return False, None

# ハンドラーを登録
for agent in agents:
    agent.register_reply(
        trigger=[ConversableAgent, None],
        reply_func=message_handler,
        config={"logger": logger},
    )
```

---

## 7. 状態管理

シミュレーション検討シートの「エージェントの状態」セクションに対応

### 7.1 エージェントの内部状態管理

```python
class StatefulAgent:
    """状態を持つエージェントのラッパー"""
    
    def __init__(self, agent: ConversableAgent, initial_state: dict):
        self.agent = agent
        self.state = initial_state.copy()
        self.state_history = [initial_state.copy()]
    
    def update_state(self, updates: dict):
        """状態を更新"""
        self.state.update(updates)
        self.state_history.append(self.state.copy())
    
    def get_state_prompt(self) -> str:
        """状態をプロンプトに変換"""
        return f"""
現在の状態:
- 所持金: {self.state.get('balance', 0)}円
- 在庫: {self.state.get('inventory', {})}
- 評判: {self.state.get('reputation', 0)}
"""
    
    def inject_state_to_system_message(self):
        """システムメッセージに状態を注入"""
        base_message = self.agent.system_message
        state_prompt = self.get_state_prompt()
        self.agent.update_system_message(f"{base_message}\n\n{state_prompt}")
```

### 7.2 SwarmAgent による状態管理

```python
from autogen import SwarmAgent, UPDATE_SYSTEM_MESSAGE

# 状態更新関数
def update_agent_state(agent: SwarmAgent, messages: List[dict]) -> str:
    """エージェントの状態を更新"""
    # メッセージから状態を抽出
    last_content = messages[-1].get("content", "")
    
    # 状態更新ロジック
    new_balance = extract_balance(last_content)
    
    return f"現在の残高: {new_balance}円"

# SwarmAgent の作成
swarm_agent = SwarmAgent(
    name="trader",
    system_message="あなたはトレーダーです。",
    llm_config=llm_config,
    update_agent_state_before_reply=[
        UPDATE_SYSTEM_MESSAGE(update_agent_state),
    ],
)
```

### 7.3 context_variables による状態共有

```python
from autogen import initiate_swarm_chat

# 共有状態
context_variables = {
    "round": 0,
    "market_state": {"price": 100},
    "player_states": {
        "player_0": {"balance": 1000},
        "player_1": {"balance": 1000},
    },
}

# Swarm チャットの開始
chat_result, updated_context, last_speaker = initiate_swarm_chat(
    initial_agent=player_agents[0],
    messages="ゲームを開始します",
    agents=player_agents,
    context_variables=context_variables,
    max_rounds=20,
)

# 更新された状態を取得
print(f"最終状態: {updated_context}")
```

---

## 8. 実装パターン集

### 8.1 囚人のジレンマシミュレーション

```python
"""囚人のジレンマ シミュレーション"""

import os
import json
import numpy as np
from autogen import ConversableAgent, GroupChat, GroupChatManager

# 設定
llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}],
    "temperature": 0.7,
}

# 利得行列
PAYOFF_MATRIX = {
    ("協力", "協力"): (3, 3),
    ("協力", "裏切り"): (0, 5),
    ("裏切り", "協力"): (5, 0),
    ("裏切り", "裏切り"): (1, 1),
}

class PrisonersDilemmaSimulation:
    def __init__(self, llm_config: dict, num_rounds: int = 10):
        self.llm_config = llm_config
        self.num_rounds = num_rounds
        self.results = []
        
    def create_agents(self):
        """エージェントを作成"""
        self.player_a = ConversableAgent(
            name="Player_A",
            system_message="""
あなたは囚人のジレンマゲームのプレイヤーAです。
各ラウンドで「協力」か「裏切り」を選択してください。

利得行列:
- 両者協力: 各3点
- 自分だけ裏切り: 5点（相手は0点）
- 自分だけ協力: 0点（相手は5点）
- 両者裏切り: 各1点

回答は必ず「選択: 協力」または「選択: 裏切り」の形式で行ってください。
""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        
        self.player_b = ConversableAgent(
            name="Player_B",
            system_message="""
あなたは囚人のジレンマゲームのプレイヤーBです。
各ラウンドで「協力」か「裏切り」を選択してください。

利得行列:
- 両者協力: 各3点
- 自分だけ裏切り: 5点（相手は0点）
- 自分だけ協力: 0点（相手は5点）
- 両者裏切り: 各1点

回答は必ず「選択: 協力」または「選択: 裏切り」の形式で行ってください。
""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        
        self.moderator = ConversableAgent(
            name="Moderator",
            system_message="ゲームの進行を管理します。",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
    
    def extract_choice(self, content: str) -> str:
        """メッセージから選択を抽出"""
        if "協力" in content:
            return "協力"
        elif "裏切り" in content:
            return "裏切り"
        return "協力"  # デフォルト
    
    def run_round(self, round_num: int, history: list) -> dict:
        """1ラウンドを実行"""
        history_str = "\n".join([
            f"ラウンド{r['round']}: A={r['choice_a']}, B={r['choice_b']}, 得点: A={r['payoff_a']}, B={r['payoff_b']}"
            for r in history
        ]) if history else "まだ履歴はありません"
        
        # Player A の選択
        result_a = self.moderator.initiate_chat(
            recipient=self.player_a,
            message=f"ラウンド{round_num}です。\n履歴:\n{history_str}\n\n選択してください。",
            max_turns=1,
        )
        choice_a = self.extract_choice(result_a.summary)
        
        # Player B の選択
        result_b = self.moderator.initiate_chat(
            recipient=self.player_b,
            message=f"ラウンド{round_num}です。\n履歴:\n{history_str}\n\n選択してください。",
            max_turns=1,
        )
        choice_b = self.extract_choice(result_b.summary)
        
        # 利得計算
        payoff_a, payoff_b = PAYOFF_MATRIX[(choice_a, choice_b)]
        
        return {
            "round": round_num,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
        }
    
    def run(self) -> dict:
        """シミュレーションを実行"""
        self.create_agents()
        history = []
        
        for round_num in range(1, self.num_rounds + 1):
            round_result = self.run_round(round_num, history)
            history.append(round_result)
            print(f"ラウンド{round_num}: A={round_result['choice_a']}, B={round_result['choice_b']}")
        
        # 集計
        total_a = sum(r["payoff_a"] for r in history)
        total_b = sum(r["payoff_b"] for r in history)
        
        return {
            "history": history,
            "total_payoff_a": total_a,
            "total_payoff_b": total_b,
            "cooperation_rate_a": sum(1 for r in history if r["choice_a"] == "協力") / len(history),
            "cooperation_rate_b": sum(1 for r in history if r["choice_b"] == "協力") / len(history),
        }

# 実行
if __name__ == "__main__":
    sim = PrisonersDilemmaSimulation(llm_config, num_rounds=5)
    results = sim.run()
    print(json.dumps(results, ensure_ascii=False, indent=2))
```

### 8.2 議論シミュレーション（GroupChat）

```python
"""複数エージェントによる議論シミュレーション"""

import os
from autogen import ConversableAgent, GroupChat, GroupChatManager

llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}],
    "temperature": 0.7,
}

# エージェントの作成
optimist = ConversableAgent(
    name="Optimist",
    system_message="""
あなたは楽観的な視点を持つ討論者です。
議題に対して肯定的な意見を述べ、メリットや可能性を強調してください。
他の参加者の意見にも建設的に応答してください。
""",
    description="楽観的な視点から意見を述べる",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

pessimist = ConversableAgent(
    name="Pessimist",
    system_message="""
あなたは慎重な視点を持つ討論者です。
議題に対してリスクや課題を指摘してください。
他の参加者の意見に対しても批判的に検討してください。
""",
    description="慎重な視点からリスクを指摘する",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

mediator = ConversableAgent(
    name="Mediator",
    system_message="""
あなたは討論の調停者です。
両者の意見を整理し、合意点を見つけ出してください。
議論が収束したら「合意に達しました」と宣言してください。
""",
    description="議論を調停し合意を導く",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "合意に達しました" in x.get("content", ""),
)

# グループチャットの設定
group_chat = GroupChat(
    agents=[optimist, pessimist, mediator],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",
    send_introductions=True,
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# 議論の開始
topic = "AIの普及は社会にとって良いことか"
chat_result = mediator.initiate_chat(
    recipient=manager,
    message=f"今日の議題: {topic}\n\nまず楽観的な意見から始めましょう。",
)

# 結果の出力
print("\n=== 議論の履歴 ===")
for msg in group_chat.messages:
    print(f"[{msg.get('name', 'unknown')}]: {msg.get('content', '')[:100]}...")

print(f"\n=== 要約 ===\n{chat_result.summary}")
```

### 8.3 市場シミュレーション（ツール付き）

```python
"""市場シミュレーション with ツール"""

import os
import json
from typing import Annotated
from autogen import ConversableAgent, GroupChat, GroupChatManager, register_function

llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}],
    "temperature": 0.5,
}

# 市場環境
class Market:
    def __init__(self):
        self.price = 100
        self.order_book = {"buy": [], "sell": []}
        self.transactions = []
        self.round = 0
    
    def get_price(self) -> int:
        return self.price
    
    def submit_order(self, agent: str, side: str, price: int, quantity: int) -> dict:
        order = {"agent": agent, "side": side, "price": price, "quantity": quantity, "round": self.round}
        self.order_book[side].append(order)
        return order
    
    def match_orders(self) -> list:
        """注文をマッチング"""
        matched = []
        buy_orders = sorted(self.order_book["buy"], key=lambda x: -x["price"])
        sell_orders = sorted(self.order_book["sell"], key=lambda x: x["price"])
        
        for buy in buy_orders:
            for sell in sell_orders:
                if buy["price"] >= sell["price"] and buy["quantity"] > 0 and sell["quantity"] > 0:
                    trade_qty = min(buy["quantity"], sell["quantity"])
                    trade_price = (buy["price"] + sell["price"]) // 2
                    
                    matched.append({
                        "buyer": buy["agent"],
                        "seller": sell["agent"],
                        "price": trade_price,
                        "quantity": trade_qty,
                    })
                    
                    buy["quantity"] -= trade_qty
                    sell["quantity"] -= trade_qty
                    self.price = trade_price
        
        self.transactions.extend(matched)
        return matched

market = Market()

# ツール関数
def get_market_price() -> str:
    """現在の市場価格を取得"""
    return f"現在の市場価格: {market.get_price()}円"

def submit_buy_order(
    agent_name: Annotated[str, "エージェント名"],
    price: Annotated[int, "買い希望価格"],
    quantity: Annotated[int, "数量"]
) -> str:
    """買い注文を提出"""
    order = market.submit_order(agent_name, "buy", price, quantity)
    return f"買い注文を受け付けました: {json.dumps(order, ensure_ascii=False)}"

def submit_sell_order(
    agent_name: Annotated[str, "エージェント名"],
    price: Annotated[int, "売り希望価格"],
    quantity: Annotated[int, "数量"]
) -> str:
    """売り注文を提出"""
    order = market.submit_order(agent_name, "sell", price, quantity)
    return f"売り注文を受け付けました: {json.dumps(order, ensure_ascii=False)}"

# エージェントの作成
buyer = ConversableAgent(
    name="Buyer",
    system_message="""
あなたは市場の買い手です。
できるだけ安く買うことを目指してください。
ツールを使って市場価格を確認し、買い注文を出してください。
""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

seller = ConversableAgent(
    name="Seller",
    system_message="""
あなたは市場の売り手です。
できるだけ高く売ることを目指してください。
ツールを使って市場価格を確認し、売り注文を出してください。
""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

executor = ConversableAgent(
    name="Executor",
    llm_config=False,
    human_input_mode="NEVER",
)

# ツールの登録
for agent in [buyer, seller]:
    register_function(get_market_price, caller=agent, executor=executor, description="市場価格を取得")
    register_function(submit_buy_order, caller=agent, executor=executor, description="買い注文を提出")
    register_function(submit_sell_order, caller=agent, executor=executor, description="売り注文を提出")

# 実行
chat_result = buyer.initiate_chat(
    recipient=seller,
    message="取引を開始しましょう。まず市場価格を確認してください。",
    max_turns=5,
)
```

---

## 補足: シミュレーション検討シートとAG2の対応表

| 検討シートの項目 | AG2での実装方法 |
|---|---|
| **エージェント** | |
| 人数 | `GroupChat(agents=[...])` のエージェントリスト |
| ロールと説明 | `system_message`, `description` |
| 内部状態 | カスタムクラス、`context_variables`、`UPDATE_SYSTEM_MESSAGE` |
| 状態更新 | `update_agent_state_before_reply`、カスタムハンドラー |
| 環境との相互作用 | `register_function`、ツール |
| **環境** | |
| 環境の構造 | カスタム環境クラス |
| 環境の状態仕様 | `context_variables`、グローバル状態オブジェクト |
| 環境の更新ルール | ツール関数、カスタムハンドラー |
| **プロトコル** | |
| ターン構造 | `max_turns`, `max_round` |
| 終了条件 | `is_termination_msg`, `max_consecutive_auto_reply` |
| フェーズ構造 | `initiate_chats`（Sequential Chat）、Nested Chat |
| 対話フロー | `speaker_selection_method`、カスタム選択関数 |
| **ルール** | |
| 共有情報 | GroupChatの`messages`、`send_introductions` |
| 非公開情報 | 個別の`system_message` |
| 意思決定ルール | `system_message`内の指示 |
| **ログ** | |
| ログ形式 | `ChatResult.chat_history`、カスタムLogger |
| 記録内容 | カスタムメッセージハンドラー |
| **分析指標** | |
| 指標の定義 | 後処理関数、カスタムMetricsクラス |
| 検証方法 | ログデータの集計・分析 |

---

## 参考リンク

- [AG2 公式ドキュメント](https://docs.ag2.ai/)
- [AG2 GitHub](https://github.com/ag2ai/ag2)
- [AG2 API Reference](https://docs.ag2.ai/docs/api-reference/)
- [Conversation Patterns](https://docs.ag2.ai/docs/tutorial/conversation-patterns)
- [GroupChat Guide](https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/groupchat/groupchat/)
