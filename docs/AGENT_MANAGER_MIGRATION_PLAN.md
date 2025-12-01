# AgentManager移植計画

AI-Scientist-v2の`AgentManager`をAI-Masistに移植するための計画書

## 1. プロジェクト構造の比較

| 項目 | AI-Scientist-v2 | AI-Masist |
|------|----------------|-----------|
| **目的** | ML実験の自動化 | MAS(Multi-Agent Simulation)の自動化 |
| **ステージ数** | 4ステージ (Initial→Baseline→Creative→Ablation) | Stage 1のみ |
| **ParallelAgent** | 2368行、ステージ対応 | 574行、シンプル版 |
| **task_desc** | JSON形式（Title, Abstract, Hypothesis等） | 日本語シミュレーション検討シート（文字列） |
| **Journal管理** | `Dict[str, Journal]`（ステージ毎） | 単一`Journal` |

---

## 2. コピペ移植対象ファイル

### 2.1 メインファイル: `agent_manager.py`

**コピー元:** `/AI-Scientist-v2/ai_scientist/treesearch/agent_manager.py` (1221行)

**コピー先:** `/AI-Masist/masist/treesearch/agent_manager.py`

#### 必要な変更箇所

##### 変更不要: main_stage_dict, main_stage_goals

AI-Scientist-v2のまま使用（変更なし）

**main_stage_dict:**

| Stage | Name |
|-------|------|
| 1 | initial_implementation |
| 2 | baseline_tuning |
| 3 | creative_research |
| 4 | ablation_studies |

**main_stage_goals（AI-MASist向け）:**

| Stage | Goals |
|-------|-------|
| 1 | - **エンドツーエンドで動く最小実装を完成させる**（仮説→シミュレーション実行→結果要約→論文ドラフト冒頭まで）<br>- **ごくシンプルなシナリオ**（例：2エージェントの交渉・協力ゲーム・役割対立など）でマルチエージェント対話シミュレーションを動かす<br>- ロール定義 / 対話プロトコル / 終了条件 / ログ形式 を「1つの型」として最後まで通す |
| 2 | - **実験条件のバリエーションを増やして、結果の安定性・妥当性を高める**<br>- シミュレーションの基本構造（エージェントフレームワークやプロトコルの大枠）は変えずに、以下を系統的に変える：<br>　- プロンプト条件（フレーミング、ルール説明の仕方など）<br>　- 環境パラメータ（報酬構造、情報の非対称性、人数、ルールなど） |
| 3 | - 新しい改善を探索<br>- 新しい知見を得るための実験を考案<br>- 創造的に、既存の枠にとらわれずに考える |
| 4 | - **各要素の貢献度を明らかにする系統的分析フェーズ**<br>- Stage 2〜3 で使用した**同じシナリオセット**を使用 |

##### 変更1: task_descをMASist専用JSON形式に変更

**AI-Scientist-v2の形式ではなく、シミュレーション検討シートの構造に合わせたMASist専用JSON型を定義する。**

**agent_manager.pyのキーチェック部分も変更が必要。**

---

#### MASist専用 task_desc JSON スキーマ

```python
MASIST_TASK_DESC_SCHEMA = {
    # ============================================
    # 基本情報
    # ============================================
    "Title": str,           # シミュレーション名（必須）
    "Name": str,            # 識別子（必須）

    # ============================================
    # 1. シミュレーション要求
    # ============================================
    "SimulationRequest": {
        "Background": str,           # 背景・文脈（必須）
        "Purpose": str,              # 目的（必須）
        "ResearchQuestions": list,   # 研究質問（必須）
        "Hypotheses": list,          # 仮説（必須）
        "Other": str,                # その他（任意）
    },

    # ============================================
    # 2. シミュレーション要件
    # ============================================
    "SimulationRequirements": {
        # --- エージェント ---
        "Agents": {
            "Count": int | str,                  # 人数
            "RolesAndDescriptions": str,         # ロールと説明
            "State": str,                        # 状態（記憶・内部状態・行動）
            "StateUpdate": str,                  # 状態更新
            "EnvironmentInteraction": str,       # 環境との相互作用
        },
        # --- 環境 ---
        "Environment": {
            "Structure": str,        # 環境の構造
            "StateSpec": str,        # 環境の状態仕様
            "UpdateRules": str,      # 環境の更新ルール
        },
        # --- プロトコル ---
        "Protocol": {
            "TurnStructure": str,          # ターン/ラウンド/タイムステップ構造
            "TerminationCondition": str,   # 終了条件
            "TrialCount": int | str,       # 各工程の試行数
            "PhaseStructure": str,         # フェーズ構造（任意）
            "DialogFlow": str,             # 対話フロー
        },
        # --- ルール ---
        "Rules": {
            "SharedInfo": str,             # 共有情報
            "PrivateInfo": str,            # 非公開情報
            "DecisionRules": str,          # 意思決定ルール（任意）
            "PayoffStructure": str,        # 利得構造（任意）
            "ExperimentConditions": list,  # 実験条件
        },
    },

    # ============================================
    # 3. ログ
    # ============================================
    "Logging": {
        "RecordContents": list,          # 記録すべき内容
        "Format": str,                   # ログ形式
        "AnalysisMetrics": list,         # 分析指標
        "HypothesisVerification": str,   # 仮説検証方法
    },

    # ============================================
    # その他
    # ============================================
    "Other": str,    # その他（任意）
}
```

---

#### 必須キーと任意キー

| カテゴリ | 必須キー | 任意キー |
|----------|----------|----------|
| 基本情報 | Title, Name | - |
| シミュレーション要求 | Background, Purpose, ResearchQuestions, Hypotheses | Other |
| エージェント | Count, RolesAndDescriptions, State, StateUpdate, EnvironmentInteraction | - |
| 環境 | Structure, StateSpec, UpdateRules | - |
| プロトコル | TurnStructure, TerminationCondition, DialogFlow | TrialCount, PhaseStructure |
| ルール | SharedInfo, PrivateInfo, ExperimentConditions | DecisionRules, PayoffStructure |
| ログ | RecordContents, AnalysisMetrics, HypothesisVerification | Format |
| その他 | - | Other |

---

#### AI-Scientist-v2キーとのマッピング

AgentManagerで内部的に使用するため、MASistキーからAI-Scientist-v2キーへのマッピングを行う：

| AI-Scientist-v2 キー | MASist キーからの変換 |
|---------------------|----------------------|
| Title | Title（そのまま） |
| Abstract | SimulationRequest.Background + SimulationRequest.Purpose |
| Short Hypothesis | SimulationRequest.Hypotheses（リストを結合） |
| Experiments | SimulationRequirements.Rules.ExperimentConditions |
| Risk Factors and Limitations | Other または空リスト |
| Name | Name（そのまま） |

---

#### TPGG実験の例（MASist JSON形式）

```python
TPGG_TASK_DESC = {
    "Title": "Threshold Public Goods Game (TPGG)",
    "Name": "tpgg",

    "SimulationRequest": {
        "Background": "4人グループで、一定額のトークンを共同で拠出できればご褒美（V）がもらえるゲームにおける、グループに提示される拠出目安（ルール）の性質が、実際の行動にどう影響するかをLLMエージェントを用いて調査する。",
        "Purpose": "グループに示される「出してほしい金額の目安（ルール）」が、「ちょうど必要な合計」か「必要以上に多い合計」かによって、LLMエージェントの行動、結果（達成率）、効率（過剰拠出）がどう変わるのかを調べる。",
        "ResearchQuestions": [
            "ルールが「必要以上に多い（多めの要求）」だと、行動が乱れやすいか？",
            "必要な分ちょうどのルールは、むしろ安定した協力を生むか？",
            "グループ全体がちょうど必要額に合わせる「効率の良さ」はどう変わるか？"
        ],
        "Hypotheses": [
            "多めに要求されたルールは、守る人が減りやすい（協調性の低下）",
            "必要額を達成できるかどうかは、ルールの違いではあまり変わらない",
            "多めのルールは「出しすぎ（無駄）」を増やし、効率を下げる",
            "1人あたりの負担がピッタリ均等割りできる場合、協力がまとまりやすい"
        ],
        "Other": ""
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": 4,
            "RolesAndDescriptions": "4人とも同じ立場。各ラウンドで、自分の持ちトークン10のうち、何トークンを共同の箱に入れるか選ぶ。",
            "State": "過去の自分の拠出額、グループ合計の拠出額、自分の得点、（後半のみ）自分に示された「出してほしい額（ルール）」",
            "StateUpdate": "ラウンドの最後に、そのラウンドの情報を記録し、次のラウンドの意思決定の参考にする",
            "EnvironmentInteraction": "全員の合計がしきい値(T)を超えれば、ご褒美(V)がもらえる"
        },
        "Environment": {
            "Structure": "4人は同じグループ。グループ間の交流はなし（閉じた小世界で毎回意思決定）",
            "StateSpec": "しきい値(T)、ご褒美額(V)、各プレイヤーの拠出額、合計拠出額",
            "UpdateRules": "各ラウンド終了時に合計拠出額を計算し、しきい値達成判定を行い、利得を計算"
        },
        "Protocol": {
            "TurnStructure": "1回のゲームは20ラウンド。各ラウンドは①4人が同時に出す額を決める→②合計額を見る→③しきい値判定→④得点を返す",
            "TerminationCondition": "20ラウンド終了",
            "TrialCount": "1つの設定につき、4人グループを複数回（例：11グループ）まわす",
            "PhaseStructure": "ラウンド1〜10: ルールなし（ベースライン）、ラウンド11〜20: 設定に応じたルールを提示",
            "DialogFlow": "環境→エージェントに状況提示、エージェント→環境に拠出額を返す"
        },
        "Rules": {
            "SharedInfo": "各自の持ちトークン10、しきい値(T)、（後半）みんなの「出してほしい額（ルール）」、ラウンド後の合計拠出額、自分の得点",
            "PrivateInfo": "他のメンバーが実際にいくら出したか、各メンバーの考え・意図",
            "DecisionRules": "0〜10の整数から1つ選んで拠出する",
            "PayoffStructure": "しきい値未達成: π_i = 10 - c_i、しきい値達成: π_i = 10 - c_i + V",
            "ExperimentConditions": [
                {"name": "FAIRSUFF", "T": 20, "R": [5,5,5,5], "description": "必要額ちょうど＋均等割りOK"},
                {"name": "FAIRINF", "T": 20, "R": [5,5,6,6], "description": "多めの要求＋均等割りOK"},
                {"name": "UNFAIRSUFF", "T": 22, "R": [5,5,6,6], "description": "必要額ちょうど＋均等割り不可"},
                {"name": "UNFAIRINF", "T": 22, "R": [6,6,6,6], "description": "多めの要求＋均等割り不可"},
                {"name": "CONTROL", "T": 22, "R": None, "description": "ルールなし（ベースライン）"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "設定情報（どの設定で行ったか）",
            "ラウンド番号",
            "各メンバーの出した額",
            "合計出した額",
            "しきい値達成の有無",
            "各メンバーの得点",
            "ルールを守ったかのフラグ（R11-20のみ）"
        ],
        "Format": "1行が1ラウンド",
        "AnalysisMetrics": [
            "しきい値達成率",
            "平均拠出額",
            "過剰拠出（C - T）",
            "ルール遵守率（R11-20のみ）",
            "10ラウンド目→11ラウンド目の変化率"
        ],
        "HypothesisVerification": "各条件間で達成率・遵守率・過剰拠出を比較し、仮説①〜④を統計的に検証"
    },

    "Other": ""
}
```

---

#### agent_manager.py の変更箇所

```python
# 元のキーチェック（AI-Scientist-v2）
for k in [
    "Title",
    "Abstract",
    "Short Hypothesis",
    "Experiments",
    "Risk Factors and Limitations",
]:
    if k not in self.task_desc.keys():
        raise ValueError(f"Key {k} not found in task_desc")

# 変更後（MASist専用）
required_keys = [
    "Title",
    "Name",
    "SimulationRequest",
    "SimulationRequirements",
    "Logging",
]
for k in required_keys:
    if k not in self.task_desc.keys():
        raise ValueError(f"Key {k} not found in task_desc")

# AI-Scientist-v2互換キーを内部生成
self._generate_compat_keys()

def _generate_compat_keys(self):
    """MASistキーからAI-Scientist-v2互換キーを生成"""
    sr = self.task_desc["SimulationRequest"]
    self.task_desc["Abstract"] = f"{sr['Background']}\n\n{sr['Purpose']}"
    self.task_desc["Short Hypothesis"] = "\n".join(sr["Hypotheses"])
    self.task_desc["Experiments"] = self.task_desc["SimulationRequirements"]["Rules"]["ExperimentConditions"]
    self.task_desc["Risk Factors and Limitations"] = self.task_desc.get("Other", "")
    # Code キーは使用しない（MASistでは初期コード不要）
```

---

### 2.2 設定追加: `config.py`

**追加先:** `/AI-Masist/masist/treesearch/utils/config.py`

```python
# ---- 追加: StagesConfig ----
@dataclass
class StagesConfig:
    """Stage-specific iteration limits"""
    stage1_max_iters: int = 20  # Initial Implementation
    stage2_max_iters: int = 12  # Scenario Tuning
    stage3_max_iters: int = 12  # Hypothesis Exploration
    stage4_max_iters: int = 18  # Sensitivity Analysis


@dataclass
class AgentConfig:
    """Agent configuration"""
    num_workers: int = 2
    # ... 既存フィールド ...
    stages: StagesConfig = field(default_factory=StagesConfig)  # ← 追加
```

---

### 2.3 ParallelAgent拡張

**変更先:** `/AI-Masist/masist/treesearch/parallel_agent.py`

#### 必要な変更

```python
# ---- コンストラクタにステージ情報を追加 ----
def __init__(
    self,
    task_desc: str,
    cfg: Any,
    journal: Journal,
    evaluation_metrics: list = None,
    # ↓ 追加パラメータ
    stage_name: str = None,
    best_stage1_node: Node = None,
    best_stage2_node: Node = None,
    best_stage3_node: Node = None,
):
    self.stage_name = stage_name
    self.best_stage1_node = best_stage1_node
    self.best_stage2_node = best_stage2_node
    self.best_stage3_node = best_stage3_node
    # ...
```

---

### 2.4 エントリーポイント追加

**新規作成:** `/AI-Masist/masist/treesearch/perform_experiments_with_agentmanager.py`

**コピー元:** `/AI-Scientist-v2/ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py` (262行)

---

#### ファイル構造

```
perform_experiments_with_agentmanager.py
├── perform_experiments_bfts()     # メイン関数
│   ├── load_cfg()                 # 設定読み込み
│   ├── load_task_desc()           # タスク説明読み込み（要変更）
│   ├── prep_agent_workspace()     # ワークスペース準備
│   ├── AgentManager()             # マネージャー初期化
│   ├── manager.run()              # 実行
│   │   ├── exec_callback          # コード実行コールバック
│   │   └── step_callback          # ステップ完了コールバック
│   ├── pickle.dump(manager)       # 状態保存
│   └── overall_summarize()        # 最終レポート生成
│
├── journal_to_rich_tree()         # 進捗表示用（Rich）
├── create_exec_callback()         # 実行コールバック生成
├── step_callback()                # ステップ完了時の処理
└── generate_live()                # Live表示生成（Rich）
```

---

#### 変更が必要な箇所

##### 1. import文の調整

```python
# 元（AI-Scientist-v2）
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg

# 変更後（MASist）: 関数をコピーで追加
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
```

| AI-Scientist-v2 関数 | 対応 | 説明 |
|---------------------|------|------|
| `load_task_desc` | **コピー追加** | task_desc読み込み（cfg.desc_file or cfg.goal） |
| `prep_agent_workspace` | **コピー追加** | ワークスペース準備（input/working作成、データコピー） |
| `save_run` | **コピー追加** | journal/config/tree_plot/best_solution保存 |
| `load_cfg` | **コピー追加** | config.yaml読み込み＋prep_cfg |

**注意:** `save_run` は以下に依存:
- `serialize.dump_json` → `serialize.py` をコピー済み
- `tree_export.generate` → `tree_export.py` をコピーが必要

その他のimport:
```python
# これらはコピーで対応済み
from .journal2report import journal2report        # Task 2.2でコピー
from .log_summarization import overall_summarize  # Task 2.3でコピー
```

##### 2. task_desc読み込み（MASist JSON形式対応）

```python
# 元（AI-Scientist-v2）: idea.jsonを読み込んでJSON文字列を返す
task_desc = load_task_desc(cfg)

# 変更後（MASist）: JSON形式のdictを読み込む
def load_task_desc(cfg) -> str:
    """MASist形式のtask_descを読み込んでJSON文字列で返す"""
    task_file = Path(cfg.workspace_dir) / "task_desc.json"
    if task_file.exists():
        with open(task_file, "r", encoding="utf-8") as f:
            task_dict = json.load(f)
        return json.dumps(task_dict, ensure_ascii=False)
    else:
        raise FileNotFoundError(f"Task description not found: {task_file}")
```

##### 3. interpreter参照の修正

```python
# 元（グローバルなinterpreterを参照）
def create_exec_callback(status_obj):
    def exec_callback(*args, **kwargs):
        status_obj.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)  # ← interpreterが未定義
        ...

# 変更後（Interpreterをimportして使用）
from .interpreter import Interpreter

def create_exec_callback(status_obj, cfg):
    interpreter = Interpreter(
        working_dir=cfg.workspace_dir,
        timeout=cfg.exec.timeout,
        format_tb_ipython=cfg.exec.format_tb_ipython,
        agent_file_name=cfg.exec.agent_file_name,
    )
    def exec_callback(*args, **kwargs):
        status_obj.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status_obj.update("[green]Generating code...")
        return res
    return exec_callback
```

##### 4. レポート生成部分（必須）

```python
# 元（AI-Scientist-v2）
if cfg.generate_report:
    (draft_summary, baseline_summary, research_summary, ablation_summary) = \
        overall_summarize(manager.journals.items(), cfg)
    ...

# 変更後（MASist）: AI-Scientist-v2と同様にoverall_summarizeを使用
# journal2report.py, log_summarization.pyをコピー済みなのでそのまま使用可能
if cfg.generate_report:
    (draft_summary, baseline_summary, research_summary, ablation_summary) = \
        overall_summarize(manager.journals.items(), cfg)
    # サマリーをファイルに保存
    for name, summary in [
        ("draft", draft_summary),
        ("baseline", baseline_summary),
        ("research", research_summary),
        ("ablation", ablation_summary),
    ]:
        if summary:
            summary_path = cfg.log_dir / f"{name}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary saved: {summary_path}")
```

---

#### コピペ後の最小限の変更リスト

| 行番号(目安) | 変更内容 |
|-------------|----------|
| 1-30 | import文の調整 |
| 64 | `load_task_desc`をMASist形式に |
| 94-101 | `create_exec_callback`でInterpreterを正しく参照 |
| 227-256 | レポート生成部分を実装（`overall_summarize`使用） |

---

#### 追加でコピーが必要なファイル

| ファイル | 説明 | コピー元 |
|------|------|------|
| `journal2report.py` | Journalからレポート生成 | `/AI-Scientist-v2/ai_scientist/treesearch/journal2report.py` |
| `log_summarization.py` | 全ステージのサマリー生成 | `/AI-Scientist-v2/ai_scientist/treesearch/log_summarization.py` |
| `llm.py` | LLM呼び出しユーティリティ | `/AI-Scientist-v2/ai_scientist/llm.py` |
| `utils/token_tracker.py` | トークン使用量追跡 | `/AI-Scientist-v2/ai_scientist/utils/token_tracker.py` |
| `utils/tree_export.py` | ツリー可視化HTML生成 | `/AI-Scientist-v2/ai_scientist/treesearch/utils/tree_export.py` |
| `utils/copytree.py` | ディレクトリコピー | `/AI-Scientist-v2/ai_scientist/treesearch/utils/__init__.py` 内の関数 |

**依存関係:**
```
perform_experiments_with_agentmanager.py
  ├── journal2report.py
  ├── log_summarization.py
  │     └── llm.py
  │           └── utils/token_tracker.py
  └── utils/config.py
        ├── save_run()
        │     ├── serialize.dump_json()
        │     └── tree_export.generate()
        └── prep_agent_workspace()
              └── copytree()
```

---

## 3. 移植タスクリスト

### 前提: task_descをMASist専用JSON形式に統一

---

### Phase 1: 依存なしの基盤ファイル（並行実行可能）

このフェーズのタスクは相互に依存しないため、並行して実行可能。

#### Task 1.1: serialize.py をコピー
- [x] `/AI-Scientist-v2/ai_scientist/treesearch/utils/serialize.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/utils/serialize.py`
- [x] import文の確認（標準ライブラリのみなので変更不要のはず）

#### Task 1.2: token_tracker.py をコピー
- [x] `masist/utils/` ディレクトリを作成（存在しない場合）
- [x] `masist/utils/__init__.py` を作成
- [x] `/AI-Scientist-v2/ai_scientist/utils/token_tracker.py` をコピー
- [x] コピー先: `/AI-Masist/masist/utils/token_tracker.py`

#### Task 1.3: tree_export.py をコピー
- [x] `/AI-Scientist-v2/ai_scientist/treesearch/utils/tree_export.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/utils/tree_export.py`
- [x] `viz_templates/` ディレクトリもコピー（HTMLテンプレート）
- [x] コピー先: `/AI-Masist/masist/treesearch/utils/viz_templates/`

#### Task 1.4: copytree関数をコピー
- [x] `/AI-Scientist-v2/ai_scientist/treesearch/utils/__init__.py` から `copytree` 関数をコピー
- [x] `/AI-Masist/masist/treesearch/utils/__init__.py` に追加
- [x] `__all__` に `copytree` を追加

---

### Phase 2: 依存ありの基盤ファイル

Phase 1完了後に実行。

#### Task 2.1: llm.py をコピー
**依存: Task 1.2 (token_tracker.py)**

- [x] `/AI-Scientist-v2/ai_scientist/llm.py` をコピー
- [x] コピー先: `/AI-Masist/masist/llm.py`
- [x] import文を調整:
  - `from ai_scientist.utils.token_tracker import ...` → `from .utils.token_tracker import ...`
- [x] 主要な関数:
  - `get_response_from_llm()` - LLM呼び出し
  - `get_batch_responses_from_llm()` - バッチ呼び出し
  - `extract_json_between_markers()` - JSON抽出
  - `create_client()` - クライアント生成

#### Task 2.2: config.py に関数・クラスを追加
**依存: Task 1.1 (serialize), Task 1.3 (tree_export), Task 1.4 (copytree)**

- [x] `StagesConfig` dataclassを追加
- [x] `AgentConfig` に `stages: StagesConfig` フィールドを追加
- [x] `generate_report: bool = True` フィールドを追加
- [x] `load_task_desc()` 関数をコピー追加
- [x] `prep_agent_workspace()` 関数をコピー追加（copytreeを使用）
- [x] `save_run()` 関数をコピー追加（serialize.dump_json, tree_export.generateを使用）
- [x] `load_cfg()` 関数をコピー追加
- [x] `prep_cfg()` 関数をコピー追加
- [x] `__init__.py` のexportを更新

---

### Phase 3: コア機能

Phase 2完了後に実行。

#### Task 3.1: parallel_agent.py にステージ関連パラメータを追加
**依存: Task 2.2 (StagesConfig)**

- [x] `__init__` に以下のパラメータを追加:
  - `stage_name: str = None`
  - `best_stage1_node: Node = None`
  - `best_stage2_node: Node = None`
  - `best_stage3_node: Node = None`
- [x] コンテキストマネージャ (`__enter__`, `__exit__`) が正しく動作するか確認

#### Task 3.2: agent_manager.py をコピー・修正
**依存: Task 2.2 (config), Task 3.1 (parallel_agent)**

- [x] `/AI-Scientist-v2/ai_scientist/treesearch/agent_manager.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/agent_manager.py`
- [x] import文を調整:
  - `from ai_scientist.treesearch...` → `from masist.treesearch...`
  - `from ai_scientist.llm...` → `from masist.llm...`
- [ ] `main_stage_goals` をMASist向けに書き換え（セクション2.1参照）
- [ ] キーチェック部分を変更（MASist専用キー）
- [ ] `_generate_compat_keys()` メソッドを追加

---

### Phase 4: レポート関連

Phase 2完了後に実行（Phase 3と並行可能）。

#### Task 4.1: journal2report.py をコピー
**依存: Task 2.1 (llm.py)**

- [x] `/AI-Scientist-v2/ai_scientist/treesearch/journal2report.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/journal2report.py`
- [x] import文を調整:
  - `from ai_scientist.llm import ...` → `from masist.llm import ...`

#### Task 4.2: log_summarization.py をコピー
**依存: Task 2.1 (llm.py)**

- [x] `/AI-Scientist-v2/ai_scientist/treesearch/log_summarization.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/log_summarization.py`
- [x] import文を調整:
  - `from ai_scientist.llm import ...` → `from masist.llm import ...`
  - `from ai_scientist.treesearch...` → `from masist.treesearch...`

---

### Phase 5: エントリーポイント

Phase 3, Phase 4完了後に実行。

#### Task 5.1: perform_experiments_with_agentmanager.py をコピー
**依存: Task 3.2 (agent_manager), Task 4.1 (journal2report), Task 4.2 (log_summarization)**

- [x] `/AI-Scientist-v2/ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py` をコピー
- [x] コピー先: `/AI-Masist/masist/treesearch/perform_experiments_with_agentmanager.py`

#### Task 5.2: import文を調整
- [x] `from ai_scientist...` → `from masist...` に変更
- [x] `from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg`

#### Task 5.3: task_desc読み込みを修正
- [x] `load_task_desc()` をMASist JSON形式対応に変更
- [x] `task_desc.json` から読み込むように修正

#### Task 5.4: interpreter参照を修正
- [x] `create_exec_callback` で `Interpreter` を正しくインスタンス化
- [x] `from .interpreter import Interpreter` を追加

#### Task 5.5: レポート生成の実装（必須）
- [x] `from .journal2report import journal2report` を追加
- [x] `from .log_summarization import overall_summarize` を追加
- [x] `overall_summarize()` を呼び出すコードを実装
- [x] 4種類のサマリー（draft, baseline, research, ablation）がファイル出力されることを確認

---

### Phase 6: task_desc形式の変換

Phase 3完了後に実行（Phase 4, 5と並行可能）。

#### Task 6.1: agent_manager.py のMASist対応（Task 3.2の残り）
- [x] `main_stage_goals` をMASist向けに書き換え（セクション2.1参照）
- [x] キーチェック部分を変更（MASist専用キー）
- [x] `_generate_compat_keys()` メソッドを追加

#### Task 6.2: experiments.py のtask_descをJSON形式に変換
- [x] `TPGG_TASK_DESC` を MASist JSON形式に変換（セクション2.1の例を参照）
- [x] `ABM_TASK_DESC` を MASist JSON形式に変換
- [x] `PGG_SANCTION_TASK_DESC` を MASist JSON形式に変換
- [x] `PGG_SELF_AWARENESS_TASK_DESC` を MASist JSON形式に変換
- [x] `get_experiment()` 関数が新形式に対応しているか確認

#### Task 6.3: task_descを使用している箇所を確認・修正
- [x] `code_generator.py` - task_descの参照方法を確認（文字列として受け取り、互換性OK）
- [x] `node_processor.py` - task_descの受け渡しを確認（文字列として受け渡し、互換性OK）
- [x] `parallel_agent.py` - task_descの使用箇所を確認（文字列として受け取り、互換性OK）
- [x] `result_evaluator.py` - task_descの使用箇所を確認（文字列として受け取り、互換性OK）
- [x] `vlm_analyzer.py` - task_descの使用箇所を確認（文字列として受け取り、互換性OK）

---

### Phase 7: テストの修正

Phase 5, Phase 6完了後に実行。

#### Task 7.1: 既存テストファイルの修正
- [x] `test_parallel.py` - task_descの渡し方を修正（json.dumps追加）
- [x] `test_worker.py` - task_descの渡し方を修正（fixtures.experimentsからimport）
- [x] `test_phases.py` - task_descの渡し方を修正（fixtures.experimentsからimport）
- [x] `test_aggregation.py` - task_descの渡し方を修正（json.dumps追加）
- [x] `test_best_node_selection.py` - 変更不要（文字列として直接使用）

#### Task 7.2: 新規テストの追加
- [x] `test_agent_manager.py` - AgentManagerの単体テスト
- [x] ステージ遷移のテスト（test_stage_transitions）
- [x] task_desc JSON形式のバリデーションテスト（test_invalid_json_format）
- [x] `_generate_compat_keys()` のテスト（test_generate_compat_keys）

---

### Phase 8: 統合テスト

全Phase完了後に実行。

#### Task 8.1: エンドツーエンドテスト
- [ ] TPGGでAgentManager経由の実行テスト
- [ ] 全4ステージが正常に遷移するか確認
- [ ] ログ・レポートが正しく出力されるか確認
- [ ] `overall_summarize()` が4種類のサマリーを生成することを確認

---

### 変更ファイル一覧

| ファイル | アクション | 変更量 | Phase |
|----------|-----------|--------|-------|
| `masist/treesearch/utils/serialize.py` | 新規コピー | ~70行 | 1 |
| `masist/utils/token_tracker.py` | 新規コピー | ~223行 | 1 |
| `masist/utils/__init__.py` | 新規作成 | 少量 | 1 |
| `masist/treesearch/utils/tree_export.py` | 新規コピー | ~200行 | 1 |
| `masist/treesearch/utils/viz_templates/` | 新規コピー | ディレクトリ | 1 |
| `masist/treesearch/utils/__init__.py` | 修正（copytree追加） | ~20行 | 1 |
| `masist/llm.py` | 新規コピー＋修正 | ~545行 | 2 |
| `masist/treesearch/utils/config.py` | 追記 | ~100行 | 2 |
| `masist/treesearch/parallel_agent.py` | 修正 | ~30行 | 3 |
| `masist/treesearch/agent_manager.py` | 新規コピー＋修正 | ~1200行 | 3 |
| `masist/treesearch/journal2report.py` | 新規コピー＋修正 | ~32行 | 4 |
| `masist/treesearch/log_summarization.py` | 新規コピー＋修正 | ~450行 | 4 |
| `masist/treesearch/perform_experiments_with_agentmanager.py` | 新規コピー＋修正 | ~260行 | 5 |
| `tests/fixtures/experiments.py` | 修正 | ~500行 | 6 |
| `masist/treesearch/code_generator.py` | 確認・修正 | 少量 | 6 |
| `masist/treesearch/node_processor.py` | 確認・修正 | 少量 | 6 |
| `tests/test_*.py` | 修正 | 各ファイル少量 | 7 |
| `tests/test_agent_manager.py` | 新規作成 | ~200行 | 7 |

---

### 依存関係図

```
Phase 1 (依存なし - 並行可能)
    │
    ├── Task 1.1 serialize.py ─────────────────┐
    ├── Task 1.2 token_tracker.py ─────┐       │
    ├── Task 1.3 tree_export.py ───────┼───────┤
    └── Task 1.4 copytree ─────────────┼───────┤
                                       │       │
                                       ▼       ▼
Phase 2 (依存あり)                 Task 2.1  Task 2.2
    │                              llm.py   config.py
    │                                 │       │
    │   ┌─────────────────────────────┼───────┘
    │   │                             │
    ▼   ▼                             ▼
Phase 3 (コア)                    Phase 4 (レポート)
    │                                 │
    ├── Task 3.1 parallel_agent.py    ├── Task 4.1 journal2report.py
    │       │                         └── Task 4.2 log_summarization.py
    │       ▼                                 │
    └── Task 3.2 agent_manager.py             │
            │                                 │
            └─────────────┬───────────────────┘
                          │
                          ▼
Phase 5 (エントリーポイント)
    │
    └── Task 5.1-5.5 perform_experiments_with_agentmanager.py
            │
            │
Phase 6 (task_desc変換) ← Phase 3完了後に開始可能（Phase 4,5と並行）
    │
    ├── Task 6.1 experiments.py
    └── Task 6.2 関連ファイル確認
            │
            ▼
Phase 7 (テスト修正) ← Phase 5,6 完了後
    │
    ├── Task 7.1 既存テスト修正
    └── Task 7.2 新規テスト追加
            │
            ▼
Phase 8 (統合テスト) ← 全Phase完了後
    │
    └── Task 8.1 エンドツーエンドテスト
```

---

### 並行実行可能なタスクグループ

| グループ | タスク | 条件 |
|---------|--------|------|
| A | Task 1.1, 1.2, 1.3, 1.4 | 最初から並行可能 |
| B | Task 2.1, 2.2 | Phase 1完了後 |
| C | Task 3.1 → 3.2, Task 4.1, 4.2 | Phase 2完了後（3と4は並行可能） |
| D | Task 5.1-5.5, Task 6.1-6.2 | Phase 3完了後（5と6は並行可能） |
| E | Task 7.1, 7.2 | Phase 5,6完了後 |
| F | Task 8.1 | 全Phase完了後 |

---

### クリティカルパス（最短実行順序）

```
1.1 → 2.2 → 3.1 → 3.2 → 5.1-5.5 → 7.1 → 8.1
 ↓
1.2 → 2.1 → 4.1, 4.2 ──────────↗
```

**推定タスク数:** 24タスク（チェックボックス単位では約70項目）

---

## 4. 互換性の確認

| 項目 | 互換性 | 対応 |
|------|--------|------|
| Node/Journal | ✅ ほぼ同一 | そのまま使用可能 |
| backend (query) | ✅ 同一 | そのまま使用可能 |
| Interpreter | ✅ 同一 | そのまま使用可能 |
| FunctionSpec | ✅ 同一 | そのまま使用可能 |
| Config構造 | ⚠️ 差分あり | StagesConfig追加 |
| task_desc形式 | ❌ 異なる | パース処理追加 |

---

## 5. 最小変更での移植アプローチ

### オプションA: task_descをJSON形式に統一

- AI-Masist側でもJSON形式のtask_descを使用
- `agent_manager.py`の変更が最小限で済む

### オプションB: task_descのパース層を追加

- `_parse_simulation_sheet()`メソッドを追加
- 日本語シミュレーション検討シートをJSON形式に変換
- より柔軟だが追加実装が必要

**推奨: オプションA**（コピペ中心の場合）

---

## 6. ファイル変更サマリー

| ファイル | アクション | 変更量 |
|----------|-----------|--------|
| `agent_manager.py` | 新規コピー | ~1200行 |
| `config.py` | 追記 | ~15行 |
| `parallel_agent.py` | 修正 | ~30行追加 |
| `perform_experiments_*.py` | 新規コピー | ~260行 |

---

## 7. アーキテクチャ図解

### AI-Scientist-v2 (現状)

```
┌─────────────────────────────────────────────────────────────────┐
│ AgentManager                                                     │
│  ├── Stage 1: Initial Implementation                            │
│  │    ├── Sub-stage 1_1_preliminary                            │
│  │    └── Sub-stage 1_2_xxx (LLM生成)                          │
│  ├── Stage 2: Baseline Tuning                                   │
│  │    └── ...                                                   │
│  ├── Stage 3: Creative Research                                 │
│  │    └── ...                                                   │
│  └── Stage 4: Ablation Studies                                  │
│       └── ...                                                   │
│                                                                  │
│  journals: { "1_initial_impl_1_xxx": Journal, ... }             │
│  ParallelAgent (per stage)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### AI-Masist (現状)

```
┌─────────────────────────────────────────────────────────────────┐
│ ParallelAgent                                                    │
│  └── Stage 1のみ (run + step)                                   │
│                                                                  │
│  journal: 単一Journal                                           │
└─────────────────────────────────────────────────────────────────┘
```

### AI-Masist (移植後)

```
┌─────────────────────────────────────────────────────────────────┐
│ AgentManager (新規追加)                                          │
│  ├── Stage 1: Initial Implementation (基本動作確認)             │
│  ├── Stage 2: Scenario Tuning (シナリオ調整)                    │
│  ├── Stage 3: Hypothesis Exploration (仮説検証)                 │
│  └── Stage 4: Sensitivity Analysis (感度分析)                   │
│                                                                  │
│  journals: { stage_name: Journal }                              │
│  ParallelAgent (per stage)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 重要な依存関係

### AgentManagerが使用する主要コンポーネント

```
agent_manager.py
├── imports
│   ├── ParallelAgent (parallel_agent.py)
│   ├── Journal, Node (journal.py)
│   ├── query, FunctionSpec (backend/__init__.py)
│   └── WorstMetricValue (utils/metric.py)
│
├── FunctionSpec定義
│   ├── stage_config_spec
│   ├── stage_progress_eval_spec
│   └── stage_completion_eval_spec
│
└── クラス定義
    ├── Stage (dataclass)
    ├── StageTransition (dataclass)
    └── AgentManager (main class)
```

### AI-Masistで既に存在するコンポーネント

- ✅ `parallel_agent.py` - 拡張が必要
- ✅ `journal.py` - Node, Journal クラス
- ✅ `backend/__init__.py` - query, FunctionSpec
- ✅ `utils/metric.py` - MetricValue, WorstMetricValue
- ✅ `interpreter.py` - コード実行

---

## 9. 次のステップ

1. **確認**: この計画で良いか確認
2. **実装**: ファイルのコピーと変更を実施
3. **テスト**: 単体テストで動作確認
4. **統合**: 既存のテストケースで統合テスト
