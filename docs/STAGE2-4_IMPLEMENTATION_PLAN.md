# Stage 2-4 実装計画

## 現状の理解

AI Masist の現在の実装（Phase 1-6）は、AI Scientist v2 の **Stage 1: Initial Implementation** に相当する。
Stage 2-4 を実装することで、「1回の実験実行」から「研究全体の自動化」へと進化させる。

---

## 前提: AgentManager の実装

Stage 2-4 を動かすには、まず **AgentManager** が必要。

### 役割
- 4つのステージ（Initial → Tuning → Creative → Ablation）の順次実行
- 各ステージの完了条件判定とステージ移行
- 前ステージのベストノードを次ステージの起点として継承
- チェックポイント保存・復元

### 主要メソッド
- `run_all_stages()`: 全ステージを順次実行
- `run_stage(stage_name)`: 特定ステージを実行
- `should_progress(stage_name)`: ステージ移行判断
- `get_best_node(stage_name)`: ステージのベストノード取得

### 設定項目
```
stage1_max_iters: 20
stage2_max_iters: 12
stage3_max_iters: 12
stage4_max_iters: 18
```

---

## Stage 2: Baseline Tuning

### 目的
Stage 1 で動作した実装の**ハイパーパラメータを最適化**し、性能を向上させる。
シミュレーションの構造（エージェント数、ゲームルール等）は変更しない。

### AI Masist での「ハイパーパラメータ」
- LLMの温度（temperature）
- ラウンド数
- エージェントのシステムプロンプトの微調整
- グループサイズ
- 初期トークン数
- 閾値の設定

### 実装すべきコンポーネント

#### 1. HyperparamTuner クラス
**ファイル**: `masist/treesearch/hyperparam_tuner.py`

**メソッド**:
- `generate_tuning_idea(base_node)`: LLMに現在の実装を見せ、調整すべきハイパーパラメータを提案させる
- `generate_tuning_node(base_node, tuning_idea)`: 提案に基づいてコードを修正したノードを生成

#### 2. 試行履歴の追跡
- Journal または別の仕組みで、試行済みのハイパーパラメータ組み合わせを記録
- 同じ組み合わせの重複試行を防ぐ

#### 3. 複数条件での検証
- Stage 1 より多くの実験条件でテスト
- 「特定条件でだけ良い」チューニングを避ける

### 完了条件
- Stage 1 のベースラインより性能向上
- 複数条件で安定した結果
- max_iters 到達

---

## Stage 3: Creative Research

### 目的
ハイパーパラメータの調整を超えて、**創造的な改善やアルゴリズムの変更**を探索する。
新しい知見を得るための実験を行う。

### AI Masist での「創造的改善」の例
- エージェント間のコミュニケーション構造の変更
- 新しいペナルティ/報酬メカニズムの導入
- エージェントのメモリ・学習機能の追加
- 異なる意思決定戦略の実装
- グループダイナミクスの変更（リーダー役の導入等）
- 時間経過による行動変化のモデリング

### 実装すべきコンポーネント

#### 1. Improver クラス
**ファイル**: `masist/treesearch/improver.py`

**メソッド**:
- `improve(node)`: LLMに創造的な改善を提案・実装させる
- `evaluate_novelty(improvement_idea)`: 提案の新規性を評価（オプション）

#### 2. 改善履歴の管理
- どのような改善を試みたかを記録
- 類似の改善の重複を避ける

#### 3. 実行時間チェック
- シミュレーションが短すぎる場合、スケールアップを指示
- 例: ラウンド数を増やす、より複雑な条件を追加

#### 4. Best-First Tree Search の活用
- 複数の改善パスを並列探索
- 有望なブランチを優先的に深掘り

### 完了条件
- Stage 2 より性能向上または新しい知見の獲得
- 全条件でテスト完了
- max_iters 到達

---

## Stage 4: Ablation Studies

### 目的
Stage 3 で得られた最良の実装について、**各コンポーネントの貢献度を系統的に分析**する。
「何が効いているのか」を明らかにする。

### AI Masist での「アブレーション」の例
- 特定のルール説明を削除した場合の影響
- コミュニケーション機能を無効化した場合
- メモリ機能を削除した場合
- 特定のエージェントタイプを標準に置き換えた場合
- 報酬構造を単純化した場合

### 実装すべきコンポーネント

#### 1. AblationGenerator クラス
**ファイル**: `masist/treesearch/ablation_generator.py`

**メソッド**:
- `identify_components(node)`: LLMに実装の主要コンポーネントを特定させる
- `generate_ablation_idea(node, component)`: 特定コンポーネントを除外/無効化するアイデアを生成
- `generate_ablation_node(node, ablation_idea)`: アブレーション版のコードを生成

#### 2. 比較分析システム
- ベースライン（フル実装）との性能差を定量化
- 各コンポーネントの貢献度をランキング

#### 3. アブレーション結果サマリ生成
- 「コンポーネントXを除外すると性能がY%低下」のような分析
- 論文の Ablation Study セクションに使えるデータ

### 完了条件
- 主要コンポーネント全てのアブレーション完了
- 貢献度の定量的分析完了
- max_iters 到達

---

## 実装の優先順位

```
1. AgentManager（必須・最優先）
   └── ステージ管理の基盤がないと Stage 2-4 は動かない

2. Stage 2: HyperparamTuner
   └── 比較的シンプル。既存コードの修正のみ

3. Stage 3: Improver
   └── 創造的な生成が必要。プロンプトエンジニアリングが重要

4. Stage 4: AblationGenerator
   └── Stage 3 の結果に依存。最後に実装
```

---

## ファイル構成（予定）

```
masist/treesearch/
├── agent_manager.py      # 新規: ステージ管理
├── hyperparam_tuner.py   # 新規: Stage 2
├── improver.py           # 新規: Stage 3
├── ablation_generator.py # 新規: Stage 4
├── stage_config.py       # 新規: ステージ別設定
│
├── code_generator.py     # 既存: Stage 1 用（変更なし）
├── parallel_agent.py     # 既存: 拡張が必要
├── journal.py            # 既存: 拡張が必要（履歴追跡）
└── ...
```

---

## config.py への追加項目

```python
@dataclass
class StageConfig:
    stage1_max_iters: int = 20
    stage2_max_iters: int = 12
    stage3_max_iters: int = 12
    stage4_max_iters: int = 18

@dataclass
class TuningConfig:
    # Stage 2 用
    tunable_params: list  # チューニング可能なパラメータのリスト

@dataclass
class ImprovementConfig:
    # Stage 3 用
    min_execution_time: int = 300  # 最低実行時間（秒）

@dataclass
class AblationConfig:
    # Stage 4 用
    max_components: int = 10  # 分析する最大コンポーネント数
```

---

## 論文生成パイプライン（Stage 4 完了後）

Stage 2-4 完了後、以下のパイプラインを実装予定:

1. **結果集約**: 全ステージの結果をまとめる
2. **論文生成**: LLM による LaTeX 論文自動生成
3. **引用収集**: Semantic Scholar API で関連論文を検索・引用
4. **レビュー**: LLM/VLM による自己レビュー

これは Stage 2-4 の実装完了後に別途計画する。

---

## 次のステップ

1. この計画のレビューと承認
2. AgentManager の設計詳細化
3. AgentManager の実装
4. Stage 2 (HyperparamTuner) の実装
5. テストと検証
6. Stage 3, 4 の順次実装
