# Best Node 選択基準の再設計

## 問題の背景

### 現状（AI Scientist / ML文脈）
```
Best Node = argmin(validation_loss)
```
MLでは「損失が低い = 良い」という明確な基準がある。

### MAS（マルチエージェントシミュレーション）での課題

MASでは「良い」の定義が多義的：

| 観点 | 例 | 問題 |
|-----|---|------|
| 高い協力率 | cooperation_rate = 0.95 | 条件に関係なく常に高い → 実験として無意味 |
| 仮説通りの結果 | ルールありで協力↑ | 予想通りすぎて新規性がない |
| 予想外の結果 | ルールなしで協力↑ | バグかもしれない |
| 条件間の差 | 差が大きい | 差があれば良いわけではない |

**核心的な問い**: 科学実験として「良い」とは何か？

---

## 使える情報

### 1. 定量的データ

```python
node.metric = {
    "threshold_achievement_rate": float,  # 閾値達成率
    "average_contribution": float,        # 平均貢献度
    "cooperation_rate": float,            # 協力率
    "rule_compliance_rate": float,        # ルール遵守率
    ...
}
```

### 2. 条件間比較データ

```python
node.condition_results = {
    "FAIRSUFF": {...metrics...},
    "UNFAIRSUFF": {...metrics...},
    "NORULE": {...metrics...},
}
```

### 3. 統計情報（マルチシード実行時）

```python
node.seed_results = [
    {"seed": 42, "metrics": {...}},
    {"seed": 123, "metrics": {...}},
    ...
]
# → 平均、標準偏差、信頼区間が計算可能
```

### 4. VLM分析結果

```python
node.vlm_feedback_summary = "..."  # 図から読み取れるパターン
node.plot_analyses = [...]          # 各プロットの分析
```

### 5. エージェント会話ログ

```python
node.agent_interactions = [
    {"round": 1, "agent": "A", "message": "...", "action": 5},
    ...
]
```

### 6. 実行メタデータ

```python
node.exec_time: float      # 実行時間
node.is_buggy: bool        # バグ有無
node.exc_type: str         # 例外情報
```

### 7. 実験設計情報

```python
experiment_config = {
    "hypothesis": "ルールが明確なほど協力率が上がる",
    "expected_ordering": ["NORULE < UNFAIR < FAIR"],
    "key_conditions": ["FAIRSUFF", "NORULE"],
    ...
}
```

---

## 候補となる選択基準

### 基準 A: 仮説検証度 (Hypothesis Validation Score)

**考え方**: 実験の目的は仮説を検証すること。仮説と結果の整合性を評価。

```
Score = 仮説で予測された条件間の順序と実際の順序の一致度
```

**例**:
- 仮説: "FAIR > UNFAIR > NORULE（協力率）"
- 結果: FAIR=0.8, UNFAIR=0.6, NORULE=0.4 → 高スコア
- 結果: NORULE=0.9, FAIR=0.5, UNFAIR=0.3 → 低スコア（ただし興味深い）

**長所**: 科学的目的に直結
**短所**: 予想外の発見を見逃す

---

### 基準 B: 条件間分離度 (Condition Separability Score)

**考え方**: 条件間で明確な差がある = 実験として成功

```
Score = Σ |metric(condition_i) - metric(condition_j)| / variance
      = 効果量（Cohen's d など）の総和
```

**例**:
- 全条件で協力率0.7±0.05 → 低スコア（差がない）
- 条件で0.3〜0.9に分布 → 高スコア（明確な差）

**長所**: 統計的に意味のある結果を優先
**短所**: 差があれば良いわけではない（ランダムノイズも差を生む）

---

### 基準 C: 再現性スコア (Reproducibility Score)

**考え方**: 複数シードで安定した結果 = 信頼できる

```
Score = 1 / (average_std_across_seeds)
```

**例**:
- シード間でメトリクスの標準偏差が小さい → 高スコア
- シードによって結果がバラバラ → 低スコア

**長所**: 信頼性の高い結果を優先
**短所**: 本質的にばらつきのある現象を見逃す

---

### 基準 D: 説明可能性スコア (Explainability Score)

**考え方**: 結果が「なぜそうなったか」説明できる = 良い実験

```
LLMに以下を評価させる:
- 結果とエージェント行動の因果関係が明確か
- 会話ログから行動の理由が読み取れるか
- 予想外の結果に合理的な説明があるか
```

**例**:
- 「エージェントAがルールを引用して貢献を増やした」→ 説明可能
- 「なぜか急に協力率が上がった」→ 説明不可

**長所**: 科学的に価値のある知見を抽出
**短所**: LLM評価のコストと主観性

---

### 基準 E: 新規性・発見スコア (Discovery Score)

**考え方**: 予想外だが興味深い結果 = 科学的価値

```
Score = f(予想との乖離度, 説明可能性, 再現性)

- 予想通りで説明可能 → 普通（確認）
- 予想外で説明不可で再現しない → バグか偶然
- 予想外で説明可能で再現する → 発見！
```

**長所**: 真に価値のある発見を見つける
**短所**: 評価が複雑

---

### 基準 F: 論文化可能性スコア (Publishability Score)

**考え方**: 最終目的は論文。論文になりうる結果か？

```
LLMに以下を評価させる:
- この結果で論文のContributionを書けるか
- 先行研究と比較して新規性があるか
- 結果が社会科学的に意味のある示唆を持つか
```

**長所**: 最終目的に直結
**短所**: 高コスト、主観的

---

## 推奨: 複合スコアリングシステム

### 設計方針

単一基準ではなく、複数基準の重み付け合計を使用。
重みは実験の目的に応じて調整可能。

### スコア計算式

```python
final_score = (
    w1 * validity_score +        # 実行が正常に完了したか
    w2 * hypothesis_score +      # 仮説との整合性
    w3 * separation_score +      # 条件間の分離度
    w4 * reproducibility_score + # 再現性
    w5 * explainability_score +  # 説明可能性
    w6 * discovery_score         # 新規性・発見
)
```

### デフォルト重み（探索的研究向け）

```python
default_weights = {
    "validity": 0.20,        # 最低限の品質
    "hypothesis": 0.15,      # 仮説検証も重要だが絶対ではない
    "separation": 0.20,      # 条件間の差は重要
    "reproducibility": 0.20, # 再現性は科学の基本
    "explainability": 0.15,  # 説明できることは重要
    "discovery": 0.10,       # 発見は嬉しいがリスクもある
}
```

### 確認的研究向けの重み

```python
confirmatory_weights = {
    "validity": 0.15,
    "hypothesis": 0.35,      # 仮説検証が主目的
    "separation": 0.20,
    "reproducibility": 0.20,
    "explainability": 0.05,
    "discovery": 0.05,
}
```

---

## 実装アプローチ

### 方法 1: ルールベース計算

各スコアを数式で計算。高速だが柔軟性に欠ける。

```python
class NodeScorer:
    def score_hypothesis(self, node, experiment_config):
        expected = experiment_config["expected_ordering"]
        actual = self.get_actual_ordering(node)
        return kendall_tau(expected, actual)

    def score_separation(self, node):
        metrics = node.condition_results
        return sum(effect_size(m1, m2) for m1, m2 in combinations(metrics))

    def score_reproducibility(self, node):
        if not node.seed_results:
            return 0.5  # 不明
        stds = [np.std(seed["metrics"]) for seed in node.seed_results]
        return 1 / (1 + np.mean(stds))
```

### 方法 2: LLM評価

LLMに総合判断させる。柔軟だがコストが高い。

```python
def llm_score_node(node, experiment_config):
    prompt = f"""
    以下の実験結果を評価してください。

    実験目的: {experiment_config["hypothesis"]}

    結果:
    {format_results(node)}

    以下の観点で1-10で評価し、総合スコアを出してください:
    1. 仮説との整合性
    2. 条件間の差の明確さ
    3. 結果の説明可能性
    4. 科学的な新規性
    5. 論文化の可能性

    JSON形式で出力:
    {{
        "scores": {{...}},
        "reasoning": "...",
        "overall_score": float
    }}
    """
    return query(prompt)
```

### 方法 3: ハイブリッド（推奨）

- 定量的スコア（validity, separation, reproducibility）はルールベース
- 定性的スコア（explainability, discovery）はLLM評価
- 最終統合もLLMで行い、理由を説明させる

```python
class HybridNodeScorer:
    def score(self, node, experiment_config):
        # ルールベースのスコア（高速）
        quantitative = {
            "validity": self.calc_validity(node),
            "separation": self.calc_separation(node),
            "reproducibility": self.calc_reproducibility(node),
        }

        # LLMベースのスコア（詳細な評価が必要な時のみ）
        qualitative = self.llm_evaluate(node, experiment_config)

        # 統合
        return self.combine_scores(quantitative, qualitative)
```

---

## 追加の考慮事項

### ステージごとの基準の違い

| ステージ | 重視すべき基準 | 理由 |
|---------|--------------|------|
| Stage 1 | validity, reproducibility | まず動くことが重要 |
| Stage 2 | separation, hypothesis | チューニングで差を明確に |
| Stage 3 | discovery, explainability | 創造的な発見を促進 |
| Stage 4 | hypothesis, explainability | コンポーネントの貢献を明確に |

### 負のメトリクスの扱い

「悪い結果」も科学的には価値がある場合がある:
- 仮説が棄却された → 仮説の修正につながる
- 全条件で同じ結果 → その変数は影響しないという知見

→ 「結果が悪い」と「実験が悪い」を区別する必要

### マルチシード評価との統合

```
1. 各シードでノードを実行
2. シードごとのスコアを計算
3. スコアの平均と分散を評価
4. 分散が大きすぎる → 信頼性低下
5. 分散が適度 → 頑健な結果
```

---

## 結論

### MASにおける「良いノード」の定義

```
良いノード =
    科学的に意味のある結果を
    再現可能な形で
    説明可能に
    生成したノード
```

### 推奨する実装順序

1. **Phase 1**: validity + separation のルールベーススコア（最小実装）
2. **Phase 2**: reproducibility の追加（マルチシード対応後）
3. **Phase 3**: LLMによる explainability 評価の追加
4. **Phase 4**: discovery + 論文化可能性の評価（論文生成との統合後）

### 設定可能にすべきパラメータ

```yaml
node_selection:
  method: "hybrid"  # "rule_based", "llm", "hybrid"
  weights:
    validity: 0.20
    hypothesis: 0.15
    separation: 0.20
    reproducibility: 0.20
    explainability: 0.15
    discovery: 0.10
  llm_evaluation:
    enabled: true
    model: "gpt-4o"
    temperature: 0.3
```

---

## 未解決の問い

1. **探索 vs 確認**: 探索的研究と確認的研究で基準を変えるべきか？
2. **発見の評価**: 「予想外だが価値のある発見」をどう自動検出するか？
3. **人間の介入**: 完全自動化 vs 人間によるキュレーションの境界は？
4. **メタ学習**: 過去の「良い実験」から基準を学習できるか？
