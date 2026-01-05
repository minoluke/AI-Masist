# LLM-as-a-Judge 実装リファレンス
## 定性データの定量化ガイド

このドキュメントは、LLMマルチエージェントシミュレーションにおいて、
会話ログや行動履歴などの定性データを、LLMを用いて定量的なスコアに変換する方法を説明する。

---

## 目次

1. [LLM-as-a-Judgeとは](#1-llm-as-a-judgeとは)
2. [ルーブリックの設計](#2-ルーブリックの設計)
3. [実装パターン](#3-実装パターン)
4. [metricsへの保存](#4-metricsへの保存)
5. [注意点・ベストプラクティス](#5-注意点ベストプラクティス)

---

## 1. LLM-as-a-Judgeとは

### 概念

LLM-as-a-Judge は、人間の評価者の代わりに LLM を使って
テキストデータを評価・採点する手法である。

**典型的なユースケース：**
- 会話ログから「協調の質」を評価
- 議論の「深さ」や「建設性」を採点
- エージェントの「社会性」「危険性」を判定
- 計画や提案の「有効性」を評価

### なぜ必要か

シミュレーション中に直接数値化できない指標がある場合、
実験後ではなく **実験中に** LLM を呼び出して評価・採点することで、
結果を metrics として保存できる。

---

## 2. ルーブリックの設計

### ルーブリックとは

評価基準を段階的に記述したもの。
評価者（この場合はLLM）が一貫した判定を行うための指針。

### 設計手順

1. **評価軸を決める**: 何を測りたいのか明確にする
2. **段階数を決める**: 2段階または3段階（下表参照）
3. **各段階を具体的に記述**: 曖昧な表現を避ける

### 段階数の選び方

| 段階数 | メリット | 適したケース |
|--------|----------|--------------|
| 2段階 | 明確・ブレが最小・判定が速い | Yes/No判定、閾値判定（「〇〇が見られたか」） |
| 3段階 | バランスが良い、中間状態も表現可能 | 程度を測りたい場合（「どの程度協力的か」） |
| 5段階以上 | 細かい差を表現できる | 細かいニュアンスが必要な場合（ブレやすくなる点に注意） |

**基本方針**: 2〜3段階が無難。5段階以上も使えるが、判定のブレに注意。

### ルーブリック記述のテンプレート

**2段階の場合（Yes/No判定向け）:**
```
【評価軸】: [評価軸の名前]
【定義】: [何を測っているのか]

【0 - No】: [該当しない場合の具体的な記述]
【1 - Yes】: [該当する場合の具体的な記述]
```

**3段階の場合（程度を測る場合）:**
```
【評価軸】: [評価軸の名前]
【定義】: [何を測っているのか]

【1 - 低】: [低い状態の具体的な記述。観察可能な行動・発言の特徴]
【2 - 中】: [中程度の状態の具体的な記述]
【3 - 高】: [高い状態の具体的な記述]
```

---

## 3. 実装パターン

### 基本パターン

```python
import json
from openai import OpenAI

def evaluate_with_llm_judge(
    messages: list,
    evaluation_axis: str,
    rubric: str,
    llm_config: dict,
    score_range: str = "0-1 or 1-3"  # ルーブリックに応じて指定
) -> dict:
    """
    LLM-as-a-Judge で定性評価を行う

    Args:
        messages: 評価対象の会話ログ
        evaluation_axis: 評価軸の名前（例: "cooperation_quality"）
        rubric: ルーブリック（評価基準の記述）
        llm_config: LLM設定（既存のllm_configを再利用）
        score_range: スコアの範囲（2段階なら"0-1"、3段階なら"1-3"など）

    Returns:
        {"score": <int>, "evidence": "判定理由"}
    """

    # 会話ログを文字列に変換
    if isinstance(messages, list):
        messages_text = "\n".join([
            f"{m.get('name', 'Unknown')}: {m.get('content', '')}"
            for m in messages if isinstance(m, dict)
        ])
    else:
        messages_text = str(messages)

    prompt = f"""あなたは評価者です。以下の会話ログを、指定されたルーブリックに基づいて評価してください。

【評価軸】
{evaluation_axis}

【ルーブリック（評価基準）】
{rubric}

【評価対象の会話ログ】
{messages_text}

【出力形式】
以下のJSON形式で出力してください。他のテキストは含めないでください。
scoreはルーブリックで定義された段階の数値を使用してください。
{{"score": <ルーブリックに基づく数値>, "evidence": "<この判定に至った具体的な根拠>"}}
"""

    client = OpenAI(
        api_key=llm_config["config_list"][0]["api_key"],
        base_url=llm_config["config_list"][0].get("base_url")
    )

    response = client.chat.completions.create(
        model=llm_config["config_list"][0]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 評価の一貫性のため低温推奨
    )

    result_text = response.choices[0].message.content.strip()

    # JSONパース（エラーハンドリング付き）
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # JSON抽出を試みる
        import re
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"score": 0, "evidence": f"Parse error: {result_text}"}

    return result
```

### シミュレーション内での使用例

**2段階評価の例（Yes/No判定）:**
```python
# 合意形成の有無を判定
consensus_rubric = """
【0 - No】: 議論が平行線のまま終了。明確な合意点がない。
【1 - Yes】: 明確な合意点がある。全員または多数が同一の結論に達した。
"""

judge_result = evaluate_with_llm_judge(
    messages=scenario_messages,
    evaluation_axis="consensus_reached",
    rubric=consensus_rubric,
    llm_config=llm_config
)

metrics["consensus_reached"] = judge_result["score"]  # 0 or 1
metrics["consensus_evidence"] = judge_result["evidence"]
```

**3段階評価の例（程度を測る）:**
```python
# 協調の質を評価
cooperation_rubric = """
【1 - 低】: 協力的な発言がほとんどない。自己利益のみを追求。
【2 - 中】: 部分的に協力的。条件付きで合意する姿勢が見られる。
【3 - 高】: 積極的に協力。他者の利益も考慮した提案を行う。
"""

judge_result = evaluate_with_llm_judge(
    messages=scenario_messages,
    evaluation_axis="cooperation_quality",
    rubric=cooperation_rubric,
    llm_config=llm_config
)

metrics["cooperation_quality_score"] = judge_result["score"]  # 1, 2, or 3
metrics["cooperation_quality_evidence"] = judge_result["evidence"]
```

### 複数の評価軸を持つ場合

```python
# 評価軸のリストを定義（2段階と3段階を混在可能）
evaluation_axes = [
    {
        "name": "consensus_reached",
        "rubric": """
【0 - No】: 明確な合意点がない。
【1 - Yes】: 明確な合意に達した。
"""
    },
    {
        "name": "cooperation_quality",
        "rubric": """
【1 - 低】: 協力的な発言がほとんどない。
【2 - 中】: 部分的に協力的。
【3 - 高】: 積極的に協力。
"""
    },
    {
        "name": "argument_depth",
        "rubric": """
【1 - 低】: 表面的な主張のみ。根拠が示されない。
【2 - 中】: 一定の根拠を伴う主張がある。
【3 - 高】: 論理的で深い議論。複数の観点から検討。
"""
    },
]

# 各評価軸について評価を実行
for axis in evaluation_axes:
    result = evaluate_with_llm_judge(
        messages=scenario_messages,
        evaluation_axis=axis["name"],
        rubric=axis["rubric"],
        llm_config=llm_config
    )
    metrics[f"{axis['name']}_score"] = result["score"]
    metrics[f"{axis['name']}_evidence"] = result["evidence"]
```

---

## 4. metricsへの保存

### 命名規則

```python
# スコア: [評価軸名]_score
metrics["cooperation_quality_score"] = 3

# 根拠: [評価軸名]_evidence
metrics["cooperation_quality_evidence"] = "エージェントAがエージェントBの提案を受け入れ..."
```

### experiment_data構造への統合

```python
experiment_data = {
    'scenarios': {
        'CONDITION_A': {
            'messages': [...],
            'events': [...],
            'metrics': {
                # 数値指標（従来どおり）
                'cooperation_rate': 0.85,
                'rounds_to_agreement': 5,

                # LLM-as-a-Judge による評価（追加）
                # 2段階評価の例
                'consensus_reached_score': 1,        # 0 or 1
                'consensus_reached_evidence': '...',
                # 3段階評価の例
                'cooperation_quality_score': 3,      # 1, 2, or 3
                'cooperation_quality_evidence': '...',
            },
            'config': {...},
        },
    },
}
```

---

## 5. 注意点・ベストプラクティス

### 評価の信頼性

| 対策 | 説明 |
|------|------|
| temperature=0.0 | 評価の一貫性を高める |
| 2〜3段階評価 | LLMの判定ブレを抑える（5段階以上は避けるのが望ましい） |
| 具体的なルーブリック | 曖昧さを減らす |
| evidence必須 | 判定根拠を残して検証可能にする |

### コスト・速度のトレードオフ

- 評価呼び出しは追加のAPI呼び出しになる
- 評価軸が多いほどコストが増加
- 必要最小限の評価軸に絞ることを推奨

### ルーブリック設計のコツ

1. **観察可能な行動・発言で記述**: 「協力的である」ではなく「他者の提案を受け入れる発言がある」
2. **各段階を相互排他的に**: 1と2、2と3の境界を明確に
3. **具体例を含める**: 可能であれば典型的な発言例を記載

### エラーハンドリング

```python
# 評価失敗時のフォールバック
try:
    result = evaluate_with_llm_judge(...)
    metrics["score"] = result["score"]
except Exception as e:
    # 評価失敗をログに残し、欠損値として記録
    print(f"LLM-as-a-Judge evaluation failed: {e}")
    metrics["score"] = None
    metrics["evaluation_error"] = str(e)
```
