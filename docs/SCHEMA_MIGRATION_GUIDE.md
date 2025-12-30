# スキーマ移行ガイド: perform_ideation.py → agent_manager.py

## 概要

`perform_ideation.py` の出力スキーマと `agent_manager.py` の入力スキーマの不整合を解決し、
**生成された全情報を活用する**ための移行ガイド。

---

## 1. 設計方針

### 従来の問題

`perform_ideation.py` が生成する詳細なシミュレーション設計情報の約70%が未使用のまま捨てられていた。

### 新方針：全情報の活用

| agent_manager.py のキー | マッピング元 | 内容 |
|------------------------|-------------|------|
| `Title` | `Title` | そのまま |
| `Abstract` | `Abstract` | そのまま |
| `Short Hypothesis` | **`SimulationRequest` 全体** | 背景・目的・研究質問・仮説・関連研究 |
| `Experiments` | **`SimulationRequirements` 全体** | エージェント・環境・プロトコル・ルール・ログ仕様 |
| `Risk Factors and Limitations` | `RiskFactorsAndLimitations` | キー名変換のみ |

---

## 2. 修正対象ファイル

| ファイル | 必須度 | 修正内容 |
|---------|-------|---------|
| `masist/treesearch/agent_manager.py` | **必須** | スキーマ正規化メソッド追加 |
| `masist/treesearch/bfts_utils.py` | **推奨** | 再帰的Markdown変換 |

---

## 3. agent_manager.py 修正内容

### 修正箇所: `__init__` メソッド（125行目〜）

```python
class AgentManager:
    def __init__(self, task_desc: str, cfg: Any, workspace_dir: Path):
        self.task_desc = json.loads(task_desc)

        # === NEW: perform_ideation.py スキーマの正規化 ===
        self._normalize_ideation_schema()

        # Key mapping for compatibility (MASIST format -> AI-Scientist-v2 format)
        if "Short Hypothesis" not in self.task_desc and "Hypothesis" in self.task_desc:
            self.task_desc["Short Hypothesis"] = self.task_desc["Hypothesis"]
        if "Experiments" not in self.task_desc and "Experimental Conditions" in self.task_desc:
            self.task_desc["Experiments"] = self.task_desc["Experimental Conditions"]

        for k in [
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Experiments",
            "Risk Factors and Limitations",
        ]:
            if k not in self.task_desc.keys():
                raise ValueError(f"Key {k} not found in task_desc")

        # ... (残りは変更なし)
```

### 新規メソッド: `_normalize_ideation_schema()`

```python
def _normalize_ideation_schema(self):
    """
    perform_ideation.py のネスト構造を変換し、全情報を保持して渡す

    マッピング:
      - SimulationRequest.Abstract → Abstract (トップレベルにない場合)
      - SimulationRequest 全体 → Short Hypothesis (JSON文字列)
      - SimulationRequirements 全体 → Experiments (dict)
      - RiskFactorsAndLimitations → Risk Factors and Limitations
    """
    td = self.task_desc

    # Abstract が SimulationRequest 内にある場合は抽出
    if "Abstract" not in td and "SimulationRequest" in td:
        sim_req = td["SimulationRequest"]
        if "Abstract" in sim_req:
            td["Abstract"] = sim_req["Abstract"]

    # SimulationRequest 全体 → Short Hypothesis（JSON文字列化）
    # 背景・目的・研究質問・仮説・関連研究がすべて含まれる
    if "SimulationRequest" in td and "Short Hypothesis" not in td:
        sim_req = td["SimulationRequest"]
        td["Short Hypothesis"] = json.dumps(sim_req, indent=2, ensure_ascii=False)

    # SimulationRequirements 全体 → Experiments（dictのまま）
    # エージェント・環境・プロトコル・ルール・ログ仕様がすべて含まれる
    if "SimulationRequirements" in td and "Experiments" not in td:
        td["Experiments"] = td["SimulationRequirements"]

    # RiskFactorsAndLimitations → Risk Factors and Limitations（キー名変換）
    if "RiskFactorsAndLimitations" in td and "Risk Factors and Limitations" not in td:
        td["Risk Factors and Limitations"] = td["RiskFactorsAndLimitations"]
```

---

## 4. bfts_utils.py 修正内容

### 問題

現在の `idea_to_markdown()` は1層のネストしか対応しておらず、
2層以上のネスト（例: `SimulationRequirements.Agents`）が Python repr 形式で出力される。

```markdown
### Agents
{'Count': '4', 'RolesAndDescriptions': '...'}  ← 読みにくい
```

### 修正: 再帰的なMarkdown変換

```python
def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    Convert a dictionary into a markdown file.
    Supports nested structures with recursive formatting.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """

    def write_value(f, value, indent_level=0):
        """Recursively write values with proper formatting"""
        indent = "  " * indent_level

        if isinstance(value, str):
            # 複数行の文字列は適切にインデント
            lines = value.split('\n')
            for line in lines:
                f.write(f"{indent}{line}\n")
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    f.write(f"{indent}- \n")
                    write_value(f, item, indent_level + 1)
                elif isinstance(item, (list, tuple)):
                    f.write(f"{indent}- \n")
                    write_value(f, item, indent_level + 1)
                else:
                    f.write(f"{indent}- {item}\n")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list, tuple)):
                    f.write(f"{indent}**{sub_key}**:\n")
                    write_value(f, sub_value, indent_level + 1)
                else:
                    f.write(f"{indent}**{sub_key}**: {sub_value}\n")
        else:
            f.write(f"{indent}{value}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        f.write("- \n")
                        write_value(f, item, 1)
                    else:
                        f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n\n")
                    write_value(f, sub_value, 0)
                    f.write("\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            assert os.path.exists(load_code), f"Code path at {load_code} must exist"
            f.write("## Code To Potentially Use\n\n")
            f.write("Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")
```

### 修正後の出力例

```markdown
## Simulationrequirements

### Agents

**Count**: 4
**RolesAndDescriptions**: Role descriptions...
**StateSpec**: State spec...

### Environment

**Structure**: Network topology
**StateSpec**: Env state...

### Rules

**SharedInformation**: Public info
**ExperimentConditions**:
  - Cond A
  - Cond B
```

---

## 5. データフロー図（修正後）

```
┌─────────────────────────────────────────────────────────────────┐
│ perform_ideation.py                                             │
│   生成: 詳細なシミュレーション設計（全情報）                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ launch_masist.py                                                │
│   1. idea_to_markdown() → idea.md（再帰的変換で可読性向上）      │
│   2. json.dump() → idea.json                                    │
│   3. config.desc_file = idea.json                              │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────────┐
│ idea.json               │    │ idea.md                         │
│ (実験実行用)             │    │ (論文・プロット用)               │
└──────────┬──────────────┘    └──────────┬──────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────────┐
│ AgentManager            │    │ perform_writeup.py              │
│ _normalize_ideation_    │    │ perform_plotting.py             │
│ schema()                │    │                                 │
│                         │    │ 可読性の高いMarkdownで           │
│ SimulationRequest       │    │ LLMプロンプト品質向上            │
│   → Short Hypothesis    │    │                                 │
│ SimulationRequirements  │    │                                 │
│   → Experiments         │    │                                 │
└─────────────────────────┘    └─────────────────────────────────┘
```

---

## 6. 後方互換性

### 既存フォーマット（multi_agent_simulation.json）

```json
{
  "Title": "...",
  "Abstract": "...",
  "Hypothesis": "...",
  "Experimental Conditions": [],
  "Risk Factors and Limitations": "..."
}
```

**既存のキーマッピング（127-130行目）により引き続き動作する。**

### 新フォーマット（pyao.json）

```json
{
  "Title": "...",
  "Abstract": "...",
  "SimulationRequest": {...},
  "SimulationRequirements": {...},
  "RiskFactorsAndLimitations": []
}
```

**新しい `_normalize_ideation_schema()` により動作する。**

---

## 7. 実装手順

### Step 1: agent_manager.py の修正

1. `_normalize_ideation_schema()` メソッドを追加
2. `__init__` で `json.loads()` の直後に呼び出し

### Step 2: bfts_utils.py の修正

1. `idea_to_markdown()` を再帰的実装に置き換え

### Step 3: テスト

```bash
# 新フォーマットのテスト
python launch_masist.py --load_ideas masist/ideas/pyao.json --skip_experiments --skip_writeup

# 後方互換性テスト
python launch_masist.py --load_ideas masist/ideas/multi_agent_simulation.json --idea_idx 0 --skip_experiments --skip_writeup
```

---

## 8. 変更前後の比較

### Before: 情報の大部分が未使用 + Markdown可読性低

```
perform_ideation.py → agent_manager.py
───────────────────────────────────────
SimulationRequest      → (大部分破棄)     ✗
SimulationRequirements → (大部分破棄)     ✗

idea.md 出力:
### Agents
{'Count': '4', ...}  ← Python repr（読みにくい）
```

### After: 全情報を活用 + Markdown可読性向上

```
perform_ideation.py → agent_manager.py
───────────────────────────────────────
SimulationRequest      → Short Hypothesis  ✓ 全体をJSON化
SimulationRequirements → Experiments       ✓ 全体をdict

idea.md 出力:
### Agents

**Count**: 4
**RolesAndDescriptions**: ...  ← 構造化（読みやすい）
```

---

## 関連ファイル

| ファイル | 役割 | 修正 |
|---------|------|------|
| `masist/perform_ideation.py` | アイデア生成 | なし |
| `masist/treesearch/agent_manager.py` | 実験管理 | **要修正** |
| `masist/treesearch/bfts_utils.py` | Markdown変換 | **要修正** |
| `masist/treesearch/code_generator.py` | コード生成 | なし |
| `launch_masist.py` | 統合ランチャー | なし |
