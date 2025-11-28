# Multi-Seed Evaluation 問題調査レポート

## 概要

`_run_multi_seed_evaluation` で `data_paths = []` が空になる問題の原因調査結果。

---

## 問題の症状

- Test 8 実行時、3つの seed_nodes のうち成功したものがない
- `aggregation_code.py` の `data_paths = []` が空
- aggregation プロットが生成されない

---

## 根本原因

### AI-Masist の実装（不完全）

```python
# parallel_agent.py _run_multi_seed_evaluation()
node_data = node.to_dict()
node_data["code"] = seed_code  # seed付きコードを設定

futures.append(
    self.executor.submit(
        process_node_wrapper,
        node_data,  # ← これを渡す
        ...
    )
)
```

```python
# node_processor.py process_node_wrapper()
parent_node = Node.from_dict(node_data, journal=None)

if parent_node is None:
    child_node = code_generator.generate()        # Draft
elif parent_node.is_buggy:
    child_node = code_generator.generate_debug()  # Debug
else:
    child_node = code_generator.generate_improve() # Improve ← これが実行される！
```

**問題**: `parent_node.is_buggy = False` なので `generate_improve()` が呼ばれ、**seed_code は無視されて新しいコードが生成される**。

---

### AI-Scientist-v2 の実装（正しい）

```python
# AI-Scientist-v2
futures.append(
    self.executor.submit(
        self._process_node_wrapper,
        node_data,
        ...
        seed_eval=True,  # ← seed評価フラグ
    )
)
```

```python
# AI-Scientist-v2 _process_node_wrapper()
if seed_eval:
    # Phase 1 をスキップし、親ノードのコードをそのまま使う
    child_node = worker_agent._generate_seed_node(parent_node)
    child_node.plot_code = parent_node.plot_code
else:
    # 通常フロー
    if parent_node is None:
        child_node = worker_agent._draft()
    elif parent_node.is_buggy:
        child_node = worker_agent._debug(parent_node)
    else:
        child_node = worker_agent._improve(parent_node)
```

```python
def _generate_seed_node(self, parent_node: Node):
    return Node(
        plan="Seed node",
        code=parent_node.code,
        parent=parent_node,
        is_seed_node=True,
    )
```

---

## AI-Masist に不足している機能

| 機能 | AI-Scientist-v2 | AI-Masist |
|-----|----------------|-----------|
| `seed_eval` パラメータ | ✓ | ✗ |
| `_generate_seed_node()` メソッド | ✓ | ✗ |
| Phase 1 スキップ機能 | ✓ | ✗ |

---

## 備考

- seed を変えても LLM の応答は非決定的なので、純粋な再現性は得られない
- 同じコードを複数回実行すること自体に意味がある（統計的評価）
- seed 機能は Python/NumPy の乱数のみ制御、LLM は制御不能

---

## 修正方針

### オプション A: AI-Scientist-v2 準拠

`process_node_wrapper` に `seed_eval` パラメータを追加し、`seed_eval=True` の場合は Phase 1 をスキップする。

**変更箇所**:
1. `node_processor.py`: `process_node_wrapper()` に `seed_eval` パラメータ追加
2. `parallel_agent.py`: `_run_multi_seed_evaluation()` で `seed_eval=True` を渡す

### オプション B: シンプルな実装

seed評価専用の軽量な関数を作成し、Phase 2-6 のみ実行する。

**変更箇所**:
1. `node_processor.py`: `process_seed_node_wrapper()` を新規作成
2. `parallel_agent.py`: `_run_multi_seed_evaluation()` で新関数を使用

---

## 関連ファイル

- `/masist/treesearch/parallel_agent.py` - `_run_multi_seed_evaluation()`
- `/masist/treesearch/node_processor.py` - `process_node_wrapper()`
- `/masist/treesearch/journal.py` - `Node` クラス

---

## 作成日

2025-11-28
