# CLAUD.md

## プロジェクト概要

**unitTest**は、オリジナル実装への段階的変更を行う際の単体テスト用フォルダです。

## 実行環境

**必須**: conda環境 `masist` を使用すること

```bash
conda activate masist
```

すべてのテスト実行は必ずこの環境で行ってください。

## 変更履歴

### MASist向けプロンプト変更（最新）

**変更対象**: `phases/code_generator.py`, `phases/metrics_extractor.py`, `phases/plot_generator.py`, `example_usage.py`

**変更内容**:
- コード生成プロンプトをMASist（マルチエージェントシミュレーション）向けに変更
- Autogen フレームワークを必須使用
- experiment_data構造を教師あり学習（train/val/test）からシミュレーション（scenario/runs/metrics）に変更
- 評価メトリクスをMAS向けに調整（協力率、合意達成率、ターン数など）

**バックアップ**: 元の`phases`ディレクトリは`phases_original`に保存
