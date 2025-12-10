# AI-MASist

## 概要

AI-MASistは、大規模言語モデル（LLM）を用いたマルチエージェントシミュレーション（MAS）の研究基盤です。自然言語によるシミュレーション設計から、実験実行、結果分析、論文執筆までを一貫して行います。


## 背景と目的

### マルチエージェントシミュレーション（MAS）とは

マルチエージェントシミュレーション（MAS）は、複数の自律的なエージェントが相互作用する系をシミュレートする手法です。各エージェントは独自のルールや目標を持ち、他のエージェントや環境と相互作用しながら行動します。個々のエージェントの振る舞いから、集団レベルでの創発的な現象（市場の価格形成、交通渋滞、社会規範の形成など）を観察・分析できます。
![MAS](./imgs/sim.png)

### 将来のビジョン

5-10年先、以下のような領域でMASによる将来予測が主流になると予想されます：

- **消費者行動予測**: 市場動向、購買パターンの予測
- **軍事シミュレーション**: 戦略的意思決定の分析
- **金融市場分析**: 株式・為替の動向予測
- **社会現象の理解**: 世論形成、情報拡散のメカニズム解明

![MASによる将来予測のイメージ](./imgs/sim2.jpg)

### 現在のフェーズ

現在のAIによるMASは、**LLMの知能レベルでどの程度MASとして機能するかを原理的に調査する段階**にあります。

- シンプルな系での検証（囚人のジレンマ、公共財ゲーム等）
- LLMエージェントの協力・裏切り行動の観察
- 人間の実験データとの比較
- LLMの行動特性の多面的理解

### 研究課題

人間という**非合理的な意思主体**をLLMがどの程度モデリングできているかは、原理的に検証可能な部分に限界があります。そのため、本研究では以下のアプローチを取ります：

1. **シンプルな系での振る舞い検証**: ゲーム理論的設定等での行動分析
2. **多面的な理解**: 協力、競争、学習、適応など多角的な観察
3. **手軽な実行環境**: 自然言語をインプットとしてLLMによるMASを素早く実行

### 多様なアウトプット形式

MASistは論文生成だけでなく、シミュレーション結果をより直感的に伝える多様な出力形式を目指しています：

- **動画生成**: エージェントの相互作用をアニメーションで可視化し、複雑なダイナミクスを一目で理解可能に
- **インタラクティブ可視化**: パラメータを変えながらリアルタイムでシミュレーションの変化を探索
- **記事・レポート生成**: 専門家でなくても理解できる簡潔なサマリーを自動生成

「問いを投げかけるだけで、AIが仮説を立て、シミュレーションを実行し、結果を動画や記事で届けてくれる」—そんな未来の研究スタイルを実現します。

---

## 主要機能

### 1. アイデア生成 (`perform_ideation.py`)

自然言語のトピック記述から、MASシミュレーションの研究アイデアを自動生成します。

- Semantic Scholar APIによる文献検索
- 複数ラウンドのリフレクションによるアイデア精緻化
- MASシミュレーション検討シート形式での出力

### 2. 実験実行 (`treesearch/`)

LLMエージェントによるシミュレーション実験を自動実行します。

- 並列エージェント実行
- 木探索によるコード改善
- 結果の自動評価・集計

### 3. 論文執筆 (`perform_writeup.py`)

実験結果から学術論文を自動生成します。

- LaTeX形式での論文生成
- 引用の自動収集・挿入
- 図表の自動生成・配置

### 4. レビュー (`perform_review.py`, `perform_vlm_review.py`)

生成された論文を自動でレビューします。

- テキストベースのレビュー
- VLM（Vision-Language Model）による図表レビュー

---

## ディレクトリ構造

```
masist/
├── README.md                 # このファイル
├── __init__.py
├── llm.py                    # LLMクライアント（OpenAI, Anthropic, DeepSeek等）
│
├── perform_ideation.py       # アイデア生成
├── perform_writeup.py        # 論文執筆
├── perform_plotting.py       # プロット生成
├── perform_review.py         # テキストレビュー
├── perform_vlm_review.py     # VLMレビュー
├── vlm.py                    # VLMクライアント
│
├── ideas/                    # トピック記述ファイル
│   └── multi_agent_simulation.md
│
├── tools/                    # 外部ツール連携
│   ├── base_tool.py
│   └── semantic_scholar.py   # Semantic Scholar API
│
├── treesearch/               # 実験実行エンジン
│   ├── agent_manager.py      # エージェント管理
│   ├── code_generator.py     # コード生成
│   ├── interpreter.py        # コード実行
│   ├── parallel_agent.py     # 並列実行
│   ├── result_evaluator.py   # 結果評価
│   ├── journal.py            # 実験ログ
│   └── backend/              # LLMバックエンド
│
├── utils/                    # ユーティリティ
├── fewshot_examples/         # Few-shot例
└── blank_icbinb_latex/       # LaTeXテンプレート
```

---

## クイックスタート

### 環境設定

```bash
# 環境変数の設定
export DEEPSEEK_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
export S2_API_KEY="your-semantic-scholar-api-key"
```

### アイデア生成

```bash
# 英語で生成
python masist/perform_ideation.py \
  --workshop-file masist/ideas/multi_agent_simulation.md \
  --model deepseek-reasoner

# 日本語で生成
python masist/perform_ideation.py \
  --workshop-file masist/ideas/multi_agent_simulation.md \
  --model deepseek-chat \
  --japanese
```

### 実験実行

```bash
# 設定ファイルを使用して実験を実行
python -m masist.treesearch.perform_experiments_with_agentmanager \
  --config masist_config.yaml
```

---

## 設定

### masist_config.yaml

主要な設定項目：

| セクション | 設定項目 | 説明 |
|-----------|---------|------|
| `agent.code.model` | `deepseek-chat` | コード生成用モデル |
| `writeup_big_model` | `deepseek-reasoner` | 論文執筆用モデル（推論） |
| `review_model` | `deepseek-chat` | レビュー用モデル |
| `vlm_model` | `gpt-4o-mini` | 画像レビュー用VLM |
| `agent.num_workers` | `4` | 並列ワーカー数 |

---

## 対応モデル

| プロバイダ | モデル | 用途 |
|-----------|--------|------|
| DeepSeek | `deepseek-reasoner` | アイデア生成、論文執筆（推論） |
| DeepSeek | `deepseek-chat` | コード生成、レビュー |
| OpenAI | `gpt-4o`, `gpt-4o-mini` | VLM、汎用 |
| OpenAI | `o1`, `o3-mini` | 推論タスク |
| Anthropic | `claude-3-5-sonnet` | 汎用 |
| Google | `gemini-2.0-flash` | 汎用 |

---

## 研究ワークフロー

```
┌──────────────────────────────────────────────────────────────────┐
│                        MASist ワークフロー                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. トピック記述      2. アイデア生成      3. 実験実行            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ ideas/*.md  │ -> │ ideation.py │ -> │ treesearch/ │          │
│  │ (自然言語)   │    │ (LLM推論)    │    │ (LLMエージェント) │      │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                            │                  │                  │
│                            v                  v                  │
│                     ideas/*.json        workspaces/              │
│                     (アイデアJSON)       (実験結果)               │
│                                               │                  │
│  4. 論文執筆         5. レビュー              │                  │
│  ┌─────────────┐    ┌─────────────┐          │                  │
│  │ writeup.py  │ <- │ review.py   │ <--------┘                  │
│  │ (LaTeX生成)  │    │ (自動評価)   │                             │
│  └─────────────┘    └─────────────┘                              │
│         │                                                        │
│         v                                                        │
│    論文PDF                                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```


## 参考文献

- **AI-Scientist**: [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) - 本プロジェクトの基盤
- **AI-Scientist-v2**: AI-Scientistの拡張版、コード改善ループを含む


---

*本プロジェクトは、LLMによるMASの可能性と限界を原理的に探求することを目的としています。*
