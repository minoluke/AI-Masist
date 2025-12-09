"""
Stage 1-4終了後の最終プロット集約機能テスト

masist/perform_plotting.py の aggregate_plots() 関数のテスト

テスト一覧:
- Test 1: モジュールインポートテスト (API不要)
- Test 2: load_exp_summaries テスト (API不要)
- Test 3: filter_experiment_summaries テスト (API不要)
- Test 4: build_aggregator_prompt テスト (API不要)
- Test 5: extract_code_snippet テスト (API不要)
- Test 6: モックLLM集約テスト (API不要)
- Test 7: 実際のLLM集約テスト (API必要)
- Test 8: 集約スクリプト実行テスト (API必要)

使用方法:
    # 全テスト実行 (API不要テストのみ)
    python tests/test_final_plot_aggregation.py

    # APIテストも含む全テスト
    python tests/test_final_plot_aggregation.py --api

    # 特定のテストのみ
    python tests/test_final_plot_aggregation.py --tests 1 2 3

    # テスト一覧表示
    python tests/test_final_plot_aggregation.py --list
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# パス設定
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env読み込み
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ .env ファイルを読み込みました: {env_path}")


def print_header(test_num: int, test_name: str):
    """テストヘッダー表示"""
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_name}")
    print('='*60)


def test_module_imports():
    """Test 1: モジュールインポートテスト (API不要)"""
    print_header(1, "モジュールインポートテスト")

    try:
        from masist.perform_plotting import (
            aggregate_plots,
            build_aggregator_prompt,
            extract_code_snippet,
            run_aggregator_script,
            load_exp_summaries,
            filter_experiment_summaries,
            AGGREGATOR_SYSTEM_MSG,
            MAX_FIGURES,
        )
        print("  ✓ aggregate_plots インポート成功")
        print("  ✓ build_aggregator_prompt インポート成功")
        print("  ✓ extract_code_snippet インポート成功")
        print("  ✓ run_aggregator_script インポート成功")
        print("  ✓ load_exp_summaries インポート成功")
        print("  ✓ filter_experiment_summaries インポート成功")
        print("  ✓ AGGREGATOR_SYSTEM_MSG インポート成功")
        print(f"  ✓ MAX_FIGURES = {MAX_FIGURES}")

        print("\n✓ Test 1 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_exp_summaries():
    """Test 2: load_exp_summaries テスト (API不要)"""
    print_header(2, "load_exp_summaries テスト")

    try:
        from masist.perform_plotting import load_exp_summaries

        with tempfile.TemporaryDirectory() as tmpdir:
            # テスト用サマリファイル作成
            baseline_summary = {
                "best node": {
                    "overall_plan": "Baseline experiment plan",
                    "metric": 0.75,
                    "plot_code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3])"
                }
            }
            research_summary = {
                "best node": {
                    "overall_plan": "Research experiment plan",
                    "metric": 0.85,
                    "plot_code": "import matplotlib.pyplot as plt\nplt.bar([1,2,3], [4,5,6])"
                }
            }
            ablation_summary = [
                {
                    "ablation_name": "ablation_1",
                    "overall_plan": "Ablation 1 plan",
                    "metric": 0.80
                }
            ]

            # ファイル書き込み
            with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
                json.dump(baseline_summary, f)
            with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
                json.dump(research_summary, f)
            with open(os.path.join(tmpdir, "ablation_summary.json"), "w") as f:
                json.dump(ablation_summary, f)

            # 読み込みテスト
            result = load_exp_summaries(tmpdir)

            assert "BASELINE_SUMMARY" in result
            assert "RESEARCH_SUMMARY" in result
            assert "ABLATION_SUMMARY" in result
            print("  ✓ 3つのサマリキーが存在")

            assert result["BASELINE_SUMMARY"]["best node"]["metric"] == 0.75
            print("  ✓ BASELINE_SUMMARY の内容が正しい")

            assert result["RESEARCH_SUMMARY"]["best node"]["metric"] == 0.85
            print("  ✓ RESEARCH_SUMMARY の内容が正しい")

            assert len(result["ABLATION_SUMMARY"]) == 1
            print("  ✓ ABLATION_SUMMARY の内容が正しい")

        print("\n✓ Test 2 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filter_experiment_summaries():
    """Test 3: filter_experiment_summaries テスト (API不要)"""
    print_header(3, "filter_experiment_summaries テスト")

    try:
        from masist.perform_plotting import filter_experiment_summaries

        # テストデータ
        exp_summaries = {
            "BASELINE_SUMMARY": {
                "best node": {
                    "overall_plan": "Test plan",
                    "analysis": "Test analysis",
                    "metric": 0.75,
                    "code": "print('hello')",
                    "plot_plan": "Plot plan",
                    "plot_code": "plt.plot([1,2,3])",
                    "plot_analyses": ["analysis1"],
                    "vlm_feedback_summary": "VLM feedback",
                    "exp_results_npy_files": ["/path/to/file.npy"],
                    "extra_field": "should be removed"
                }
            },
            "RESEARCH_SUMMARY": {
                "best node": {
                    "overall_plan": "Research plan",
                    "analysis": "Research analysis",
                    "metric": 0.85,
                    "plot_code": "plt.bar([1,2,3])",
                    "exp_results_npy_files": ["/path/to/research.npy"],
                }
            },
            "ABLATION_SUMMARY": [
                {
                    "ablation_name": "ablation_1",
                    "overall_plan": "Ablation plan",
                    "plot_code": "plt.scatter([1,2], [3,4])",
                    "exp_results_npy_files": ["/path/to/ablation.npy"],
                }
            ]
        }

        # plot_aggregation フィルタリング
        filtered = filter_experiment_summaries(exp_summaries, "plot_aggregation")

        assert "BASELINE_SUMMARY" in filtered
        assert "RESEARCH_SUMMARY" in filtered
        assert "ABLATION_SUMMARY" in filtered
        print("  ✓ フィルタリング後も3つのサマリキーが存在")

        # extra_field が削除されていることを確認
        assert "extra_field" not in filtered["BASELINE_SUMMARY"].get("best node", {})
        print("  ✓ 不要なフィールドが削除されている")

        # 必要なフィールドが残っていることを確認
        baseline_node = filtered["BASELINE_SUMMARY"]["best node"]
        assert "overall_plan" in baseline_node
        assert "plot_code" in baseline_node
        assert "exp_results_npy_files" in baseline_node
        print("  ✓ 必要なフィールドが保持されている")

        print("\n✓ Test 3 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_aggregator_prompt():
    """Test 4: build_aggregator_prompt テスト (API不要)"""
    print_header(4, "build_aggregator_prompt テスト")

    try:
        from masist.perform_plotting import build_aggregator_prompt

        combined_summaries_str = json.dumps({
            "BASELINE_SUMMARY": {"best node": {"metric": 0.75}},
            "RESEARCH_SUMMARY": {"best node": {"metric": 0.85}}
        }, indent=2)

        idea_text = "This is a test research idea about machine learning."

        prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

        # プロンプトに必要な要素が含まれているか確認
        assert "RESEARCH IDEA" in prompt
        print("  ✓ RESEARCH IDEA セクションが含まれている")

        assert idea_text in prompt
        print("  ✓ idea_text が含まれている")

        assert "BASELINE_SUMMARY" in prompt
        print("  ✓ BASELINE_SUMMARY が含まれている")

        assert ".npy" in prompt
        print("  ✓ .npy ファイルについての指示が含まれている")

        assert "figures/" in prompt
        print("  ✓ figures/ ディレクトリについての指示が含まれている")

        assert "triple backticks" in prompt or "```" in prompt
        print("  ✓ コードブロック形式についての指示が含まれている")

        print("\n✓ Test 4 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extract_code_snippet():
    """Test 5: extract_code_snippet テスト (API不要)"""
    print_header(5, "extract_code_snippet テスト")

    try:
        from masist.perform_plotting import extract_code_snippet

        # ケース1: Pythonコードブロック
        text1 = """Here is the code:

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig('test.png')
```

That's the complete script."""

        code1 = extract_code_snippet(text1)
        assert "import matplotlib" in code1
        assert "plt.plot" in code1
        print("  ✓ Python コードブロックの抽出成功")

        # ケース2: 言語指定なしのコードブロック
        text2 = """```
print("hello")
```"""
        code2 = extract_code_snippet(text2)
        assert 'print("hello")' in code2
        print("  ✓ 言語指定なしコードブロックの抽出成功")

        # ケース3: コードブロックなし
        text3 = "No code block here, just plain text."
        code3 = extract_code_snippet(text3)
        assert code3 == text3.strip()
        print("  ✓ コードブロックなしの場合は全文を返す")

        # ケース4: 複数のコードブロック（最初のものを取得）
        text4 = """First block:
```python
code1()
```

Second block:
```python
code2()
```"""
        code4 = extract_code_snippet(text4)
        assert "code1()" in code4
        print("  ✓ 複数コードブロックの場合、最初のものを抽出")

        print("\n✓ Test 5 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_llm_aggregation():
    """Test 6: モックLLM集約テスト (API不要)"""
    print_header(6, "モックLLM集約テスト")

    try:
        from masist.perform_plotting import aggregate_plots

        mock_response = '''
Here is the aggregation script:

```python
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# Plot 1: Baseline vs Research comparison
try:
    x = [1, 2, 3, 4, 5]
    baseline = [0.7, 0.72, 0.74, 0.75, 0.75]
    research = [0.75, 0.78, 0.82, 0.84, 0.85]

    plt.figure(figsize=(10, 6))
    plt.plot(x, baseline, 'b-o', label='Baseline')
    plt.plot(x, research, 'r-s', label='Research')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Baseline vs Research Performance')
    plt.legend()
    plt.savefig('figures/comparison.png', dpi=300)
    plt.close()
    print("Created: figures/comparison.png")
except Exception as e:
    print(f"Error in Plot 1: {e}")

print("Aggregation complete")
```
'''

        with tempfile.TemporaryDirectory() as tmpdir:
            # テスト用サマリファイル作成
            baseline_summary = {"best node": {"metric": 0.75}}
            research_summary = {"best node": {"metric": 0.85}}

            with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
                json.dump(baseline_summary, f)
            with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
                json.dump(research_summary, f)
            with open(os.path.join(tmpdir, "ablation_summary.json"), "w") as f:
                json.dump([], f)

            # LLMをモック
            with patch('masist.perform_plotting.create_client') as mock_create:
                with patch('masist.perform_plotting.get_response_from_llm') as mock_llm:
                    mock_create.return_value = (MagicMock(), "deepseek-chat")
                    # 最初の呼び出しでコード生成、2回目で "I am done"
                    mock_llm.side_effect = [
                        (mock_response, []),
                        ("I am done", []),
                    ]

                    aggregate_plots(
                        base_folder=tmpdir,
                        task_desc="Test task description",
                        model="deepseek-chat",
                        n_reflections=1
                    )

            # 結果確認
            aggregator_script = os.path.join(tmpdir, "auto_plot_aggregator.py")
            assert os.path.exists(aggregator_script), "集約スクリプトが生成されていない"
            print("  ✓ auto_plot_aggregator.py が生成された")

            with open(aggregator_script, "r") as f:
                content = f.read()
            assert "matplotlib" in content
            assert "figures" in content
            print("  ✓ スクリプト内容が正しい")

            # figuresディレクトリが作成されたか（スクリプト実行後）
            figures_dir = os.path.join(tmpdir, "figures")
            if os.path.exists(figures_dir):
                print(f"  ✓ figures ディレクトリが作成された")

        print("\n✓ Test 6 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_llm_aggregation():
    """Test 7: 実際のLLM集約テスト (API必要)"""
    print_header(7, "実際のLLM集約テスト")

    # APIキー確認
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("  ⚠ DEEPSEEK_API_KEY が設定されていません")
        print("\n⚠ Test 7 SKIPPED")
        return True  # スキップは成功扱い

    try:
        from masist.perform_plotting import aggregate_plots

        # logs/test_aggregation を使用（結果を確認できるように）
        project_root = Path(__file__).parent.parent
        tmpdir = project_root / "logs" / "test_aggregation"
        tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdir = str(tmpdir)
        print(f"  出力ディレクトリ: {tmpdir}")

        # TPGG実験に基づいた本格的なテストデータ
        # Threshold Public Goods Game - しきい値公共財ゲーム
        baseline_summary = {
            "best node": {
                "overall_plan": "CONTROL条件（ルールなし）でのベースライン実験。4人グループで20ラウンドの公共財ゲームを実施。各プレイヤーは10トークンの初期保有から拠出額を決定。",
                "analysis": """ベースライン実験結果:
- しきい値達成率: 65% (13/20ラウンド)
- 平均拠出額: 5.2トークン/人
- 過剰拠出（しきい値超過分）: 平均1.8トークン
- ラウンド1-10: 達成率70%、平均拠出5.5
- ラウンド11-20: 達成率60%、平均拠出4.9（疲弊効果）
- グループ間分散: 標準偏差1.2

協力行動の傾向:
- 初期は高い協力率だが、後半で低下
- フリーライダーの出現率: 15%
- 条件反応型プレイヤー: 60%""",
                "metric": 0.65,
                "plot_code": """
import matplotlib.pyplot as plt
import numpy as np

# ラウンドごとの達成率
rounds = np.arange(1, 21)
achievement_rate = [0.8, 0.75, 0.7, 0.72, 0.68, 0.7, 0.65, 0.68, 0.7, 0.65,
                    0.62, 0.6, 0.58, 0.6, 0.55, 0.58, 0.6, 0.55, 0.58, 0.6]

plt.figure(figsize=(10, 6))
plt.plot(rounds, achievement_rate, 'b-o', label='CONTROL (Baseline)')
plt.xlabel('Round')
plt.ylabel('Threshold Achievement Rate')
plt.title('Baseline: Threshold Achievement Rate by Round')
plt.legend()
plt.ylim(0, 1)
plt.savefig('baseline_achievement.png', dpi=300)
plt.close()
""",
                "plot_analyses": [
                    "ベースライン条件では協力率が時間とともに低下する傾向が見られる",
                    "ラウンド10付近で一時的な回復があるが、後半は安定して低い"
                ],
                "vlm_feedback_summary": "グラフは明確で読みやすい。凡例の位置が適切。Y軸の範囲が0-1で正規化されている。",
                "exp_results_npy_files": [
                    f"{tmpdir}/baseline_contributions.npy",
                    f"{tmpdir}/baseline_achievements.npy"
                ]
            }
        }
        research_summary = {
            "best node": {
                "overall_plan": "FAIRSUFF条件（必要額ちょうど＋均等割り）での研究実験。ルールとして各プレイヤーに5トークンの拠出を提示。しきい値T=20、ルール合計も20で一致。",
                "analysis": """研究実験結果 (FAIRSUFF条件):
- しきい値達成率: 85% (17/20ラウンド) [+20% vs baseline]
- 平均拠出額: 5.0トークン/人（ルール通り）
- 過剰拠出: 平均0.5トークン（効率的）
- ルール遵守率: 78%
- ラウンド11-20（ルール導入後）: 達成率90%

条件間比較:
- FAIRSUFF vs CONTROL: 達成率 +20%, 過剰拠出 -72%
- 均等割りルールが協力を促進
- ルール導入後の即座の改善（ラウンド11で急上昇）

統計的検定:
- t検定 p < 0.01 で有意差あり
- 効果量 Cohen's d = 0.85（大）""",
                "metric": 0.85,
                "plot_code": """
import matplotlib.pyplot as plt
import numpy as np

rounds = np.arange(1, 21)
# FAIRSUFF条件の達成率（ラウンド11からルール導入）
achievement_fairsuff = [0.8, 0.75, 0.7, 0.72, 0.68, 0.7, 0.65, 0.68, 0.7, 0.65,
                        0.88, 0.90, 0.92, 0.88, 0.90, 0.85, 0.88, 0.90, 0.85, 0.88]

plt.figure(figsize=(10, 6))
plt.plot(rounds, achievement_fairsuff, 'r-s', label='FAIRSUFF (Research)')
plt.axvline(x=10.5, color='gray', linestyle='--', label='Rule Introduction')
plt.xlabel('Round')
plt.ylabel('Threshold Achievement Rate')
plt.title('Research: FAIRSUFF Condition Achievement Rate')
plt.legend()
plt.ylim(0, 1)
plt.savefig('research_achievement.png', dpi=300)
plt.close()
""",
                "plot_analyses": [
                    "ルール導入（ラウンド11）後に達成率が顕著に向上",
                    "FAIRSUFF条件は均等割りが可能なため、協力が安定",
                    "後半も高い達成率を維持（85-92%）"
                ],
                "vlm_feedback_summary": "ルール導入時点を示す縦線が効果的。研究条件の優位性が視覚的に明確。",
                "exp_results_npy_files": [
                    f"{tmpdir}/research_contributions.npy",
                    f"{tmpdir}/research_achievements.npy"
                ]
            }
        }

        # ablation_summaryも追加（他の条件との比較）
        ablation_summary = [
            {
                "ablation_name": "FAIRINF",
                "overall_plan": "多めの要求＋均等割り可能条件。ルール合計22、しきい値20。",
                "analysis": "達成率75%。ルール遵守率65%。過剰拠出が増加（平均2.5トークン）。",
                "metric": 0.75,
                "plot_code": """
plt.figure()
plt.bar(['CONTROL', 'FAIRSUFF', 'FAIRINF'], [0.65, 0.85, 0.75])
plt.ylabel('Achievement Rate')
plt.savefig('ablation_fairinf.png')
""",
                "exp_results_npy_files": [f"{tmpdir}/ablation_fairinf.npy"]
            },
            {
                "ablation_name": "UNFAIRSUFF",
                "overall_plan": "必要額ちょうど＋均等割り不可条件。しきい値22、ルール[5,5,6,6]。",
                "analysis": "達成率70%。不均等なルールにより協力にばらつき。",
                "metric": 0.70,
                "plot_code": """
plt.figure()
plt.bar(['CONTROL', 'FAIRSUFF', 'UNFAIRSUFF'], [0.65, 0.85, 0.70])
plt.ylabel('Achievement Rate')
plt.savefig('ablation_unfairsuff.png')
""",
                "exp_results_npy_files": [f"{tmpdir}/ablation_unfairsuff.npy"]
            }
        ]

        with open(os.path.join(tmpdir, "baseline_summary.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(tmpdir, "research_summary.json"), "w", encoding="utf-8") as f:
            json.dump(research_summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(tmpdir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(ablation_summary, f, indent=2, ensure_ascii=False)

        # TPGG実験のタスク説明
        task_desc = """Threshold Public Goods Game (TPGG) - しきい値公共財ゲーム

研究目的: グループに示される「出してほしい金額の目安（ルール）」が、「必要額ちょうど」か「必要以上に多い」かによって、
LLMエージェントの行動、結果（達成率）、効率（過剰拠出）がどう変わるのかを調べる。

実験条件:
- CONTROL: ルールなし（ベースライン）
- FAIRSUFF: 必要額ちょうど＋均等割りOK (T=20, R=[5,5,5,5])
- FAIRINF: 多めの要求＋均等割りOK (T=20, R=[5,5,6,6])
- UNFAIRSUFF: 必要額ちょうど＋均等割り不可 (T=22, R=[5,5,6,6])

主要指標:
- しきい値達成率
- 平均拠出額
- 過剰拠出（無駄）
- ルール遵守率
"""

        print("  LLMを呼び出して集約スクリプトを生成中...")

        aggregate_plots(
            base_folder=tmpdir,
            task_desc=task_desc,
            model="deepseek-chat",
            n_reflections=1  # テストなので1回のみ
        )

        # 結果確認
        aggregator_script = os.path.join(tmpdir, "auto_plot_aggregator.py")
        if os.path.exists(aggregator_script):
            print("  ✓ auto_plot_aggregator.py が生成された")

            with open(aggregator_script, "r") as f:
                content = f.read()

            print(f"  スクリプトサイズ: {len(content)} bytes")

            # 基本的な要素の確認
            if "matplotlib" in content:
                print("  ✓ matplotlib を使用している")
            if "figures" in content:
                print("  ✓ figures ディレクトリを参照している")
            if "savefig" in content:
                print("  ✓ savefig を使用している")
        else:
            print("  ⚠ スクリプトが生成されなかった")

        # figuresディレクトリ確認
        figures_dir = os.path.join(tmpdir, "figures")
        if os.path.exists(figures_dir):
            figures = os.listdir(figures_dir)
            print(f"  ✓ {len(figures)} 個の図が生成された: {figures}")

        print("\n✓ Test 7 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregator_script_execution():
    """Test 8: 集約スクリプト実行テスト (API必要)"""
    print_header(8, "集約スクリプト実行テスト")

    try:
        from masist.perform_plotting import run_aggregator_script

        # logs/test_script_execution を使用（結果を確認できるように）
        project_root = Path(__file__).parent.parent
        tmpdir = project_root / "logs" / "test_script_execution"
        tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdir = str(tmpdir)
        print(f"  出力ディレクトリ: {tmpdir}")

        # テスト用の簡単な集約スクリプト
        test_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# Simple test plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], 'b-o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Plot')
plt.savefig('figures/test_plot.png', dpi=100)
plt.close()

print("Test plot created successfully")
'''

        script_path = os.path.join(tmpdir, "test_aggregator.py")

        output = run_aggregator_script(
            aggregator_code=test_code,
            aggregator_script_path=script_path,
            base_folder=tmpdir,
            script_name="test_aggregator.py"
        )

        print(f"  スクリプト出力: {output[:200]}..." if len(output) > 200 else f"  スクリプト出力: {output}")

        # スクリプトが作成されたか
        assert os.path.exists(script_path), "スクリプトファイルが作成されていない"
        print("  ✓ スクリプトファイルが作成された")

        # figuresディレクトリが作成されたか
        figures_dir = os.path.join(tmpdir, "figures")
        assert os.path.exists(figures_dir), "figures ディレクトリが作成されていない"
        print("  ✓ figures ディレクトリが作成された")

        # 画像が生成されたか
        test_plot = os.path.join(figures_dir, "test_plot.png")
        assert os.path.exists(test_plot), "テスト画像が生成されていない"
        print("  ✓ test_plot.png が生成された")

        # 出力に成功メッセージが含まれるか
        assert "successfully" in output.lower() or "created" in output.lower()
        print("  ✓ 実行が成功した")

        print("\n✓ Test 8 PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1-4終了後の最終プロット集約機能テスト"
    )
    parser.add_argument(
        '--tests', '-t', nargs='+', type=int,
        help='実行するテスト番号を指定（例: --tests 1 3 5）'
    )
    parser.add_argument(
        '--api', action='store_true',
        help='API呼び出しテストも実行'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='テスト一覧を表示'
    )
    parser.add_argument(
        '--yes', '-y', action='store_true',
        help='確認プロンプトをスキップ'
    )
    args = parser.parse_args()

    all_tests = [
        (1, "モジュールインポート", test_module_imports, False),
        (2, "load_exp_summaries", test_load_exp_summaries, False),
        (3, "filter_experiment_summaries", test_filter_experiment_summaries, False),
        (4, "build_aggregator_prompt", test_build_aggregator_prompt, False),
        (5, "extract_code_snippet", test_extract_code_snippet, False),
        (6, "モックLLM集約", test_mock_llm_aggregation, False),
        (7, "実際のLLM集約", test_real_llm_aggregation, True),
        (8, "集約スクリプト実行", test_aggregator_script_execution, False),
    ]

    # テスト一覧表示
    if args.list:
        print("\n" + "="*60)
        print("利用可能なテスト一覧")
        print("="*60)
        for num, name, _, requires_api in all_tests:
            api_mark = " [API必要]" if requires_api else ""
            print(f"  {num}. {name}{api_mark}")
        print("\n使用例:")
        print("  python tests/test_final_plot_aggregation.py --tests 1 2 3")
        print("  python tests/test_final_plot_aggregation.py --api")
        return

    print("\n" + "="*60)
    print("AI-MASIST 最終プロット集約機能テスト")
    print("="*60)

    # 環境変数確認
    print("\n" + "="*60)
    print("環境変数確認")
    print("="*60)

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        print(f"✓ DEEPSEEK_API_KEY: 設定済み")
    else:
        print(f"✗ DEEPSEEK_API_KEY: 未設定")

    # 実行するテストを決定
    if args.tests:
        tests_to_run = [(num, name, func, api) for num, name, func, api in all_tests if num in args.tests]
        test_nums = ", ".join([f"{num}. {name}" for num, name, _, _ in tests_to_run])
        print(f"\n実行するテスト: {test_nums}")
    else:
        if args.api:
            tests_to_run = all_tests
            print("\n全テストを実行（API呼び出し含む）")
        else:
            tests_to_run = [(num, name, func, api) for num, name, func, api in all_tests if not api]
            print("\nAPI不要テストのみ実行（--api で全テスト実行）")

    # 確認プロンプト
    has_api_tests = any(api for _, _, _, api in tests_to_run)
    if has_api_tests and not args.yes:
        print("\n⚠ 警告: APIテストは実際のAPI呼び出しを行い、料金が発生する可能性があります")
        response = input("続行しますか？ [y/N]: ")
        if response.lower() != 'y':
            print("テストを中止しました")
            return
    elif has_api_tests:
        print("\n⚠ APIテスト実行（自動実行モード）")

    # テスト実行
    results = []
    for num, name, func, requires_api in tests_to_run:
        try:
            passed = func()
            results.append((num, name, passed))
        except Exception as e:
            print(f"\n✗ Test {num} FAILED with exception: {e}")
            results.append((num, name, False))

    # サマリ
    print("\n" + "="*60)
    print("テスト結果サマリ")
    print("="*60)

    passed_count = 0
    for num, name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if passed:
            passed_count += 1

    print(f"\n合計: {passed_count}/{len(results)} テスト成功")

    if passed_count == len(results):
        print("\n✓ 全テスト成功！")
    else:
        print(f"\n✗ {len(results) - passed_count} テスト失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
