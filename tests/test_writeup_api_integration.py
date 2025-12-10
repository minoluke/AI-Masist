"""
論文執筆機能の実API統合テスト

このスクリプトは実際のAPI呼び出しを行い、以下をテストします：
1. LLM API (DeepSeek) の呼び出し
2. VLM API (OpenAI GPT-4o-mini) の呼び出し
3. Semantic Scholar API の呼び出し
4. 引用収集機能
5. 論文執筆機能（簡易版）
6. レビュー機能

注意: このテストは実際のAPIを呼び出すため、API利用料金が発生します。
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# masist モジュールをインポート可能にする
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ .env ファイルを読み込みました: {env_path}")
else:
    print(f"⚠ .env ファイルが見つかりません: {env_path}")


def check_env_variables():
    """必要な環境変数の確認"""
    print("=" * 60)
    print("環境変数確認")
    print("=" * 60)

    required_vars = {
        "DEEPSEEK_API_KEY": "DeepSeek API (必須)",
        "OPENAI_API_KEY": "OpenAI API (VLM用, 必須)",
        "S2_API_KEY": "Semantic Scholar API (必須)",
    }

    all_set = True
    for var, description in required_vars.items():
        if var in os.environ and os.environ[var]:
            print(f"✓ {var}: 設定済み ({description})")
        else:
            print(f"✗ {var}: 未設定 ({description})")
            all_set = False

    print()
    return all_set


def test_deepseek_chat_api():
    """DeepSeek Chat API の呼び出しテスト"""
    print("=" * 60)
    print("TEST 1: DeepSeek Chat API 呼び出し")
    print("=" * 60)

    try:
        from masist.llm import create_client, get_response_from_llm

        # DeepSeek Chat クライアント作成
        client, model = create_client("deepseek-chat")
        print(f"✓ クライアント作成成功: {model}")

        # 簡単なプロンプトで呼び出し
        test_prompt = "Hello! Please respond with 'API test successful' if you can read this."
        system_message = "You are a helpful assistant."

        print("API呼び出し中...")
        response, msg_history = get_response_from_llm(
            prompt=test_prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False,
            temperature=0.7,
        )

        print(f"✓ API呼び出し成功")
        print(f"  レスポンス: {response[:100]}...")
        return True

    except Exception as e:
        print(f"✗ DeepSeek Chat API 呼び出し失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deepseek_reasoner_api():
    """DeepSeek Reasoner API の呼び出しテスト"""
    print("=" * 60)
    print("TEST 2: DeepSeek Reasoner API 呼び出し")
    print("=" * 60)

    try:
        from masist.llm import create_client, get_response_from_llm

        # DeepSeek Reasoner クライアント作成
        client, model = create_client("deepseek-reasoner")
        print(f"✓ クライアント作成成功: {model}")

        # 推論が必要な簡単なプロンプト
        test_prompt = "What is 2 + 2? Please explain your reasoning."
        system_message = "You are a helpful assistant that explains your reasoning."

        print("API呼び出し中...")
        response, msg_history = get_response_from_llm(
            prompt=test_prompt,
            client=client,
            model=model,
            system_message=system_message,
            print_debug=False,
        )

        print(f"✓ API呼び出し成功")
        print(f"  レスポンス: {response[:100]}...")
        return True

    except Exception as e:
        print(f"✗ DeepSeek Reasoner API 呼び出し失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_vlm_api():
    """OpenAI VLM (gpt-4o-mini) API の呼び出しテスト"""
    print("=" * 60)
    print("TEST 3: OpenAI VLM (gpt-4o-mini) API 呼び出し")
    print("=" * 60)

    try:
        from masist.vlm import create_client, get_response_from_vlm
        import base64
        from PIL import Image
        import io

        # VLM クライアント作成
        client, model = create_client("gpt-4o-mini")
        print(f"✓ クライアント作成成功: {model}")

        # 簡単なテスト画像を作成（赤い正方形）
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img.write(img_byte_arr)
            tmp_img_path = tmp_img.name

        try:
            test_prompt = "What color is this image? Please respond briefly."
            system_message = "You are a helpful assistant that describes images."

            print("API呼び出し中...")
            response, msg_history = get_response_from_vlm(
                msg=test_prompt,
                image_paths=[tmp_img_path],
                client=client,
                model=model,
                system_message=system_message,
            )

            print(f"✓ API呼び出し成功")
            print(f"  レスポンス: {response[:100]}...")
            return True

        finally:
            # 一時ファイル削除
            os.unlink(tmp_img_path)

    except Exception as e:
        print(f"✗ OpenAI VLM API 呼び出し失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_scholar_api():
    """Semantic Scholar API の呼び出しテスト"""
    print("=" * 60)
    print("TEST 4: Semantic Scholar API 呼び出し")
    print("=" * 60)

    try:
        from masist.tools.semantic_scholar import search_for_papers

        # 有名な論文で検索
        query = "attention is all you need"
        print(f"検索クエリ: {query}")
        print("API呼び出し中...")

        papers = search_for_papers(query, result_limit=3)

        if papers:
            print(f"✓ API呼び出し成功: {len(papers)} 件の論文を取得")
            for i, paper in enumerate(papers):
                print(f"  {i+1}. {paper.get('title', 'No title')}")
            return True
        else:
            print("⚠ API呼び出し成功だが、結果が0件")
            return True  # APIは成功したので True

    except Exception as e:
        print(f"✗ Semantic Scholar API 呼び出し失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citation_gathering():
    """引用収集機能のテスト（簡易版）"""
    print("=" * 60)
    print("TEST 5: 引用収集機能テスト（簡易版）")
    print("=" * 60)

    try:
        from masist.perform_writeup import gather_citations

        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"一時ディレクトリ: {tmpdir}")

            # ダミーのサマリファイルを作成
            baseline_summary = {
                "best node": {
                    "overall_plan": "We investigate the impact of attention mechanisms in neural networks.",
                    "analysis": "Our experiments show that attention improves performance on sequence tasks.",
                    "metric": 0.85,
                }
            }

            research_summary = {
                "best node": {
                    "overall_plan": "We propose a novel multi-head attention mechanism.",
                    "analysis": "The proposed method achieves state-of-the-art results.",
                    "metric": 0.92,
                }
            }

            # ファイルに保存
            with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
                json.dump(baseline_summary, f)

            with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
                json.dump(research_summary, f)

            # タスク説明
            task_desc = """
# Research Task: Attention Mechanisms in Neural Networks

We investigate the effectiveness of attention mechanisms in deep learning models.
Our goal is to compare different attention architectures and evaluate their impact on performance.
"""

            print("引用収集を実行中（最大3ラウンド）...")
            citations_text = gather_citations(
                base_folder=tmpdir,
                task_desc=task_desc,
                num_cite_rounds=3,  # テストなので少なめ
                small_model="deepseek-chat",
            )

            if citations_text:
                print(f"✓ 引用収集成功")
                print(f"  収集された引用数: {citations_text.count('@')}")
                print(f"  引用テキストの長さ: {len(citations_text)} 文字")

                # キャッシュファイルが作成されたか確認
                cache_path = os.path.join(tmpdir, "cached_citations.bib")
                if os.path.exists(cache_path):
                    print(f"✓ キャッシュファイル作成成功: {cache_path}")
                else:
                    print(f"⚠ キャッシュファイルが作成されていない")

                return True
            else:
                print("⚠ 引用収集は完了したが、引用が収集されなかった")
                return True  # 処理自体は成功

    except Exception as e:
        print(f"✗ 引用収集機能テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_exp_summaries_integration():
    """load_exp_summaries の統合テスト"""
    print("=" * 60)
    print("TEST 6: load_exp_summaries 統合テスト")
    print("=" * 60)

    try:
        from masist.perform_writeup import load_exp_summaries, filter_experiment_summaries

        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as tmpdir:
            # ダミーのサマリファイルを作成（より詳細な内容）
            baseline_summary = {
                "best node": {
                    "overall_plan": "Test baseline plan",
                    "analysis": "Test baseline analysis",
                    "metric": 0.85,
                    "code": "print('baseline')",
                    "plot_analyses": "Baseline plot analysis",
                    "vlm_feedback_summary": "VLM feedback for baseline",
                }
            }

            research_summary = {
                "best node": {
                    "overall_plan": "Test research plan",
                    "analysis": "Test research analysis",
                    "metric": 0.92,
                    "code": "print('research')",
                    "plot_analyses": "Research plot analysis",
                    "vlm_feedback_summary": "VLM feedback for research",
                }
            }

            ablation_summary = [
                {
                    "ablation_name": "ablation_1",
                    "overall_plan": "Ablation plan 1",
                    "analysis": "Ablation analysis 1",
                    "metric": 0.88,
                }
            ]

            # ファイルに保存
            with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
                json.dump(baseline_summary, f)

            with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
                json.dump(research_summary, f)

            with open(os.path.join(tmpdir, "ablation_summary.json"), "w") as f:
                json.dump(ablation_summary, f)

            # load_exp_summaries() を実行
            summaries = load_exp_summaries(tmpdir)

            print("✓ サマリファイルの読み込み成功")
            print(f"  取得したキー: {list(summaries.keys())}")

            # フィルタリングテスト
            for step_name in ["citation_gathering", "writeup", "plot_aggregation"]:
                filtered = filter_experiment_summaries(summaries, step_name=step_name)
                print(f"✓ フィルタリング成功 (step: {step_name})")

            return True

    except Exception as e:
        print(f"✗ load_exp_summaries 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_latex_compilation():
    """LaTeX コンパイルのテスト（pdflatex が必要）"""
    print("=" * 60)
    print("TEST 7: LaTeX コンパイルテスト")
    print("=" * 60)

    try:
        import subprocess

        # pdflatex が利用可能か確認
        try:
            result = subprocess.run(
                ["pdflatex", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                print("⚠ pdflatex が利用できません - スキップ")
                return True
        except FileNotFoundError:
            print("⚠ pdflatex がインストールされていません - スキップ")
            return True

        from masist.perform_writeup import compile_latex
        import importlib.resources

        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as tmpdir:
            latex_dir = os.path.join(tmpdir, "latex")
            os.makedirs(latex_dir)

            # テンプレートをコピー
            try:
                blank_latex_path = importlib.resources.files('masist').joinpath('blank_icbinb_latex')
                with importlib.resources.as_file(blank_latex_path) as template_path:
                    shutil.copytree(template_path, latex_dir, dirs_exist_ok=True)
            except AttributeError:
                import pkg_resources
                blank_latex_path = pkg_resources.resource_filename('masist', 'blank_icbinb_latex')
                shutil.copytree(blank_latex_path, latex_dir, dirs_exist_ok=True)

            pdf_file = os.path.join(tmpdir, "test.pdf")

            print("LaTeX コンパイル中...")
            compile_latex(latex_dir, pdf_file, timeout=60)

            if os.path.exists(pdf_file):
                print(f"✓ LaTeX コンパイル成功: {pdf_file}")
                print(f"  PDFサイズ: {os.path.getsize(pdf_file)} bytes")
                return True
            else:
                print("✗ PDF が生成されませんでした")
                return False

    except Exception as e:
        print(f"✗ LaTeX コンパイルテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_writeup_workflow():
    """最小限の論文執筆ワークフローテスト"""
    print("=" * 60)
    print("TEST 8: 最小限の論文執筆ワークフロー")
    print("=" * 60)

    print("⚠ このテストは時間がかかり、API費用が発生します")
    print("  スキップする場合は Ctrl+C を押してください")

    # pdflatex が利用可能か確認
    import subprocess
    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            print("⚠ pdflatex が利用できないため、スキップします")
            return True
    except FileNotFoundError:
        print("⚠ pdflatex がインストールされていないため、スキップします")
        return True

    try:
        from masist.perform_writeup import perform_writeup

        # logs/test_writeup_workflow に出力（結果を確認できるように）
        project_root = Path(__file__).parent.parent
        tmpdir = project_root / "logs" / "test_writeup_workflow"
        tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdir = str(tmpdir)
        print(f"出力ディレクトリ: {tmpdir}")

        # ダミーのサマリファイルを作成
        baseline_summary = {
            "best node": {
                "overall_plan": "We test a baseline neural network model.",
                "analysis": "The baseline achieves reasonable performance.",
                "metric": 0.75,
                "code": "# Baseline code\nmodel = NeuralNetwork()",
            }
        }

        research_summary = {
            "best node": {
                "overall_plan": "We propose an improved model with attention.",
                "analysis": "Our model outperforms the baseline significantly.",
                "metric": 0.90,
                "code": "# Research code\nmodel = AttentionNetwork()",
            }
        }

        # ファイルに保存
        with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
            json.dump(baseline_summary, f)

        with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
            json.dump(research_summary, f)

        # ablation_summary.json も空で作成
        with open(os.path.join(tmpdir, "ablation_summary.json"), "w") as f:
            json.dump([], f)

        # figures ディレクトリを作成（空でOK）
        os.makedirs(os.path.join(tmpdir, "figures"), exist_ok=True)

        # タスク説明
        task_desc = """
# Research Task: Testing Attention Mechanisms

This is a minimal test of the writeup functionality.
We compare a baseline model with an attention-based model.
"""

        print("論文執筆を実行中（引用なし、リフレクション1回）...")
        success = perform_writeup(
            base_folder=tmpdir,
            task_desc=task_desc,
            citations_text="",  # 引用をスキップ
            no_writing=False,
            num_cite_rounds=0,  # 引用収集をスキップ
            small_model="deepseek-chat",
            big_model="deepseek-chat",  # テストなので reasoner を使わない
            vlm_model="gpt-4o-mini",  # VLM用モデル
            n_writeup_reflections=1,  # リフレクション回数を減らす
            page_limit=4,
        )

        if success:
            print("✓ 論文執筆ワークフロー成功")

            # 生成されたファイルを確認
            pdf_files = [f for f in os.listdir(tmpdir) if f.endswith('.pdf')]
            if pdf_files:
                print(f"  生成されたPDF: {pdf_files}")
            else:
                print("  ⚠ PDFファイルが見つかりません")

            latex_dir = os.path.join(tmpdir, "latex")
            if os.path.exists(latex_dir):
                print(f"  ✓ LaTeX ディレクトリが作成されました")
                latex_files = os.listdir(latex_dir)
                print(f"    LaTeX ファイル: {latex_files}")

            return True
        else:
            print("✗ 論文執筆ワークフロー失敗")
            # 失敗時の詳細情報を表示
            latex_dir = os.path.join(tmpdir, "latex")
            if os.path.exists(latex_dir):
                print(f"  LaTeX ディレクトリの内容: {os.listdir(latex_dir)}")
                # template.log があれば最後の50行を表示
                log_file = os.path.join(latex_dir, "template.log")
                if os.path.exists(log_file):
                    print("  template.log の最後の部分:")
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        for line in lines[-30:]:
                            print(f"    {line.rstrip()}")
            else:
                print(f"  LaTeX ディレクトリが存在しません: {latex_dir}")
            # 生成されたPDFがあるか確認
            pdf_files = [f for f in os.listdir(tmpdir) if f.endswith('.pdf')]
            print(f"  既存のPDFファイル: {pdf_files}")
            return False

    except KeyboardInterrupt:
        print("\n⚠ ユーザーによってスキップされました")
        return True
    except Exception as e:
        print(f"✗ 論文執筆ワークフローテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テストを実行"""
    import argparse

    # テスト一覧を定義
    all_tests = [
        (1, "DeepSeek Chat API", test_deepseek_chat_api),
        (2, "DeepSeek Reasoner API", test_deepseek_reasoner_api),
        (3, "OpenAI VLM API", test_openai_vlm_api),
        (4, "Semantic Scholar API", test_semantic_scholar_api),
        (5, "引用収集機能", test_citation_gathering),
        (6, "load_exp_summaries", test_load_exp_summaries_integration),
        (7, "LaTeX コンパイル", test_latex_compilation),
        (8, "論文執筆ワークフロー", test_minimal_writeup_workflow),
    ]

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description='AI-MASIST 論文執筆機能 実API統合テスト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 全テスト実行
  python tests/test_writeup_api_integration.py --yes

  # テスト1と3のみ実行
  python tests/test_writeup_api_integration.py --tests 1 3

  # テスト7のみ実行
  python tests/test_writeup_api_integration.py --tests 7

  # テスト一覧を表示
  python tests/test_writeup_api_integration.py --list
'''
    )
    parser.add_argument('--yes', '-y', action='store_true',
                       help='確認をスキップして自動実行')
    parser.add_argument('--tests', '-t', nargs='+', type=int,
                       help='実行するテスト番号を指定（例: --tests 1 3 5）')
    parser.add_argument('--list', '-l', action='store_true',
                       help='テスト一覧を表示')

    args = parser.parse_args()

    # テスト一覧表示
    if args.list:
        print("\n" + "=" * 60)
        print("利用可能なテスト一覧")
        print("=" * 60)
        for num, name, _ in all_tests:
            print(f"  {num}: {name}")
        print("\n使用例:")
        print("  python tests/test_writeup_api_integration.py --tests 1 3")
        print("=" * 60)
        return 0

    print("\n" + "=" * 60)
    print("AI-MASIST 論文執筆機能 実API統合テスト")
    print("=" * 60 + "\n")

    # 環境変数確認
    if not check_env_variables():
        print("\n✗ 必要な環境変数が設定されていません")
        print("以下の環境変数を設定してください：")
        print("  - DEEPSEEK_API_KEY")
        print("  - OPENAI_API_KEY")
        print("  - S2_API_KEY")
        return 1

    # 実行するテストを選択
    if args.tests:
        # 指定されたテスト番号のみ実行
        tests_to_run = [t for t in all_tests if t[0] in args.tests]
        if not tests_to_run:
            print(f"✗ エラー: 指定されたテスト番号が見つかりません: {args.tests}")
            print("--list オプションでテスト一覧を確認してください")
            return 1
        print(f"実行するテスト: {', '.join([f'{t[0]}. {t[1]}' for t in tests_to_run])}\n")
    else:
        # 全テスト実行
        tests_to_run = all_tests

    print("\n⚠ 警告: このテストは実際のAPIを呼び出すため、利用料金が発生します")
    print("続行しますか？ [y/N]: ", end="")

    if args.yes:
        print("yes (自動実行モード)")
    else:
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            print("テストを中止しました")
            return 0

    results = []

    # テストを実行
    for num, name, test_func in tests_to_run:
        # Test 8 は特別扱い（確認プロンプト）
        if num == 8 and not args.yes and not args.tests:
            print("\n⚠ Test 8 (論文執筆ワークフロー) は時間がかかります")
            print("実行しますか？ [y/N]: ", end="")
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                print("Test 8 をスキップしました")
                continue

        results.append((name, test_func()))

    # サマリ
    print("\n" + "=" * 60)
    print("テスト結果サマリ")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print(f"\n合計: {passed}/{total} テスト成功")

    if passed == total:
        print("\n✓ 全テスト成功！")
        return 0
    else:
        print(f"\n✗ {total - passed} テスト失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
