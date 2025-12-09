"""
論文執筆機能の統合テスト

このスクリプトは以下をテストします：
1. モジュールのインポート
2. 各関数の基本的な動作確認
3. ダミーデータでの実行（実際のAPI呼び出しなし）
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# masist モジュールをインポート可能にする
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """モジュールのインポートテスト"""
    print("=" * 60)
    print("TEST 1: モジュールインポート確認")
    print("=" * 60)

    try:
        from masist.llm import create_client, get_response_from_llm, AVAILABLE_LLMS
        print("✓ masist.llm インポート成功")

        # deepseek-reasoner がリストにあるか確認
        if "deepseek-reasoner" in AVAILABLE_LLMS:
            print("✓ deepseek-reasoner が AVAILABLE_LLMS に存在")
        else:
            print("✗ deepseek-reasoner が AVAILABLE_LLMS に存在しない")

        if "deepseek-chat" in AVAILABLE_LLMS:
            print("✓ deepseek-chat が AVAILABLE_LLMS に存在")
        else:
            print("✗ deepseek-chat が AVAILABLE_LLMS に存在しない")

    except ImportError as e:
        print(f"✗ masist.llm インポート失敗: {e}")
        return False

    try:
        from masist.vlm import create_client as create_vlm_client, AVAILABLE_VLMS
        print("✓ masist.vlm インポート成功")

        if "gpt-4o-mini" in AVAILABLE_VLMS:
            print("✓ gpt-4o-mini が AVAILABLE_VLMS に存在")
        else:
            print("✗ gpt-4o-mini が AVAILABLE_VLMS に存在しない")

    except ImportError as e:
        print(f"✗ masist.vlm インポート失敗: {e}")
        return False

    try:
        from masist.tools.semantic_scholar import search_for_papers
        print("✓ masist.tools.semantic_scholar インポート成功")
    except ImportError as e:
        print(f"✗ masist.tools.semantic_scholar インポート失敗: {e}")
        return False

    try:
        from masist.perform_writeup import (
            perform_writeup,
            gather_citations,
            load_exp_summaries,
            compile_latex,
        )
        print("✓ masist.perform_writeup インポート成功")
    except ImportError as e:
        print(f"✗ masist.perform_writeup インポート失敗: {e}")
        return False

    try:
        from masist.perform_review import perform_review, load_paper
        print("✓ masist.perform_review インポート成功")
    except ImportError as e:
        print(f"✗ masist.perform_review インポート失敗: {e}")
        return False

    try:
        from masist.perform_vlm_review import (
            perform_imgs_cap_ref_review,
            generate_vlm_img_review,
            detect_duplicate_figures,
        )
        print("✓ masist.perform_vlm_review インポート成功")
    except ImportError as e:
        print(f"✗ masist.perform_vlm_review インポート失敗: {e}")
        return False

    print("\n✓ 全モジュールのインポートに成功\n")
    return True


def test_load_exp_summaries():
    """load_exp_summaries() のテスト"""
    print("=" * 60)
    print("TEST 2: load_exp_summaries() テスト")
    print("=" * 60)

    from masist.perform_writeup import load_exp_summaries

    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        # ダミーのサマリファイルを作成
        baseline_summary = {
            "best node": {
                "overall_plan": "Test baseline plan",
                "analysis": "Test baseline analysis",
                "metric": 0.85,
            }
        }

        research_summary = {
            "best node": {
                "overall_plan": "Test research plan",
                "analysis": "Test research analysis",
                "metric": 0.92,
            }
        }

        # ファイルに保存（AI-MASIST形式: logs/0-run/ プレフィックスなし）
        with open(os.path.join(tmpdir, "baseline_summary.json"), "w") as f:
            json.dump(baseline_summary, f)

        with open(os.path.join(tmpdir, "research_summary.json"), "w") as f:
            json.dump(research_summary, f)

        # load_exp_summaries() を実行
        summaries = load_exp_summaries(tmpdir)

        # 確認
        if "BASELINE_SUMMARY" in summaries and "RESEARCH_SUMMARY" in summaries:
            print("✓ サマリファイルの読み込み成功")
            print(f"  - BASELINE_SUMMARY: {summaries['BASELINE_SUMMARY']}")
            print(f"  - RESEARCH_SUMMARY: {summaries['RESEARCH_SUMMARY']}")
            return True
        else:
            print("✗ サマリファイルの読み込み失敗")
            return False


def test_latex_template_path():
    """LaTeX テンプレートパスの確認"""
    print("=" * 60)
    print("TEST 3: LaTeX テンプレートパス確認")
    print("=" * 60)

    try:
        import importlib.resources

        # Python 3.9+
        try:
            blank_latex_path = importlib.resources.files('masist').joinpath('blank_icbinb_latex')
            print(f"✓ LaTeX テンプレートパス (Python 3.9+): {blank_latex_path}")

            # ファイルが存在するか確認
            with importlib.resources.as_file(blank_latex_path) as template_path:
                if os.path.exists(template_path):
                    files = os.listdir(template_path)
                    print(f"✓ テンプレートディレクトリ内容: {files}")

                    if "template.tex" in files:
                        print("✓ template.tex が存在")
                        return True
                    else:
                        print("✗ template.tex が存在しない")
                        return False
                else:
                    print("✗ テンプレートディレクトリが存在しない")
                    return False

        except AttributeError:
            # Python 3.7-3.8 fallback
            import pkg_resources
            blank_latex_path = pkg_resources.resource_filename('masist', 'blank_icbinb_latex')
            print(f"✓ LaTeX テンプレートパス (Python 3.7-3.8): {blank_latex_path}")

            if os.path.exists(blank_latex_path):
                files = os.listdir(blank_latex_path)
                print(f"✓ テンプレートディレクトリ内容: {files}")

                if "template.tex" in files:
                    print("✓ template.tex が存在")
                    return True
                else:
                    print("✗ template.tex が存在しない")
                    return False
            else:
                print("✗ テンプレートディレクトリが存在しない")
                return False

    except Exception as e:
        print(f"✗ LaTeX テンプレートパスの確認失敗: {e}")
        return False


def test_fewshot_examples():
    """fewshot_examples の確認"""
    print("=" * 60)
    print("TEST 4: fewshot_examples 確認")
    print("=" * 60)

    try:
        import importlib.resources

        # Python 3.9+
        try:
            fewshot_path = importlib.resources.files('masist').joinpath('fewshot_examples')
            print(f"✓ fewshot_examples パス (Python 3.9+): {fewshot_path}")

            with importlib.resources.as_file(fewshot_path) as examples_path:
                if os.path.exists(examples_path):
                    files = os.listdir(examples_path)
                    print(f"✓ fewshot_examples 内容: {files}")

                    required_files = [
                        "132_automated_relational.pdf",
                        "132_automated_relational.json",
                        "attention.pdf",
                        "attention.json",
                    ]

                    missing = [f for f in required_files if f not in files]
                    if not missing:
                        print("✓ 必要なファイルが全て存在")
                        return True
                    else:
                        print(f"✗ 欠落ファイル: {missing}")
                        return False
                else:
                    print("✗ fewshot_examples ディレクトリが存在しない")
                    return False

        except AttributeError:
            # Python 3.7-3.8 fallback
            import pkg_resources
            fewshot_path = pkg_resources.resource_filename('masist', 'fewshot_examples')
            print(f"✓ fewshot_examples パス (Python 3.7-3.8): {fewshot_path}")

            if os.path.exists(fewshot_path):
                files = os.listdir(fewshot_path)
                print(f"✓ fewshot_examples 内容: {files}")

                required_files = [
                    "132_automated_relational.pdf",
                    "132_automated_relational.json",
                    "attention.pdf",
                    "attention.json",
                ]

                missing = [f for f in required_files if f not in files]
                if not missing:
                    print("✓ 必要なファイルが全て存在")
                    return True
                else:
                    print(f"✗ 欠落ファイル: {missing}")
                    return False
            else:
                print("✗ fewshot_examples ディレクトリが存在しない")
                return False

    except Exception as e:
        print(f"✗ fewshot_examples の確認失敗: {e}")
        return False


def test_config_file():
    """設定ファイルの確認"""
    print("=" * 60)
    print("TEST 5: masist_config.yaml 確認")
    print("=" * 60)

    config_path = Path(__file__).parent.parent / "masist_config.yaml"

    if not config_path.exists():
        print(f"✗ 設定ファイルが存在しない: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 必要な設定項目を確認
        required_keys = [
            "perform_writeup",
            "writeup_small_model",
            "writeup_big_model",
            "writeup_reflections",
            "writeup_page_limit",
            "gather_citations",
            "num_cite_rounds",
            "perform_review",
            "review_model",
            "vlm_model",
        ]

        missing_keys = [k for k in required_keys if k not in config]

        if missing_keys:
            print(f"✗ 欠落している設定項目: {missing_keys}")
            return False
        else:
            print("✓ 全ての必要な設定項目が存在")
            print(f"  - perform_writeup: {config['perform_writeup']}")
            print(f"  - writeup_small_model: {config['writeup_small_model']}")
            print(f"  - writeup_big_model: {config['writeup_big_model']}")
            print(f"  - perform_review: {config['perform_review']}")
            print(f"  - review_model: {config['review_model']}")
            print(f"  - vlm_model: {config['vlm_model']}")
            return True

    except ImportError:
        print("✗ PyYAML がインストールされていません")
        print("  インストールコマンド: pip install pyyaml")
        return False
    except Exception as e:
        print(f"✗ 設定ファイルの読み込み失敗: {e}")
        return False


def test_llm_create_client():
    """create_client() の基本テスト（API呼び出しなし）"""
    print("=" * 60)
    print("TEST 6: create_client() 基本テスト")
    print("=" * 60)

    from masist.llm import create_client

    # 環境変数が設定されていない場合のテスト
    test_models = [
        ("deepseek-chat", "DeepSeek"),
        ("deepseek-reasoner", "DeepSeek"),
        ("gpt-4o-mini", "OpenAI"),
    ]

    results = []
    for model, expected_provider in test_models:
        try:
            # 環境変数を一時的に設定（ダミー値）
            if model.startswith("deepseek"):
                os.environ.setdefault("DEEPSEEK_API_KEY", "dummy_key")
            elif model.startswith("gpt"):
                os.environ.setdefault("OPENAI_API_KEY", "dummy_key")

            client, client_model = create_client(model)
            print(f"✓ {model}: クライアント作成成功 (provider: {expected_provider})")
            results.append(True)
        except Exception as e:
            print(f"✗ {model}: クライアント作成失敗: {e}")
            results.append(False)

    return all(results)


def main():
    """全テストを実行"""
    print("\n" + "=" * 60)
    print("AI-MASIST 論文執筆機能 統合テスト")
    print("=" * 60 + "\n")

    results = []

    # Test 1: モジュールインポート
    results.append(("モジュールインポート", test_imports()))

    # Test 2: load_exp_summaries
    results.append(("load_exp_summaries", test_load_exp_summaries()))

    # Test 3: LaTeX テンプレートパス
    results.append(("LaTeX テンプレート", test_latex_template_path()))

    # Test 4: fewshot_examples
    results.append(("fewshot_examples", test_fewshot_examples()))

    # Test 5: 設定ファイル
    results.append(("設定ファイル", test_config_file()))

    # Test 6: create_client
    results.append(("create_client", test_llm_create_client()))

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
