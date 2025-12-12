"""
AI-MASIST 統合ランチャー
AI-Scientist-v2/launch_scientist_bfts.py から移植

使用例:
    python launch_masist.py --load_ideas masist/ideas/multi_agent_simulation.json
    python launch_masist.py --load_ideas masist/ideas/multi_agent_simulation.json --idea_idx 1
    python launch_masist.py --load_ideas masist/ideas/multi_agent_simulation.json --skip_writeup --skip_review
"""
import os
import os.path as osp
import json
import argparse
import shutil
import re
import sys
import yaml
from datetime import datetime
from contextlib import contextmanager

from masist.llm import create_client
from masist.vlm import create_client as create_vlm_client
from masist.treesearch.perform_experiments_with_agentmanager import (
    perform_experiments_bfts,
)
from masist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_masist_config_file,
)
from masist.perform_plotting import aggregate_plots
from masist.perform_writeup import perform_writeup, gather_citations
from masist.perform_review import perform_review, load_paper
from masist.perform_vlm_review import perform_imgs_cap_ref_review
from masist.utils.token_tracker import token_tracker


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    """トークン使用量をJSONファイルに保存"""
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f, indent=2)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f, indent=2)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI-MASIST experiments")

    # === 基本設定 ===
    parser.add_argument(
        "--load_ideas",
        type=str,
        required=False,
        help="Path to a JSON file containing pregenerated ideas (array format)",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run from the ideas array",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="masist_config.yaml",
        help="Path to MASIST configuration file",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as load_ideas file but .py extension",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )

    # === 論文設定 ===
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )

    # === モデル設定 ===
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="deepseek-chat",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="deepseek-reasoner",
        help="Model to use for writeup (reasoning model recommended)",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="deepseek-chat",
        help="Smaller model to use for writeup subtasks",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="deepseek-chat",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="deepseek-chat",
        help="Model to use for review main text and captions",
    )

    # === スキップオプション ===
    parser.add_argument(
        "--skip_experiments",
        action="store_true",
        help="If set, skip Tree Search experiments (use existing results)",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    parser.add_argument(
        "--only_review",
        type=str,
        default=None,
        help="Run only review on existing experiment folder (e.g., experiments/2025-12-11_...)",
    )
    parser.add_argument(
        "--from_plotting",
        type=str,
        default=None,
        help="Run from plot aggregation on existing experiment folder",
    )
    parser.add_argument(
        "--from_writeup",
        type=str,
        default=None,
        help="Run from writeup on existing experiment folder (skips plot aggregation)",
    )

    return parser.parse_args()


def find_pdf_path_for_review(idea_dir):
    """レビュー用のPDFファイルを検索"""
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    else:
        # No reflection PDFs, use the first PDF found
        pdf_path = osp.join(idea_dir, pdf_files[0])

    return pdf_path


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    """標準出力・エラーをファイルにリダイレクト"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


def cleanup_processes():
    """子プロセスをクリーンアップ"""
    import psutil
    import signal

    print("Start cleaning up processes")

    # Get the current process and all its children
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # First try graceful termination
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait briefly for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # If any processes remain, force kill them
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Additional cleanup: find any orphaned processes containing specific keywords
    # Note: 'torch' removed since MASIST is LLM API-based
    keywords = ["python", "mp", "bfts", "experiment", "masist"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Check both process name and command line arguments
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue


def run_review_only(idea_dir, config_path, model_review):
    """既存の実験フォルダに対してレビューのみ実行"""
    print_time()
    print(f"Running review only on: {idea_dir}")

    pdf_path = find_pdf_path_for_review(idea_dir)
    if not pdf_path or not os.path.exists(pdf_path):
        print("No PDF found for review.")
        return

    print(f"Paper found at: {pdf_path}")

    # テキストレビュー
    paper_content = load_paper(pdf_path)
    client, client_model = create_client(model_review)
    review_text = perform_review(paper_content, client_model, client)

    # VLMレビュー（configのvlm_feedback.modelを使用）
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    vlm_model = cfg.get("treesearch", {}).get("vlm_feedback", {}).get("model", "gpt-4o-mini")
    vlm_client, vlm_model = create_vlm_client(vlm_model)
    review_img_cap_ref = perform_imgs_cap_ref_review(
        vlm_client, vlm_model, pdf_path
    )

    # レビュー結果保存
    with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
        f.write(json.dumps(review_text, indent=4, ensure_ascii=False))
    with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
        json.dump(review_img_cap_ref, f, indent=4, ensure_ascii=False)

    print("Paper review completed.")


def run_from_plotting(idea_dir, args):
    """既存の実験フォルダからプロット集約以降を実行"""
    print_time()
    print(f"Running from plotting: {idea_dir}")

    # experiment_results が idea_dir にない場合、logs からコピー
    exp_results_src = osp.join(idea_dir, "logs/0-run/experiment_results")
    exp_results_dst = osp.join(idea_dir, "experiment_results")
    if os.path.exists(exp_results_src) and not os.path.exists(exp_results_dst):
        print("Copying experiment_results from logs/0-run...")
        shutil.copytree(exp_results_src, exp_results_dst, dirs_exist_ok=True)

    # summary JSONが存在するか確認
    log_dir = osp.join(idea_dir, "logs/0-run")
    if not os.path.exists(log_dir):
        print(f"Error: {log_dir} not found")
        return False

    # === プロット集約 ===
    print_time()
    print("Running plot aggregation...")
    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    # === 論文執筆以降 ===
    return run_from_writeup(idea_dir, args)


def run_from_writeup(idea_dir, args):
    """既存の実験フォルダからwriteup以降を実行"""
    print_time()
    print(f"Running from writeup: {idea_dir}")

    # === 論文執筆 ===
    if not args.skip_writeup:
        print_time()
        print("Starting paper writeup...")

        # 引用収集
        print("Gathering citations...")
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )

        # 論文生成（リトライ付き）
        writeup_success = False
        page_limit = 8 if args.writeup_type == "normal" else 4

        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
            writeup_success = perform_writeup(
                base_folder=idea_dir,
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=page_limit,
                citations_text=citations_text,
            )
            if writeup_success:
                print("Writeup completed successfully!")
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")
            return False
    else:
        print("Skipping writeup (--skip_writeup)")

    # === レビュー ===
    if not args.skip_review and not args.skip_writeup:
        run_review_only(idea_dir, args.config, args.model_review)
    else:
        print("Skipping review (--skip_review or --skip_writeup)")

    return True


if __name__ == "__main__":
    args = parse_arguments()

    # 環境変数設定
    os.environ["AI_MASIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_MASIST_ROOT to {os.environ['AI_MASIST_ROOT']}")

    # --only_review モード
    if args.only_review:
        if not os.path.isdir(args.only_review):
            print(f"Error: {args.only_review} is not a valid directory")
            sys.exit(1)
        run_review_only(args.only_review, args.config, args.model_review)
        sys.exit(0)

    # --from_plotting モード
    if args.from_plotting:
        if not os.path.isdir(args.from_plotting):
            print(f"Error: {args.from_plotting} is not a valid directory")
            sys.exit(1)
        success = run_from_plotting(args.from_plotting, args)
        cleanup_processes()
        sys.exit(0 if success else 1)

    # --from_writeup モード
    if args.from_writeup:
        if not os.path.isdir(args.from_writeup):
            print(f"Error: {args.from_writeup} is not a valid directory")
            sys.exit(1)
        success = run_from_writeup(args.from_writeup, args)
        cleanup_processes()
        sys.exit(0 if success else 1)

    # 通常モードでは --load_ideas が必須
    if not args.load_ideas:
        print("Error: --load_ideas is required (unless using --only_review, --from_plotting, or --from_writeup)")
        sys.exit(1)

    # アイデア読み込み（配列形式）
    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
    if not isinstance(ideas, list):
        ideas = [ideas]  # 単体JSONの場合は配列にラップ
    if args.idea_idx >= len(ideas):
        print(f"Error: idea_idx {args.idea_idx} is out of range (0-{len(ideas)-1})")
        sys.exit(1)
    idea = ideas[args.idea_idx]
    print(f"Loaded idea {args.idea_idx} from {args.load_ideas}")

    # 実験ディレクトリ作成
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_name = idea.get("Name", "unnamed")
    idea_dir = f"experiments/{date}_{idea_name}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # アイデアをMarkdownに変換
    idea_path_md = osp.join(idea_dir, "idea.md")

    # コード読み込み（オプション）
    code = None
    code_path = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
            print(f"Loaded code from {code_path}")
        else:
            print(f"Warning: Code file {code_path} not found")
            code_path = None

    idea_to_markdown(idea, idea_path_md, code_path)

    # コードをアイデアに追加
    if code is not None:
        idea["Code"] = code

    # アイデアJSONを保存
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(idea, f, indent=4, ensure_ascii=False)

    # 設定ファイル編集
    idea_config_path = edit_masist_config_file(
        args.config,
        idea_dir,
        idea_path_json,
    )

    # === [4] Tree Search 実験実行 ===
    if not args.skip_experiments:
        print_time()
        print("Starting Tree Search experiments...")
        perform_experiments_bfts(idea_config_path)

        # experiment_results をコピー
        experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
        if os.path.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                osp.join(idea_dir, "experiment_results"),
                dirs_exist_ok=True,
            )
    else:
        print("Skipping Tree Search experiments (--skip_experiments)")

    # === [5] プロット集約 ===
    print_time()
    print("Aggregating plots...")
    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)

    # experiment_results を削除（プロット集約後）
    exp_results_in_idea = osp.join(idea_dir, "experiment_results")
    if os.path.exists(exp_results_in_idea):
        shutil.rmtree(exp_results_in_idea)

    # === [6] トークン使用量保存 ===
    save_token_tracker(idea_dir)

    # === [7] 論文執筆 ===
    if not args.skip_writeup:
        print_time()
        print("Starting paper writeup...")

        # 引用収集
        print("Gathering citations...")
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )

        # 論文生成（リトライ付き）
        writeup_success = False
        page_limit = 8 if args.writeup_type == "normal" else 4

        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
            writeup_success = perform_writeup(
                base_folder=idea_dir,
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=page_limit,
                citations_text=citations_text,
            )
            if writeup_success:
                print("Writeup completed successfully!")
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")
    else:
        print("Skipping writeup (--skip_writeup)")

    # === [8] トークン使用量保存 ===
    save_token_tracker(idea_dir)

    # === [9] レビュー ===
    if not args.skip_review and not args.skip_writeup:
        print_time()
        print("Starting paper review...")

        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print(f"Paper found at: {pdf_path}")

            # テキストレビュー
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)

            # VLMレビュー（configのvlm_feedback.modelを使用）
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            vlm_model = cfg.get("treesearch", {}).get("vlm_feedback", {}).get("model", "gpt-4o-mini")
            vlm_client, vlm_model = create_vlm_client(vlm_model)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                vlm_client, vlm_model, pdf_path
            )

            # レビュー結果保存
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4, ensure_ascii=False))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4, ensure_ascii=False)

            print("Paper review completed.")
        else:
            print("No PDF found for review.")
    else:
        print("Skipping review (--skip_review or --skip_writeup)")

    # === [10] クリーンアップ ===
    cleanup_processes()

    print_time()
    print(f"All done! Results saved in {idea_dir}")
    sys.exit(0)
