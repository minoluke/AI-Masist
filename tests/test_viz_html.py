"""
HTML可視化のテスト用スクリプト

使い方:
    # テスト用HTMLを生成（すべてのステージデータを埋め込み）
    python tests/test_viz_html.py

    # 特定の実験ディレクトリを指定
    python tests/test_viz_html.py experiments/2025-12-10_xxx/logs/0-run

    # ローカルサーバーを起動して確認（log_dirで起動）
    cd experiments/2025-12-10_xxx/logs/0-run
    python -m http.server 8000
    # ブラウザで http://localhost:8000/test_viz.html を開く
"""

import json
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_log_dir() -> Path | None:
    """最新の実験のログディレクトリを探す"""
    experiments_dir = project_root / "experiments"
    if not experiments_dir.exists():
        return None

    # 最新の実験ディレクトリを探す
    exp_dirs = sorted(experiments_dir.iterdir(), reverse=True)
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        # logs/0-run ディレクトリを探す
        logs_dir = exp_dir / "logs" / "0-run"
        if logs_dir.exists():
            return logs_dir

    return None


def collect_all_stage_data(log_dir: Path) -> dict:
    """
    ログディレクトリからすべてのステージのtree_dataを収集する

    Returns:
        {
            "all_stages": {
                "Stage_1": {...},
                "Stage_2": {...},
                ...
            },
            "completed_stages": ["Stage_1", "Stage_2", ...],
            "current_stage": "Stage_2"
        }
    """
    all_stages = {}
    completed_stages = []
    current_stage = None

    # stage_* ディレクトリを探す
    stage_dirs = sorted([
        d for d in log_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage_")
    ])

    for stage_dir in stage_dirs:
        tree_data_path = stage_dir / "tree_data.json"
        if not tree_data_path.exists():
            continue

        # ステージ番号を抽出 (stage_1_xxx -> Stage_1)
        parts = stage_dir.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            stage_id = f"Stage_{parts[1]}"
        else:
            continue

        # tree_dataを読み込む
        with open(tree_data_path) as f:
            tree_data = json.load(f)

        all_stages[stage_id] = tree_data
        completed_stages.append(stage_id)
        current_stage = stage_id  # 最後のステージをcurrent_stageとする

        print(f"  Loaded: {stage_dir.name} -> {stage_id}")

    return {
        "all_stages": all_stages,
        "completed_stages": completed_stages,
        "current_stage": current_stage,
    }


def generate_embedded_html(combined_data: dict) -> str:
    """
    すべてのステージデータを埋め込んだHTMLを生成する
    """
    template_dir = project_root / "masist" / "treesearch" / "utils" / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()

    with open(template_dir / "template.html") as f:
        html = f.read()

    # プレースホルダーを置換
    js = js.replace('"PLACEHOLDER_TREE_DATA"', json.dumps(combined_data))

    # HTMLにJSを埋め込む
    html = html.replace("<!-- placeholder -->", js)

    return html


def generate_test_html(log_dir_path: Path | str | None = None, output_path: Path | str | None = None):
    """
    テスト用HTMLを生成する（すべてのステージデータを埋め込み）

    Args:
        log_dir_path: ログディレクトリのパス (experiments/xxx/logs/0-run)
        output_path: 出力先のパス
    """
    # ログディレクトリのパスを決定
    if log_dir_path is None:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            print("Error: No experiment logs found in experiments/")
            print("Please specify a path: python tests/test_viz_html.py <log_dir_path>")
            return False
    else:
        log_dir = Path(log_dir_path)

    if not log_dir.exists():
        print(f"Error: {log_dir} not found")
        return False

    # 出力先を決定（デフォルトはlog_dir内に出力）
    if output_path is None:
        output_path = log_dir / "test_viz.html"
    else:
        output_path = Path(output_path)

    print(f"Log directory: {log_dir}")
    print("Collecting stage data...")

    # すべてのステージデータを収集
    combined_data = collect_all_stage_data(log_dir)

    if not combined_data["completed_stages"]:
        print("Error: No stage data found")
        return False

    print(f"Found stages: {combined_data['completed_stages']}")
    print(f"Current stage: {combined_data['current_stage']}")

    # HTMLを生成
    html = generate_embedded_html(combined_data)

    # 出力
    with open(output_path, "w") as f:
        f.write(html)

    print()
    print(f"Generated: {output_path}")
    print()
    print("To view:")
    print(f"  cd {output_path.parent}")
    print("  python -m http.server 8000")
    print(f"  # Open http://localhost:8000/{output_path.name}")

    return True


if __name__ == "__main__":
    # コマンドライン引数からログディレクトリのパスを取得
    log_dir_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = generate_test_html(log_dir_path, output_path)
    sys.exit(0 if success else 1)
