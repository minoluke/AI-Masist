"""
Individual Phase Testing Script
各フェーズを個別にテストするためのスクリプト
"""

import json
import logging
import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to sys.path for imports to work from both locations
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

from masist.treesearch.utils.config import _load_cfg, prep_cfg, Config
import shutil

# 実験設定を fixtures からインポート
from fixtures.experiments import get_experiment

# テスト用設定ファイルパス
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"


def create_test_config(workspace_name: str = "test_phases") -> Config:
    """Create a test configuration from YAML file (AI-Scientist-v2準拠)"""
    cfg = _load_cfg(TEST_CONFIG_PATH)
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.goal = "Test goal for phases"
    cfg.exec.timeout = 300  # 5 minutes for phase tests
    cfg = prep_cfg(cfg)
    return cfg


def cleanup_test_dirs(cfg: Config):
    """Clean up test directories after tests"""
    try:
        if hasattr(cfg, 'log_dir') and Path(cfg.log_dir).exists():
            shutil.rmtree(cfg.log_dir)
        if hasattr(cfg, 'workspace_dir') and Path(cfg.workspace_dir).exists():
            shutil.rmtree(cfg.workspace_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up test directories: {e}")


# TPGG 実験を使用
_exp = get_experiment("tpgg")
SAMPLE_TASK_DESC = json.dumps(_exp["task_desc"], ensure_ascii=False, indent=2)
SAMPLE_METRICS = _exp["metrics"]


def test_phase1_code_generation():
    """Phase 1: コード生成のテスト"""
    print("\n" + "=" * 80)
    print("Phase 1: Code Generation Test")
    print("=" * 80)

    from masist.treesearch.code_generator import CodeGenerator

    cfg = create_test_config()
    generator = CodeGenerator(
        task_desc=SAMPLE_TASK_DESC,
        evaluation_metrics=SAMPLE_METRICS,
        cfg=cfg,
        memory_summary=None,
    )

    print("Generating code...")
    node = generator.generate()

    print(f"\n✓ Plan generated ({len(node.plan)} chars):")
    print(node.plan[:500] + "..." if len(node.plan) > 500 else node.plan)

    print(f"\n✓ Code generated ({len(node.code)} chars):")
    print(node.code[:1000] + "..." if len(node.code) > 1000 else node.code)

    # Save generated code
    output_dir = Path("./test_output/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "generated_code.py", "w") as f:
        f.write(node.code)
    with open(output_dir / "plan.txt", "w") as f:
        f.write(node.plan)

    print(f"\n✓ Files saved to {output_dir}/")
    return node


def test_phase2_execution(node=None):
    """Phase 2: コード実行のテスト"""
    print("\n" + "=" * 80)
    print("Phase 2: Code Execution Test")
    print("=" * 80)

    from masist.treesearch.interpreter import Interpreter
    from masist.treesearch import Node

    # Load code if not provided
    if node is None:
        code_path = Path("./test_output/phase1/generated_code.py")
        if not code_path.exists():
            print("❌ No generated code found. Run phase 1 first.")
            return None

        with open(code_path, "r") as f:
            code = f.read()

        node = Node(code=code)

    # Setup working directory
    working_dir = "./test_output/phase2_working"
    os.makedirs(working_dir, exist_ok=True)

    # Inject working directory
    code_with_wd = f"""import os
working_dir = '{working_dir}'
os.makedirs(working_dir, exist_ok=True)

{node.code}
"""

    cfg = create_test_config()
    interpreter = Interpreter(working_dir=working_dir, timeout=cfg.exec.timeout)

    print("Executing code...")
    exec_result = interpreter.run(code_with_wd, reset_session=True)
    interpreter.cleanup_session()

    print(f"\n✓ Execution completed in {exec_result.exec_time:.2f}s")

    if exec_result.exc_type:
        print(f"❌ Exception: {exec_result.exc_type}")
        print(f"   Info: {exec_result.exc_info}")
    else:
        print("✓ No exceptions")

    print(f"\n--- Output (first 500 chars) ---")
    print(exec_result.term_out[:500] if exec_result.term_out else "(no output)")

    # Save execution results for reuse by later phases
    result_cache_path = Path("./test_output/phase2_result.pkl")
    with open(result_cache_path, "wb") as f:
        pickle.dump({"exec_result": exec_result, "node": node}, f)
    print(f"\n✓ Saved execution results to {result_cache_path}")

    return exec_result, node


def test_phase3_evaluation(node=None, exec_result=None):
    """Phase 3: 結果評価のテスト"""
    print("\n" + "=" * 80)
    print("Phase 3: Result Evaluation Test")
    print("=" * 80)

    from masist.treesearch.result_evaluator import ResultEvaluator
    from masist.treesearch import Node

    # Load from cache if not provided
    if node is None or exec_result is None:
        result_cache_path = Path("./test_output/phase2_result.pkl")
        if not result_cache_path.exists():
            print("❌ No cached execution results found. Run phase 2 first.")
            return None

        print(f"Loading cached execution results from {result_cache_path}")
        with open(result_cache_path, "rb") as f:
            cached_data = pickle.load(f)
            exec_result = cached_data["exec_result"]
            node = cached_data["node"]

    cfg = create_test_config()
    evaluator = ResultEvaluator(task_desc=SAMPLE_TASK_DESC, cfg=cfg)

    working_dir = "./test_output/phase2_working/working"
    print("Evaluating results...")
    evaluator.evaluate(node, exec_result, workspace=working_dir)

    print(f"\n✓ Is buggy: {node.is_buggy}")
    print(f"✓ Analysis:\n{node.analysis}")

    return node


def test_phase4_metrics_extraction(node=None):
    """Phase 4: メトリクス抽出のテスト"""
    print("\n" + "=" * 80)
    print("Phase 4: Metrics Extraction Test")
    print("=" * 80)

    from masist.treesearch.metrics_extractor import MetricsExtractor
    from masist.treesearch.interpreter import Interpreter

    # Load from cache if not provided
    if node is None:
        result_cache_path = Path("./test_output/phase2_result.pkl")
        if not result_cache_path.exists():
            print("❌ No cached execution results found. Run phase 2 first.")
            return None

        print(f"Loading cached execution results from {result_cache_path}")
        with open(result_cache_path, "rb") as f:
            cached_data = pickle.load(f)
            node = cached_data["node"]

    base_dir = "./test_output/phase2_working"
    working_dir = os.path.join(base_dir, "working")
    if not Path(working_dir).exists():
        print(f"❌ Working directory {working_dir} not found.")
        return None

    cfg = create_test_config()
    interpreter = Interpreter(working_dir=base_dir, timeout=cfg.exec.timeout)
    extractor = MetricsExtractor(cfg=cfg, interpreter=interpreter)

    print("Extracting metrics...")
    extractor.extract(node, working_dir)

    print(f"\n✓ Metric: {node.metric}")
    if hasattr(node, "parse_term_out") and node.parse_term_out:
        print(f"✓ Parse output:\n{node.parse_term_out[:500]}")

    return node


def test_phase5_plot_generation(node=None):
    """Phase 5: プロット生成のテスト"""
    print("\n" + "=" * 80)
    print("Phase 5: Plot Generation Test")
    print("=" * 80)

    from masist.treesearch.plot_generator import PlotGenerator
    from masist.treesearch.interpreter import Interpreter

    # Load from cache if not provided
    if node is None:
        result_cache_path = Path("./test_output/phase2_result.pkl")
        if not result_cache_path.exists():
            print("❌ No cached execution results found. Run phase 2 first.")
            return None

        print(f"Loading cached execution results from {result_cache_path}")
        with open(result_cache_path, "rb") as f:
            cached_data = pickle.load(f)
            node = cached_data["node"]

    base_dir = "./test_output/phase2_working"
    working_dir = os.path.join(base_dir, "working")
    if not Path(working_dir).exists():
        print(f"❌ Working directory {working_dir} not found.")
        return None

    cfg = create_test_config()
    interpreter = Interpreter(working_dir=base_dir, timeout=cfg.exec.timeout)
    generator = PlotGenerator(cfg=cfg, interpreter=interpreter)

    print("Generating plots...")
    generator.generate_and_execute(node, working_dir)

    print(f"\n✓ Plots generated: {len(node.plots) if node.plots else 0}")
    if node.plots:
        for i, plot in enumerate(node.plots, 1):
            print(f"   {i}. {plot}")

    return node


def test_phase6_vlm_analysis(node=None):
    """Phase 6: VLM分析のテスト"""
    print("\n" + "=" * 80)
    print("Phase 6: VLM Analysis Test")
    print("=" * 80)

    from masist.treesearch.vlm_analyzer import VLMAnalyzer

    if node is None or not node.plots:
        print("❌ Need node with generated plots. Run phases 1-5 first.")
        return None

    cfg = create_test_config()
    analyzer = VLMAnalyzer(task_desc=SAMPLE_TASK_DESC, cfg=cfg)

    print("Analyzing plots with VLM...")
    analyzer.analyze(node)

    print(f"\n✓ Analyses: {len(node.plot_analyses) if hasattr(node, 'plot_analyses') else 0}")
    if hasattr(node, "vlm_feedback_summary"):
        print(f"✓ VLM Feedback:\n{node.vlm_feedback_summary[:500]}")

    return node


def run_all_phases():
    """全フェーズを順次実行"""
    print("\n" + "=" * 80)
    print("Running All Phases Sequentially")
    print("=" * 80)

    try:
        # Phase 1
        node = test_phase1_code_generation()

        # Phase 2
        exec_result, node = test_phase2_execution(node)

        # Phase 3
        node = test_phase3_evaluation(node, exec_result)

        # Phase 4
        node = test_phase4_metrics_extraction(node)

        # Phase 5
        node = test_phase5_plot_generation(node)

        # Phase 6
        node = test_phase6_vlm_analysis(node)

        print("\n" + "=" * 80)
        print("All Phases Completed Successfully!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


def main():
    """メイン関数：コマンドライン引数でフェーズを指定"""
    if len(sys.argv) < 2:
        print("Usage: python test_phases.py <phase_number|all>")
        print("\nAvailable phases:")
        print("  1 - Code Generation")
        print("  2 - Code Execution")
        print("  3 - Result Evaluation")
        print("  4 - Metrics Extraction")
        print("  5 - Plot Generation")
        print("  6 - VLM Analysis")
        print("  all - Run all phases sequentially")
        print("\nExample:")
        print("  python test_phases.py 1")
        print("  python test_phases.py all")
        sys.exit(1)

    phase = sys.argv[1].lower()

    # Create output directory
    Path("./test_output").mkdir(exist_ok=True)

    if phase == "1":
        test_phase1_code_generation()
    elif phase == "2":
        test_phase2_execution()
    elif phase == "3":
        # Phase 3 will load cached results automatically
        test_phase3_evaluation()
    elif phase == "4":
        # Phase 4 will load cached results automatically
        test_phase4_metrics_extraction()
    elif phase == "5":
        # Phase 5 will load cached results automatically
        test_phase5_plot_generation()
    elif phase == "6":
        # Phase 6 will load cached results automatically
        node = test_phase5_plot_generation()
        test_phase6_vlm_analysis(node)
    elif phase == "all":
        run_all_phases()
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
