"""
Individual Phase Testing Script
各フェーズを個別にテストするためのスクリプト
"""

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

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")
else:
    print(f"⚠ .env file not found at {env_path}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

from unitTest.config import Config


def create_test_config(workspace_name: str = "test_phases") -> Config:
    """Create a test configuration with workspace under project root"""
    config = Config(workspace_name=workspace_name)
    config.exec.timeout = 300  # 5 minutes for phase tests
    return config


# サンプルタスク（MASist向け）
SAMPLE_TASK_DESC = """
シミュレーション検討シート（TPGG）
1. シミュレーション要求
背景・文脈：
4人グループで「一定以上のお金（トークン）をみんなで出し合えたらご褒美がもらえる」というゲームがあります。
 ただし、ご褒美がもらえるライン（＝しきい値）にギリギリ必要な額を目指すのか、
 それより多めの額を要求するのかによって、みんなの出し方が変わると言われています。
この「必要な分ちょうど」と「必要以上に多め」の違いが、
 実際の行動にどう影響するのかを LLM で調べます。

目的：
グループに「出してほしい金額の目安（ルール）」を示したとき、
 そのルールが「ちょうど必要な合計」か「必要以上に多い合計」かによって、
 行動や結果がどう変わるのかを調べる。



研究質問：
ルールが「必要以上に多い（多めの要求）」だと、行動が乱れやすい？


必要な分ちょうどのルールは、むしろ安定した協力を生む？


グループ全体がちょうど必要額に合わせる「効率の良さ」はどう変わる？



仮説：
① 多めに要求されたルールは、守る人が減りやすい。


② 必要額を達成できるかどうかは、ルールの違いではあまり変わらない。


③ 多めのルールは「出しすぎ（無駄）」を増やし、効率を下げる。


④ 1人あたりの負担がピッタリ均等割りできる場合、協力がまとまりやすい。



2. シミュレーション要件

エージェント
人数：
 4人（1グループ）
ロールと説明：
4人とも同じ立場。


各ラウンドで、自分の10トークンのうち何トークンを共同の箱に入れるか選ぶ。


全員の合計がしきい値を超えれば、ご褒美 (V) がもらえる。


記憶・内部状態：
過去の自分の拠出額


グループ合計の拠出額


自分の得点
 -（後半だけ）自分に示された「出してほしい額（ルール）」


更新ルール：ラウンドの最後に、そのラウンドの情報を記録して次のラウンドの参考にする。



プロトコル
ターン/ラウンド構造：
1回のゲームは20ラウンド。


各ラウンドは


4人が同時に出す額を決める


合計額を見る


しきい値を超えたかどうか判定


得点を返す


最大ラウンド数：
 20
終了条件：
 20ラウンド終わったら終了。
各工程の試行数：
1つの設定につき、4人グループを複数回（例：11グループ）まわす。


フェーズ構造（任意）：
ラウンド1〜10：ルールなし


ラウンド11〜20：設定に応じたルールを提示（またはなし）



環境・ルール

ネットワーク構造：
4人は同じグループ


グループ間の交流はなし
 （＝4人だけの閉じた小世界で毎回意思決定）



行動空間 / アクションセット：
桁数：0〜10 の整数から1つ選んで出すだけ。



共有情報：
各自の持ちトークンは10


しきい値 (T)（条件ごとに違う）
 -（後半）みんなの「出してほしい額（ルール）」


ラウンドが終わった後の


合計拠出額


自分の得点



非公開情報：
他のメンバーが実際にいくら出したかは見えない（自分の分は見える）


各メンバーの考え・意図



必要なルール、利得構造：
利得（1人あたり）：
自分が出した額を (c_i)


全員の合計を (C = \sum c_i)


しきい値未達成（C < T）：
 [
 \pi_i = 10 - c_i
 ]
しきい値達成（C \ge T）：
 [
 \pi_i = 10 - c_i + V
 ]
しきい値を超えた分には追加のご褒美なし
 （＝出しすぎは「無駄」）



実験条件（＝比較する設定の一覧）
5つの設定を作る：
必要額ちょうど・均等割り OK（FAIRSUFF）


T = 20


ルール = (5,5,5,5)


多めの要求・均等割り OK（FAIRINF）


T = 20


ルール = (5,5,6,6)


必要額ちょうど・均等割り不可（UNFAIRSUFF）


T = 22


ルール = (5,5,6,6)


多めの要求・均等割り不可（UNFAIRINF）


T = 22


ルール = (6,6,6,6)


ルールなし（CONTROL）


T = 22


ルールなし



ログ・分析指標
ログ形式：
記録すべき内容：
ラウンド番号


各メンバーの出した額


合計出した額


しきい値達成の有無


各メンバーの得点
 -（後半）ルールを守ったかのフラグ


どの設定で行ったか


分析指標：
しきい値達成率（成功の割合）


平均の出した額


必要額よりどれだけ多く出たか（過剰分）


ルールを守った割合


10ラウンド目→11ラウンド目の変化（ルール導入効果）


"""

SAMPLE_METRICS = [
    "cooperation_rate",
    "consensus_rate",
    "average_turns",
    "convergence_time",
]


def test_phase1_code_generation():
    """Phase 1: コード生成のテスト"""
    print("\n" + "=" * 80)
    print("Phase 1: Code Generation Test")
    print("=" * 80)

    try:
        from unitTest.phases import CodeGenerator
    except ModuleNotFoundError:
        from phases import CodeGenerator

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

    try:
        from unitTest.executor.interpreter import Interpreter
    except ModuleNotFoundError:
        from executor.interpreter import Interpreter

    # Load code if not provided
    if node is None:
        code_path = Path("./test_output/phase1/generated_code.py")
        if not code_path.exists():
            print("❌ No generated code found. Run phase 1 first.")
            return None

        with open(code_path, "r") as f:
            code = f.read()

        try:
            from unitTest.core.node import Node
        except ModuleNotFoundError:
            from core.node import Node

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

    try:
        from unitTest.phases import ResultEvaluator
    except ModuleNotFoundError:
        from phases import ResultEvaluator

    try:
        from unitTest.core.node import Node
    except ModuleNotFoundError:
        from core.node import Node

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

    try:
        from unitTest.phases import MetricsExtractor
    except ModuleNotFoundError:
        from phases import MetricsExtractor

    try:
        from unitTest.executor.interpreter import Interpreter
    except ModuleNotFoundError:
        from executor.interpreter import Interpreter

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

    try:
        from unitTest.phases import PlotGenerator
    except ModuleNotFoundError:
        from phases import PlotGenerator

    try:
        from unitTest.executor.interpreter import Interpreter
    except ModuleNotFoundError:
        from executor.interpreter import Interpreter

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

    try:
        from unitTest.phases import VLMAnalyzer
    except ModuleNotFoundError:
        from phases import VLMAnalyzer

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
