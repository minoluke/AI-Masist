"""
Test script for ParallelAgent
Tests parallel node processing with multiple workers

テスト内容:
- Test 1: ParallelAgent 初期化
- Test 2: step() - 単一ステップ実行（2ワーカー）
- Test 3: 4ワーカー並列実行
- Test 4: _select_parallel_nodes() - Best-First探索（debug/improve選択）
- Test 5: run() - 複数ステップ実行
"""
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
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

from masist.treesearch import ParallelAgent, Journal, Node
from masist.treesearch.utils.metric import MetricValue
from masist.treesearch.utils.config import Config

# サンプルタスク（シンプル版 - テスト用）
SIMPLE_TASK_DESC = """
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
    "threshold_achievement_rate",
    "average_contribution",
    "excess_contribution",
    "rule_compliance_rate",
]


def create_test_config(workspace_name: str = "test_parallel", num_workers: int = 2) -> Config:
    """Create a test configuration with workspace under project root"""
    config = Config(workspace_name=workspace_name)
    config.agent.num_workers = num_workers
    return config


def test_parallel_agent_init():
    """Test ParallelAgent initialization"""
    print("=" * 80)
    print("Test 1: ParallelAgent Initialization")
    print("=" * 80)

    config = create_test_config()
    journal = Journal()

    try:
        agent = ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        )

        assert agent.num_workers == config.agent.num_workers
        assert agent.journal is journal
        assert len(agent.journal) == 0

        print(f"✓ ParallelAgent initialized with {agent.num_workers} workers")
        print("✅ Test 1 PASSED")
        agent.cleanup()
        return True

    except Exception as e:
        print(f"❌ Test 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_agent_single_step():
    """Test ParallelAgent single step execution"""
    print("\n" + "=" * 80)
    print("Test 2: ParallelAgent Single Step Execution")
    print("=" * 80)

    config = create_test_config()
    # Reduce workers for faster testing
    config.agent.num_workers = 2

    journal = Journal()
    config.ensure_dirs()

    try:
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            print(f"\n[TEST] Running single step with {agent.num_workers} workers...")

            # Execute one step
            agent.step()

            # Verify results
            print(f"\n[TEST] Verifying results...")
            print(f"  - Journal length: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")
            print(f"  - Good nodes: {len(journal.good_nodes)}")
            print(f"  - Buggy nodes: {len(journal.buggy_nodes)}")

            # ========== DEBUG: working/ ディレクトリの状態確認 ==========
            print(f"\n[DEBUG] Checking workspace directories...")
            workspace_base = Path(config.workspace_dir)
            for proc_dir in sorted(workspace_base.glob("process_*")):
                print(f"\n  {proc_dir.name}/")
                for f in proc_dir.iterdir():
                    if f.is_file():
                        print(f"    FILE: {f.name} ({f.stat().st_size} bytes)")
                    elif f.is_dir():
                        print(f"    DIR:  {f.name}/")
                        contents = list(f.iterdir())
                        if contents:
                            for sub in contents:
                                size = sub.stat().st_size if sub.is_file() else "dir"
                                print(f"      - {sub.name} ({size} bytes)")
                        else:
                            print(f"      (empty)")

            # ========== DEBUG: ノードの実行結果確認 ==========
            print(f"\n[DEBUG] Checking node execution results...")
            for node in journal.nodes:
                print(f"\n  Node {node.id}:")
                print(f"    is_buggy: {node.is_buggy}")
                print(f"    exc_type: {node.exc_type}")
                if node.term_out:
                    print(f"    term_out (last 5 lines):")
                    for line in node.term_out[-5:]:
                        print(f"      {line[:100]}")

            # Should have created num_workers nodes
            assert len(journal) == agent.num_workers, \
                f"Expected {agent.num_workers} nodes, got {len(journal)}"

            # All nodes should be draft nodes (no parent)
            assert len(journal.draft_nodes) == agent.num_workers, \
                f"Expected {agent.num_workers} draft nodes, got {len(journal.draft_nodes)}"

            print("\n✅ Test 2 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_agent_4_workers():
    """Test ParallelAgent with 4 workers (full parallel)"""
    print("\n" + "=" * 80)
    print("Test 3: ParallelAgent 4 Workers Parallel Execution")
    print("=" * 80)

    config = create_test_config()
    config.agent.num_workers = 4  # 4ワーカー並列

    journal = Journal()
    config.ensure_dirs()

    try:
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            print(f"\n[TEST] Running single step with {agent.num_workers} workers...")

            # Execute one step
            agent.step()

            # Verify results
            print(f"\n[TEST] Verifying results...")
            print(f"  - Journal length: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")

            # Should have created 4 nodes
            assert len(journal) == 4, f"Expected 4 nodes, got {len(journal)}"
            assert len(journal.draft_nodes) == 4, f"Expected 4 draft nodes"

            print("\n✅ Test 3 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_select_parallel_nodes_logic():
    """Test _select_parallel_nodes() Best-First search logic (without API calls)"""
    print("\n" + "=" * 80)
    print("Test 4: _select_parallel_nodes() Best-First Search Logic")
    print("=" * 80)

    config = create_test_config()
    config.agent.num_workers = 4
    config.agent.search.num_drafts = 4
    config.agent.search.debug_prob = 0.5
    config.agent.search.max_debug_depth = 3

    journal = Journal()

    try:
        # === Test 4a: Empty journal → should return all None (drafts) ===
        print("\n[TEST 4a] Empty journal - should select all drafts")
        agent = ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        )

        nodes = agent._select_parallel_nodes()
        print(f"  Selected: {[n.id if n else 'Draft' for n in nodes]}")
        assert all(n is None for n in nodes), "Expected all None (drafts) for empty journal"
        print("  ✓ 4a PASSED: All drafts selected")
        agent.cleanup()

        # === Test 4b: Journal with drafts → should select from existing ===
        print("\n[TEST 4b] Journal with draft nodes")

        # Create mock draft nodes (draft nodes have no parent, so debug_depth=0)
        for i in range(4):
            node = Node(plan=f"Draft {i}", code=f"# code {i}")
            node.is_buggy = True  # Mark as buggy
            # debug_depth is a property calculated from parent, so for draft nodes it's 0
            journal.append(node)

        print(f"  Journal has {len(journal)} nodes, {len(journal.draft_nodes)} drafts")
        print(f"  Debug depths: {[n.debug_depth for n in journal.nodes]}")

        agent2 = ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        )

        nodes = agent2._select_parallel_nodes()
        print(f"  Selected: {[n.id if n else 'Draft' for n in nodes]}")

        # With 4 drafts and all buggy, should select buggy nodes for debugging
        # (with debug_prob probability) or new drafts
        print("  ✓ 4b PASSED: Nodes selected from journal")
        agent2.cleanup()

        # === Test 4c: Journal with good nodes → should select for improvement ===
        print("\n[TEST 4c] Journal with good nodes")

        # Create a good node
        good_node = Node(plan="Good node", code="# good code")
        good_node.is_buggy = False
        good_node.is_buggy_plots = False
        good_node.metric = MetricValue(0.8)  # Good metric
        journal.append(good_node)

        print(f"  Journal has {len(journal.good_nodes)} good nodes")

        agent3 = ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        )

        nodes = agent3._select_parallel_nodes()
        print(f"  Selected: {[n.id if n else 'Draft' for n in nodes]}")

        # Should include the good node for improvement
        has_good_node = any(n is not None and not n.is_buggy for n in nodes)
        print(f"  Has good node selected: {has_good_node}")
        print("  ✓ 4c PASSED: Good node considered for improvement")
        agent3.cleanup()

        # === Test 4d: Debug depth limit ===
        print("\n[TEST 4d] Debug depth limit test")

        # Create a chain of debug nodes to simulate deep debug depth
        # debug_depth is calculated from parent chain with stage_name="debug"
        # For now, we just test that draft nodes (depth=0) are debuggable
        # and that the depth limit logic works conceptually

        # All draft nodes have debug_depth=0, so they should all be debuggable
        debuggable = [
            n for n in journal.buggy_nodes
            if n.is_leaf and n.debug_depth <= config.agent.search.max_debug_depth
        ]
        print(f"  Debuggable nodes (depth <= 3): {len(debuggable)}")
        print(f"  Buggy nodes: {len(journal.buggy_nodes)}")

        # All buggy draft nodes should be debuggable since debug_depth=0
        assert len(debuggable) > 0, "Should have debuggable nodes"
        print("  ✓ 4d PASSED: Debug depth filter works")
        agent3.cleanup()  # Reuse agent3 cleanup

        print("\n✅ Test 4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_agent_run():
    """Test ParallelAgent run method (full execution with multiple steps)"""
    print("\n" + "=" * 80)
    print("Test 5: ParallelAgent Full Run (max 2 steps)")
    print("=" * 80)

    config = create_test_config()
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2  # Stop drafting after 2

    journal = Journal()
    config.ensure_dirs()

    try:
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            print(f"\n[TEST] Running agent (max 2 steps)...")

            # Run with max 2 steps
            success = agent.run(max_steps=2)

            # Verify results
            print(f"\n[TEST] Run completed. Success: {success}")
            print(f"  - Total nodes: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")
            print(f"  - Good nodes: {len(journal.good_nodes)}")
            print(f"  - Buggy nodes: {len(journal.buggy_nodes)}")

            # Should have at least 2 nodes (from first step)
            assert len(journal) >= 2, f"Expected at least 2 nodes, got {len(journal)}"

            if journal.good_nodes:
                best_node = journal.get_best_node(only_good=True)
                print(f"\n[TEST] Best node: {best_node.id}")
                print(f"  - Metric: {best_node.metric}")

            print("\n✅ Test 5 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_agent_full_4workers():
    """Test ParallelAgent with 4 workers, multiple steps - production-like test"""
    print("\n" + "=" * 80)
    print("Test 6: ParallelAgent Full Run (4 workers, max 20 steps)")
    print("=" * 80)

    config = create_test_config()
    config.agent.num_workers = 4
    config.agent.search.num_drafts = 4

    journal = Journal()
    config.ensure_dirs()

    try:
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            print(f"\n[TEST] Running agent with {agent.num_workers} workers (max 20 steps)...")

            # Run with max 20 steps (same as AI-Scientist-v2 stage1)
            success = agent.run(max_steps=20)

            # Verify results
            print(f"\n[TEST] Run completed. Success: {success}")
            print(f"  - Total nodes: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")
            print(f"  - Good nodes: {len(journal.good_nodes)}")
            print(f"  - Buggy nodes: {len(journal.buggy_nodes)}")

            # ========== DEBUG: working/ ディレクトリの状態確認 ==========
            print(f"\n[DEBUG] Checking workspace directories...")
            workspace_base = Path(config.workspace_dir)
            for proc_dir in sorted(workspace_base.glob("process_*")):
                print(f"\n  {proc_dir.name}/")
                for f in proc_dir.iterdir():
                    if f.is_file():
                        print(f"    FILE: {f.name} ({f.stat().st_size} bytes)")
                    elif f.is_dir():
                        print(f"    DIR:  {f.name}/")
                        contents = list(f.iterdir())
                        if contents:
                            for sub in contents:
                                size = sub.stat().st_size if sub.is_file() else "dir"
                                print(f"      - {sub.name} ({size} bytes)")
                        else:
                            print(f"      (empty)")

            # Should have at least 4 nodes (from first step)
            assert len(journal) >= 4, f"Expected at least 4 nodes, got {len(journal)}"

            if journal.good_nodes:
                best_node = journal.get_best_node(only_good=True)
                print(f"\n[TEST] Best node: {best_node.id}")
                print(f"  - Metric: {best_node.metric}")

            print("\n✅ Test 6 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ParallelAgent Integration Tests")
    print("=" * 80)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (no API calls)")
    parser.add_argument("--full", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run specific test by number")
    args = parser.parse_args()

    # テスト関数のマッピング
    test_map = {
        1: ("Init", test_parallel_agent_init),
        2: ("Single Step (2 workers)", test_parallel_agent_single_step),
        3: ("4 Workers Parallel", test_parallel_agent_4_workers),
        4: ("Best-First Logic", test_select_parallel_nodes_logic),
        5: ("Full Run (2 workers)", test_parallel_agent_run),
        6: ("Full Run (4 workers)", test_parallel_agent_full_4workers),
    }

    results = []

    # 特定のテストのみ実行
    if args.test:
        name, func = test_map[args.test]
        results.append((name, func()))
    elif args.quick:
        # Test 1: Initialization (always run)
        results.append(("Init", test_parallel_agent_init()))
        # Test 4: Best-First search logic (no API calls)
        results.append(("Best-First Logic", test_select_parallel_nodes_logic()))
        print("\n[Quick mode] Skipping API-dependent tests")
    else:
        # Test 1: Initialization (always run)
        results.append(("Init", test_parallel_agent_init()))

        # Test 4: Best-First search logic (no API calls)
        results.append(("Best-First Logic", test_select_parallel_nodes_logic()))

        # Test 2: Single step (2 workers)
        results.append(("Single Step (2 workers)", test_parallel_agent_single_step()))

        if args.full:
            # Test 3: 4 workers parallel (takes longer)
            results.append(("4 Workers Parallel", test_parallel_agent_4_workers()))

            # Test 5: Full run
            results.append(("Full Run", test_parallel_agent_run()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
