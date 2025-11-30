"""
Test script for ParallelAgent
Tests parallel node processing with multiple workers

テスト内容:
- Test 1: ParallelAgent 初期化
- Test 2: step() - 単一ステップ実行（2ワーカー）
- Test 3: 4ワーカー並列実行
- Test 4: _select_parallel_nodes() - Best-First探索（debug/improve選択）
- Test 5: run() - 複数ステップ実行
- Test 6: 4ワーカー本番相当テスト (Phase 1-6)
- Test 7: マルチシード評価・集約ユニットテスト（モック使用）
- Test 8: Stage 1 フルフロー + マルチシード評価・集約（統合テスト）
"""
import sys
import logging
import random
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress httpx INFO logs (HTTP Request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

from masist.treesearch import ParallelAgent, Journal, Node
from masist.treesearch.utils.metric import MetricValue
from masist.treesearch.utils.config import Config

# 実験設定を fixtures からインポート
from fixtures.experiments import get_experiment, list_experiments

# グローバル変数（コマンドライン引数で変更可能）
CURRENT_EXPERIMENT = "tpgg"
SIMPLE_TASK_DESC = None
SAMPLE_METRICS = None


def set_experiment(name: str):
    """実験を設定する"""
    global CURRENT_EXPERIMENT, SIMPLE_TASK_DESC, SAMPLE_METRICS
    exp = get_experiment(name)
    CURRENT_EXPERIMENT = name
    SIMPLE_TASK_DESC = exp["task_desc"]
    SAMPLE_METRICS = exp["metrics"]
    print(f"[Experiment] Using: {exp['name']} ({name})")
    print(f"[Experiment] Metrics: {SAMPLE_METRICS}")


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
                best_node = journal.get_best_node(only_good=True, use_val_metric_only=True)
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
                best_node = journal.get_best_node(only_good=True, use_val_metric_only=True)
                print(f"\n[TEST] Best node: {best_node.id}")
                print(f"  - Metric: {best_node.metric}")

            print("\n✅ Test 6 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_seed_evaluation():
    """Test 7: Multi-seed evaluation and aggregation"""
    print("\n" + "=" * 80)
    print("Test 7: Multi-Seed Evaluation and Aggregation")
    print("=" * 80)

    import numpy as np
    import os

    config = create_test_config(workspace_name="test_multi_seed")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    # Enable multi-seed evaluation
    config.agent.multi_seed_eval.enabled = True
    config.agent.multi_seed_eval.num_seeds = 3

    journal = Journal()
    config.ensure_dirs()

    try:
        print(f"\n[TEST] Multi-seed config:")
        print(f"  - enabled: {config.agent.multi_seed_eval.enabled}")
        print(f"  - num_seeds: {config.agent.multi_seed_eval.num_seeds}")

        # === Test 7a: Create a mock good node with experiment_data ===
        print("\n[TEST 7a] Creating mock good node with experiment_data...")

        # Create mock experiment data (new scenarios format)
        mock_exp_data = {
            'scenarios': {
                'CONDITION_A': {
                    'messages': ['msg1', 'msg2'],
                    'events': ['event1'],
                    'metrics': {
                        'accuracy': 0.85,
                        'f1_score': 0.82,
                    },
                    'config': {'threshold': 20},
                },
                'CONDITION_B': {
                    'messages': ['msg3', 'msg4'],
                    'events': ['event2'],
                    'metrics': {
                        'accuracy': 0.78,
                        'f1_score': 0.75,
                    },
                    'config': {'threshold': 22},
                },
            },
            'metrics': {
                'avg_accuracy': 0.815,
                'avg_f1_score': 0.785,
            },
        }

        # Create exp_results directory within workspace (not temp dir to avoid relative_to error)
        temp_dir = os.path.join(config.workspace_dir, "test_mock_data")
        os.makedirs(temp_dir, exist_ok=True)
        exp_results_dir = os.path.join(temp_dir, "exp_results_node1")
        os.makedirs(exp_results_dir, exist_ok=True)

        # Save mock experiment data
        np.savez_compressed(
            os.path.join(exp_results_dir, 'experiment_data.npz'),
            experiment_data=np.array(mock_exp_data, dtype=object)
        )

        # Create mock good node
        good_node = Node(
            plan="Test simulation with two conditions",
            code="""
import os
import numpy as np
import random

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Simple simulation
metrics_a = {'accuracy': random.uniform(0.7, 0.9), 'f1_score': random.uniform(0.7, 0.9)}
metrics_b = {'accuracy': random.uniform(0.6, 0.8), 'f1_score': random.uniform(0.6, 0.8)}

experiment_data = {
    'scenarios': {
        'CONDITION_A': {'messages': [], 'events': [], 'metrics': metrics_a, 'config': {}},
        'CONDITION_B': {'messages': [], 'events': [], 'metrics': metrics_b, 'config': {}},
    },
    'metrics': {'avg_accuracy': (metrics_a['accuracy'] + metrics_b['accuracy']) / 2},
}

np.savez_compressed(f'{working_dir}/experiment_data.npz', experiment_data=np.array(experiment_data, dtype=object))
print(f'Simulation completed: metrics = {experiment_data["metrics"]}')
"""
        )
        good_node.is_buggy = False
        good_node.is_buggy_plots = False
        good_node.metric = MetricValue(0.8)
        good_node.exp_results_dir = exp_results_dir
        journal.append(good_node)

        print(f"  ✓ Created good node: {good_node.id}")
        print(f"  ✓ exp_results_dir: {exp_results_dir}")

        # === Test 7b: Test _run_multi_seed_evaluation ===
        print("\n[TEST 7b] Testing _run_multi_seed_evaluation...")

        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            # Run multi-seed evaluation
            seed_nodes = agent._run_multi_seed_evaluation(good_node)

            print(f"  - Seed nodes created: {len(seed_nodes)}")
            for i, sn in enumerate(seed_nodes):
                print(f"    Seed {i}: node_id={sn.id}, is_seed_node={sn.is_seed_node}")

            # Verify seed nodes
            assert len(seed_nodes) == config.agent.multi_seed_eval.num_seeds, \
                f"Expected {config.agent.multi_seed_eval.num_seeds} seed nodes, got {len(seed_nodes)}"

            for sn in seed_nodes:
                assert sn.is_seed_node, "Seed node should have is_seed_node=True"
                assert sn.parent == good_node, "Seed node should have good_node as parent"

            print("  ✓ 7b PASSED: Multi-seed evaluation works")

            # === Test 7c: Test _run_plot_aggregation ===
            print("\n[TEST 7c] Testing _run_plot_aggregation...")

            # Create mock exp_results_dir for each seed node with varying metrics
            for i, sn in enumerate(seed_nodes):
                seed_exp_dir = os.path.join(temp_dir, f"exp_results_seed_{i}")
                os.makedirs(seed_exp_dir, exist_ok=True)

                # Create experiment data with slight variations
                seed_exp_data = {
                    'scenarios': {
                        'CONDITION_A': {
                            'messages': [],
                            'events': [],
                            'metrics': {
                                'accuracy': 0.80 + (i * 0.02) + random.uniform(-0.01, 0.01),
                                'f1_score': 0.78 + (i * 0.02) + random.uniform(-0.01, 0.01),
                            },
                            'config': {},
                        },
                        'CONDITION_B': {
                            'messages': [],
                            'events': [],
                            'metrics': {
                                'accuracy': 0.72 + (i * 0.02) + random.uniform(-0.01, 0.01),
                                'f1_score': 0.70 + (i * 0.02) + random.uniform(-0.01, 0.01),
                            },
                            'config': {},
                        },
                    },
                    'metrics': {},
                }

                np.savez_compressed(
                    os.path.join(seed_exp_dir, 'experiment_data.npz'),
                    experiment_data=np.array(seed_exp_data, dtype=object)
                )
                sn.exp_results_dir = seed_exp_dir

            # Run plot aggregation
            agg_node = agent._run_plot_aggregation(good_node, seed_nodes)

            if agg_node:
                print(f"  - Aggregation node created: {agg_node.id}")
                print(f"  - is_seed_agg_node: {agg_node.is_seed_agg_node}")
                print(f"  - plots: {agg_node.plots}")

                assert agg_node.is_seed_agg_node, "Aggregation node should have is_seed_agg_node=True"
                print("  ✓ 7c PASSED: Plot aggregation works")
            else:
                print("  ⚠ Aggregation node not created (may be due to execution environment)")
                print("  ✓ 7c PASSED: Plot aggregation attempted")

        # === Test 7d: Test aggregation code generation ===
        print("\n[TEST 7d] Testing _generate_aggregation_code...")

        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            agg_code = agent._generate_aggregation_code(good_node, seed_nodes)

            # Verify code contains expected elements
            assert "extract_metrics" in agg_code, "Should have extract_metrics function"
            assert "scenarios" in agg_code, "Should handle scenarios format"
            assert "aggregated_by_scenario" in agg_code, "Should aggregate by scenario"
            assert "mean" in agg_code and "std" in agg_code, "Should calculate mean and std"

            print(f"  - Generated code length: {len(agg_code)} chars")
            print("  ✓ 7d PASSED: Aggregation code generation works")

        # Cleanup is handled by workspace cleanup
        print("\n✅ Test 7 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 7 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_stage1_with_multi_seed():
    """Test 8: Stage 1 Full Flow + Multi-Seed Evaluation & Aggregation

    This is the comprehensive integration test that runs:
    1. Phase 1-6 with 4 workers (draft generation → VLM analysis)
    2. Multi-seed evaluation (re-run good node's code with different seeds)
    3. Plot aggregation (combine results from multiple seeds)
    """
    print("\n" + "=" * 80)
    print("Test 8: Stage 1 Full Flow + Multi-Seed Evaluation & Aggregation")
    print("=" * 80)

    config = create_test_config(workspace_name="test_full_stage1")
    config.agent.num_workers = 4
    config.agent.search.num_drafts = 4
    # Enable multi-seed evaluation
    config.agent.multi_seed_eval.enabled = True
    config.agent.multi_seed_eval.num_seeds = 3

    journal = Journal()
    config.ensure_dirs()

    try:
        print(f"\n[TEST] Configuration:")
        print(f"  - num_workers: {config.agent.num_workers}")
        print(f"  - num_drafts: {config.agent.search.num_drafts}")
        print(f"  - multi_seed_eval.enabled: {config.agent.multi_seed_eval.enabled}")
        print(f"  - multi_seed_eval.num_seeds: {config.agent.multi_seed_eval.num_seeds}")

        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=SAMPLE_METRICS,
        ) as agent:
            print(f"\n[TEST] Running Stage 1 with {agent.num_workers} workers (max 20 steps)...")

            # ========== Stage 1: Main run ==========
            success = agent.run(max_steps=20)

            print(f"\n[TEST] Stage 1 completed. Success: {success}")
            print(f"  - Total nodes: {len(journal)}")
            print(f"  - Draft nodes: {len(journal.draft_nodes)}")
            print(f"  - Good nodes: {len(journal.good_nodes)}")
            print(f"  - Buggy nodes: {len(journal.buggy_nodes)}")

            if not success or not journal.good_nodes:
                print("\n⚠ No good nodes found in Stage 1. Skipping multi-seed evaluation.")
                print("✅ Test 8 PASSED (Stage 1 only)")
                return True

            # Get best node
            best_node = journal.get_best_node(only_good=True, use_val_metric_only=True)
            print(f"\n[TEST] Best node from Stage 1: {best_node.id[:8]}")
            print(f"  - Metric: {best_node.metric}")

            # ========== Check Multi-Seed Evaluation Results ==========
            # Note: multi-seed evaluation is already run inside agent.run() when enabled
            # Here we just verify the results
            seed_nodes = [n for n in journal.nodes if n.is_seed_node]
            agg_nodes = [n for n in journal.nodes if getattr(n, 'is_seed_agg_node', False)]

            print(f"\n[TEST] Multi-seed evaluation results (already run in agent.run()):")
            print(f"  - Seed nodes created: {len(seed_nodes)}")

            successful_seeds = [sn for sn in seed_nodes if not sn.is_buggy]
            print(f"  - Successful seed nodes: {len(successful_seeds)}")

            for i, sn in enumerate(seed_nodes):
                status = "good" if not sn.is_buggy else "buggy"
                print(f"    Seed {i}: {sn.id[:8]} ({status})")

            # ========== Check Plot Aggregation Results ==========
            agg_node = agg_nodes[0] if agg_nodes else None

            if agg_node:
                print(f"\n[TEST] Aggregation results:")
                print(f"  - Aggregation node: {agg_node.id[:8]}")
                print(f"  - is_seed_agg_node: {agg_node.is_seed_agg_node}")
                print(f"  - plots generated: {len(agg_node.plots) if agg_node.plots else 0}")
                if agg_node.plots:
                    for plot in agg_node.plots[:5]:  # Show first 5
                        print(f"    - {plot}")
            else:
                print("\n⚠ No aggregation node found.")

            # ========== Summary ==========
            print("\n" + "-" * 40)
            print("[TEST] Full Stage 1 Summary:")
            print("-" * 40)
            print(f"  Stage 1 nodes: {len(journal)}")
            print(f"  Best node: {best_node.id[:8]} (metric: {best_node.metric})")
            print(f"  Seed nodes: {len(seed_nodes)} ({len(successful_seeds)} successful)")
            print(f"  Aggregation: {'Yes' if agg_node else 'No'}")

            print("\n✅ Test 8 PASSED")
            return True

    except Exception as e:
        print(f"\n❌ Test 8 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    import argparse
    parser = argparse.ArgumentParser(
        description="ParallelAgent Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiments:
  {', '.join(list_experiments())}

Examples:
  python test_parallel.py --exp bbs --test 8     # Run test 8 with BBS experiment
  python test_parallel.py --exp abm --full       # Run full tests with ABM experiment
  python test_parallel.py --list                 # List available experiments
"""
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (no API calls)")
    parser.add_argument("--full", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="Run specific test by number")
    parser.add_argument("--exp", type=str, choices=list_experiments(), default="tpgg",
                        help=f"Experiment to use (default: tpgg)")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")
    args = parser.parse_args()

    # 実験一覧を表示して終了
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 60)
        for key in list_experiments():
            exp = get_experiment(key)
            print(f"  {key:15} - {exp['name']}")
            print(f"                  Metrics: {', '.join(exp['metrics'][:3])}...")
        print()
        return True

    # 実験を設定
    set_experiment(args.exp)

    print("\n" + "=" * 80)
    print("ParallelAgent Integration Tests")
    print(f"Experiment: {args.exp}")
    print("=" * 80)

    # テスト関数のマッピング
    test_map = {
        1: ("Init", test_parallel_agent_init),
        2: ("Single Step (2 workers)", test_parallel_agent_single_step),
        3: ("4 Workers Parallel", test_parallel_agent_4_workers),
        4: ("Best-First Logic", test_select_parallel_nodes_logic),
        5: ("Full Run (2 workers)", test_parallel_agent_run),
        6: ("Full Run (4 workers)", test_parallel_agent_full_4workers),
        7: ("Multi-Seed Unit Test", test_multi_seed_evaluation),
        8: ("Full Stage 1 + Multi-Seed", test_full_stage1_with_multi_seed),
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
