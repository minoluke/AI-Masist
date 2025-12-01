"""
Test script for get_best_node() with hypothesis validation

テスト内容:
- Test 1: use_val_metric_only=True の場合（メトリクス最大を選択）
- Test 2: 単一ノードの場合（そのまま返す）
- Test 3: LLM選択のモック（hypothesis_validation, scientific_soundness）
- Test 4: LLMエラー時のフォールバック
- Test 5: 無効なnode ID時のフォールバック
- Test 6: プロンプト構造の検証
- Test 7: seed_node の除外確認
- Test 8: 実LLM呼び出し（統合テスト）
- Test 9: return_selection_info=True で選択情報を取得・検証
"""
import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from masist.treesearch import Journal, Node
from masist.treesearch.utils.metric import MetricValue
from masist.treesearch.utils.config import Config


def create_test_node(
    node_id: str,
    metric_value: float,
    plan: str = "Test plan",
    code: str = "print('test')",
    is_buggy: bool = False,
    is_buggy_plots: bool = False,
    is_seed_node: bool = False,
    vlm_feedback_summary: str = "",
    analysis: str = "",
) -> Node:
    """Create a test node with specified attributes"""
    node = Node(
        id=node_id,
        plan=plan,
        code=code,
        metric=MetricValue(value=metric_value, maximize=True),
        is_buggy=is_buggy,
        is_buggy_plots=is_buggy_plots,
        is_seed_node=is_seed_node,
    )
    node.vlm_feedback_summary = vlm_feedback_summary
    node.analysis = analysis
    return node


def test_use_val_metric_only():
    """Test 1: use_val_metric_only=True uses max metric"""
    print("=" * 80)
    print("Test 1: use_val_metric_only=True")
    print("=" * 80)

    journal = Journal()

    # Add nodes with different metrics
    node1 = create_test_node("node1", 0.5)
    node2 = create_test_node("node2", 0.9)  # Best metric
    node3 = create_test_node("node3", 0.7)

    journal.append(node1)
    journal.append(node2)
    journal.append(node3)

    # Should return node with highest metric
    best = journal.get_best_node(use_val_metric_only=True)

    assert best is not None, "Should return a node"
    assert best.id == "node2", f"Expected node2 (highest metric), got {best.id}"

    print(f"✓ Selected node: {best.id} with metric {best.metric.value}")
    print("✅ Test 1 PASSED")
    return True


def test_single_node():
    """Test 2: Single node returns immediately"""
    print("\n" + "=" * 80)
    print("Test 2: Single node case")
    print("=" * 80)

    journal = Journal()
    node1 = create_test_node("only_node", 0.5)
    journal.append(node1)

    # Should return the only node without LLM call
    with patch("masist.treesearch.journal.query") as mock_query:
        best = journal.get_best_node(use_val_metric_only=False)

        # LLM should NOT be called for single node
        mock_query.assert_not_called()

        assert best is not None, "Should return a node"
        assert best.id == "only_node", f"Expected only_node, got {best.id}"

    print(f"✓ Single node returned without LLM call: {best.id}")
    print("✅ Test 2 PASSED")
    return True


def test_llm_selection_with_hypothesis():
    """Test 3: LLM selection with hypothesis_validation and scientific_soundness"""
    print("\n" + "=" * 80)
    print("Test 3: LLM selection with hypothesis validation")
    print("=" * 80)

    journal = Journal()

    # Node with lower metric but better hypothesis validation
    node1 = create_test_node(
        "node_good_hypothesis",
        metric_value=0.6,
        plan="Test FAIR vs NORULE conditions",
        code="# Implements proper A/B comparison\nfor condition in ['FAIR', 'NORULE']: ...",
        vlm_feedback_summary="Clear difference between conditions",
        analysis="",
    )

    # Node with higher metric but poor hypothesis validation
    node2 = create_test_node(
        "node_high_metric",
        metric_value=0.9,
        plan="Maximize cooperation",
        code="# Just maximizes metric\ncooperation = 1.0",
        vlm_feedback_summary="High values but no comparison",
        analysis="",
    )

    journal.append(node1)
    journal.append(node2)

    # Mock LLM response to select node1 based on hypothesis validation
    mock_response = {
        "selected_id": "node_good_hypothesis",
        "reasoning": "Node1 properly tests the hypothesis with clear condition comparisons",
        "hypothesis_validation": "Provides strong evidence by comparing FAIR vs NORULE conditions",
        "scientific_soundness": 4,
    }

    task_desc = "Test hypothesis: Rules increase cooperation in public goods games"

    with patch("masist.treesearch.journal.query", return_value=mock_response) as mock_query:
        best = journal.get_best_node(
            use_val_metric_only=False,
            task_desc=task_desc,
        )

        # Verify LLM was called
        mock_query.assert_called_once()

        # Check that task_desc was included in prompt
        call_args = mock_query.call_args
        prompt = call_args.kwargs.get("system_message") or call_args[1].get("system_message")
        assert task_desc in str(prompt), "task_desc should be in prompt"

        assert best is not None, "Should return a node"
        assert best.id == "node_good_hypothesis", f"Expected node_good_hypothesis, got {best.id}"

    print(f"✓ LLM selected node based on hypothesis validation: {best.id}")
    print(f"✓ Hypothesis validation: {mock_response['hypothesis_validation']}")
    print(f"✓ Scientific soundness: {mock_response['scientific_soundness']}/5")
    print("✅ Test 3 PASSED")
    return True


def test_llm_fallback_on_error():
    """Test 4: Falls back to metric-based selection on LLM error"""
    print("\n" + "=" * 80)
    print("Test 4: LLM error fallback")
    print("=" * 80)

    journal = Journal()

    node1 = create_test_node("node1", 0.5)
    node2 = create_test_node("node2", 0.9)  # Best metric

    journal.append(node1)
    journal.append(node2)

    # Mock LLM to raise an error
    with patch("masist.treesearch.journal.query", side_effect=Exception("API Error")):
        best = journal.get_best_node(use_val_metric_only=False)

        # Should fall back to max metric
        assert best is not None, "Should return a node"
        assert best.id == "node2", f"Expected node2 (fallback to max metric), got {best.id}"

    print(f"✓ Fell back to metric-based selection: {best.id}")
    print("✅ Test 4 PASSED")
    return True


def test_llm_fallback_on_invalid_id():
    """Test 5: Falls back when LLM returns invalid node ID"""
    print("\n" + "=" * 80)
    print("Test 5: Invalid node ID fallback")
    print("=" * 80)

    journal = Journal()

    node1 = create_test_node("node1", 0.5)
    node2 = create_test_node("node2", 0.9)

    journal.append(node1)
    journal.append(node2)

    # Mock LLM response with invalid ID
    mock_response = {
        "selected_id": "nonexistent_node",
        "reasoning": "This node doesn't exist",
        "hypothesis_validation": "N/A",
        "scientific_soundness": 3,
    }

    with patch("masist.treesearch.journal.query", return_value=mock_response):
        best = journal.get_best_node(use_val_metric_only=False)

        # Should fall back to max metric
        assert best is not None, "Should return a node"
        assert best.id == "node2", f"Expected node2 (fallback), got {best.id}"

    print(f"✓ Fell back on invalid ID: {best.id}")
    print("✅ Test 5 PASSED")
    return True


def test_prompt_structure():
    """Test 6: Verify prompt contains all required fields"""
    print("\n" + "=" * 80)
    print("Test 6: Prompt structure verification")
    print("=" * 80)

    journal = Journal()

    node1 = create_test_node(
        "node1",
        metric_value=0.7,
        plan="Test plan for node1",
        code="def experiment(): pass",
        vlm_feedback_summary="Good visualization",
        analysis="No bugs found",
    )
    node2 = create_test_node(
        "node2",
        metric_value=0.8,
        plan="Test plan for node2",
        code="def run(): return True",
    )

    journal.append(node1)
    journal.append(node2)

    task_desc = "Hypothesis: Explicit rules increase cooperation"
    captured_prompt = None

    def capture_prompt(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get("system_message")
        return {
            "selected_id": "node1",
            "reasoning": "Test",
            "hypothesis_validation": "Test",
            "scientific_soundness": 3,
        }

    with patch("masist.treesearch.journal.query", side_effect=capture_prompt):
        journal.get_best_node(use_val_metric_only=False, task_desc=task_desc)

    # Verify prompt structure
    assert captured_prompt is not None, "Prompt should be captured"

    # Check required sections
    assert "Introduction" in captured_prompt, "Should have Introduction"
    assert "Experiment Goal" in captured_prompt, "Should have Experiment Goal"
    assert "Evaluation Criteria" in captured_prompt, "Should have Evaluation Criteria"
    assert "Task" in captured_prompt, "Should have Task"
    assert "Candidates" in captured_prompt, "Should have Candidates"

    # Check task_desc is included
    assert task_desc in str(captured_prompt["Experiment Goal"]), "task_desc should be in Experiment Goal"

    # Check candidate info
    candidates = captured_prompt["Candidates"]
    assert "node1" in candidates, "node1 should be in candidates"
    assert "node2" in candidates, "node2 should be in candidates"
    assert "Design Plan:" in candidates, "Should include Design Plan"
    assert "Code (first 1500 chars):" in candidates, "Should include Code"
    assert "Metric:" in candidates, "Should include Metric"

    print("✓ Prompt contains all required sections")
    print(f"✓ Experiment Goal: {captured_prompt['Experiment Goal']}")
    print("✅ Test 6 PASSED")
    return True


def test_seed_nodes_excluded():
    """Test 7: Seed nodes are excluded from candidates"""
    print("\n" + "=" * 80)
    print("Test 7: Seed nodes excluded")
    print("=" * 80)

    journal = Journal()

    node1 = create_test_node("node1", 0.7, is_seed_node=False)
    node2 = create_test_node("seed_node", 0.9, is_seed_node=True)  # Should be excluded
    node3 = create_test_node("node3", 0.8, is_seed_node=False)

    journal.append(node1)
    journal.append(node2)
    journal.append(node3)

    captured_prompt = None

    def capture_prompt(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get("system_message")
        return {
            "selected_id": "node3",
            "reasoning": "Test",
            "hypothesis_validation": "Test",
            "scientific_soundness": 3,
        }

    with patch("masist.treesearch.journal.query", side_effect=capture_prompt):
        journal.get_best_node(use_val_metric_only=False)

    candidates = captured_prompt["Candidates"]
    assert "node1" in candidates, "node1 should be in candidates"
    assert "node3" in candidates, "node3 should be in candidates"
    assert "seed_node" not in candidates, "seed_node should NOT be in candidates"

    print("✓ Seed nodes excluded from candidates")
    print("✅ Test 7 PASSED")
    return True


def test_with_real_llm():
    """Test 8: Real LLM call (optional, requires API key)"""
    print("\n" + "=" * 80)
    print("Test 8: Real LLM call (integration test)")
    print("=" * 80)

    import os
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️ DEEPSEEK_API_KEY not set, skipping real LLM test")
        print("✅ Test 8 SKIPPED")
        return True

    journal = Journal()

    # Create nodes with distinct characteristics
    node1 = create_test_node(
        "node_proper_experiment",
        metric_value=0.65,
        plan="Compare FAIR vs NORULE conditions to test if rules increase cooperation",
        code="""
# Proper experimental design
conditions = ['FAIR', 'NORULE']
results = {}
for condition in conditions:
    results[condition] = run_simulation(condition)
compare_conditions(results)
""",
        vlm_feedback_summary="Shows clear difference between FAIR (0.8) and NORULE (0.4)",
    )

    node2 = create_test_node(
        "node_high_metric_only",
        metric_value=0.95,
        plan="Maximize cooperation metric",
        code="""
# Just returns high values
cooperation = 0.95
print(f"Cooperation: {cooperation}")
""",
        vlm_feedback_summary="High cooperation but only single condition tested",
    )

    journal.append(node1)
    journal.append(node2)

    task_desc = """
    Hypothesis: Explicit rules increase cooperation in public goods games.
    The experiment should compare conditions WITH rules vs WITHOUT rules.
    """

    try:
        best = journal.get_best_node(
            use_val_metric_only=False,
            task_desc=task_desc,
        )

        print(f"✓ LLM selected: {best.id}")
        print(f"✓ Selected metric: {best.metric.value}")

        # We expect LLM to prefer node1 due to proper experimental design
        if best.id == "node_proper_experiment":
            print("✓ LLM correctly preferred proper experimental design over high metric")
        else:
            print("⚠️ LLM selected high metric node (may vary based on LLM judgment)")

        print("✅ Test 8 PASSED")
        return True

    except Exception as e:
        print(f"❌ Test 8 FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Running Best Node Selection Tests")
    print("=" * 80 + "\n")

    results = []

    tests = [
        ("Test 1: use_val_metric_only", test_use_val_metric_only),
        ("Test 2: Single node", test_single_node),
        ("Test 3: LLM selection with hypothesis", test_llm_selection_with_hypothesis),
        ("Test 4: LLM error fallback", test_llm_fallback_on_error),
        ("Test 5: Invalid node ID fallback", test_llm_fallback_on_invalid_id),
        ("Test 6: Prompt structure", test_prompt_structure),
        ("Test 7: Seed nodes excluded", test_seed_nodes_excluded),
        ("Test 8: Real LLM call", test_with_real_llm),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED with exception: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return all(r for _, r in results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        test_map = {
            "1": test_use_val_metric_only,
            "2": test_single_node,
            "3": test_llm_selection_with_hypothesis,
            "4": test_llm_fallback_on_error,
            "5": test_llm_fallback_on_invalid_id,
            "6": test_prompt_structure,
            "7": test_seed_nodes_excluded,
            "8": test_with_real_llm,
            "9": test_return_selection_info,
            "10": test_return_selection_info_with_real_llm,
        }
        if test_name in test_map:
            test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {list(test_map.keys())}")
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
