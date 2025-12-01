"""
Test script for AgentManager
Tests AgentManager initialization, stage transitions, and task_desc JSON format validation.

テスト内容:
- Test 1: AgentManager 初期化（MASist JSON形式）
- Test 2: MASist必須キーのバリデーション
- Test 3: _generate_compat_keys() メソッド
- Test 4: ステージ遷移
- Test 5: _curate_task_desc() メソッド
- Test 6: task_desc JSON形式バリデーション（不正なデータ）
"""
import sys
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import shutil

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

from masist.treesearch.agent_manager import AgentManager, Stage
from masist.treesearch.utils.config import _load_cfg, prep_cfg, Config

# 実験設定を fixtures からインポート
from fixtures.experiments import get_experiment, TPGG_TASK_DESC

# テスト用設定ファイルパス
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"


def create_test_config(workspace_name: str = "test_agent_manager") -> Config:
    """Create a test configuration from YAML file (AI-Scientist-v2準拠)"""
    # Load base config from YAML
    cfg = _load_cfg(TEST_CONFIG_PATH)

    # Set test-specific values
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")

    # Create test data directory if it doesn't exist
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Set goal for prep_cfg validation
    cfg.goal = "Test goal for AgentManager"

    # Process config (creates log_dir and workspace_dir)
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


def test_agent_manager_init():
    """Test 1: AgentManager initialization with MASist JSON format"""
    print("=" * 80)
    print("Test 1: AgentManager Initialization")
    print("=" * 80)

    config = create_test_config()

    try:
        # Get TPGG task_desc and convert to JSON string
        task_desc_str = json.dumps(TPGG_TASK_DESC, ensure_ascii=False, indent=2)

        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )

        # Verify initialization
        assert manager.task_desc is not None, "task_desc should be set"
        assert isinstance(manager.task_desc, dict), "task_desc should be a dict"
        assert manager.current_stage is not None, "current_stage should be set"
        assert manager.current_stage_number == 1, "current_stage_number should be 1"
        assert len(manager.stages) == 1, "Should have one initial stage"

        print(f"✓ AgentManager initialized successfully")
        print(f"✓ Current stage: {manager.current_stage.name}")
        print(f"✓ Stage number: {manager.current_stage_number}")
        print("✅ Test 1 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_masist_required_keys():
    """Test 2: MASist required keys validation"""
    print("\n" + "=" * 80)
    print("Test 2: MASist Required Keys Validation")
    print("=" * 80)

    config = create_test_config()

    try:
        # Test with missing required key
        invalid_task_desc = {
            "Title": "Test",
            "Name": "test",
            # Missing: SimulationRequest, SimulationRequirements, Logging
        }
        task_desc_str = json.dumps(invalid_task_desc, ensure_ascii=False)

        try:
            manager = AgentManager(
                task_desc=task_desc_str,
                cfg=config,
                workspace_dir=Path(config.workspace_dir),
            )
            print("❌ Should have raised ValueError for missing keys")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")

        # Test with all required keys
        valid_task_desc = {
            "Title": "Test Experiment",
            "Name": "test",
            "SimulationRequest": {
                "Background": "Test background",
                "Purpose": "Test purpose",
                "Hypotheses": ["H1", "H2"],
            },
            "SimulationRequirements": {
                "Agents": {},
                "Rules": {},
            },
            "Logging": {
                "RecordContents": [],
            },
        }
        task_desc_str = json.dumps(valid_task_desc, ensure_ascii=False)

        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )
        print("✓ Valid task_desc accepted")

        print("✅ Test 2 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_generate_compat_keys():
    """Test 3: _generate_compat_keys() method"""
    print("\n" + "=" * 80)
    print("Test 3: _generate_compat_keys() Method")
    print("=" * 80)

    config = create_test_config()

    try:
        task_desc = {
            "Title": "Test Experiment Title",
            "Name": "test_exp",
            "SimulationRequest": {
                "Background": "This is the background.",
                "Purpose": "This is the purpose.",
                "Hypotheses": ["Hypothesis 1", "Hypothesis 2"],
                "ResearchQuestions": ["RQ1", "RQ2"],
            },
            "SimulationRequirements": {
                "Agents": {"Count": 4},
                "Rules": {
                    "ExperimentConditions": [
                        {"name": "COND_A", "param": 1},
                        {"name": "COND_B", "param": 2},
                    ],
                },
            },
            "Logging": {
                "RecordContents": ["action", "result"],
            },
            "Other": "Some risk factors and limitations.",
        }
        task_desc_str = json.dumps(task_desc, ensure_ascii=False)

        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )

        # Verify compat keys were generated
        assert "Abstract" in manager.task_desc, "Abstract should be generated"
        assert "Short Hypothesis" in manager.task_desc, "Short Hypothesis should be generated"
        assert "Experiments" in manager.task_desc, "Experiments should be generated"
        assert "Risk Factors and Limitations" in manager.task_desc, "Risk Factors should be generated"

        # Verify content
        assert "This is the background." in manager.task_desc["Abstract"]
        assert "This is the purpose." in manager.task_desc["Abstract"]
        print(f"✓ Abstract: {manager.task_desc['Abstract'][:50]}...")

        assert "Hypothesis 1" in manager.task_desc["Short Hypothesis"]
        assert "Hypothesis 2" in manager.task_desc["Short Hypothesis"]
        print(f"✓ Short Hypothesis: {manager.task_desc['Short Hypothesis'][:50]}...")

        assert len(manager.task_desc["Experiments"]) == 2
        print(f"✓ Experiments: {manager.task_desc['Experiments']}")

        assert manager.task_desc["Risk Factors and Limitations"] == "Some risk factors and limitations."
        print(f"✓ Risk Factors: {manager.task_desc['Risk Factors and Limitations']}")

        print("✅ Test 3 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_stage_transitions():
    """Test 4: Stage transitions"""
    print("\n" + "=" * 80)
    print("Test 4: Stage Transitions")
    print("=" * 80)

    config = create_test_config()

    try:
        task_desc_str = json.dumps(TPGG_TASK_DESC, ensure_ascii=False, indent=2)

        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )

        # Verify initial stage
        assert manager.current_stage_number == 1
        assert "initial_implementation" in manager.current_stage.name
        print(f"✓ Initial stage: {manager.current_stage.name}")

        # Verify main_stage_dict
        assert 1 in manager.main_stage_dict
        assert manager.main_stage_dict[1] == "initial_implementation"
        assert manager.main_stage_dict[2] == "baseline_tuning"
        assert manager.main_stage_dict[3] == "creative_research"
        assert manager.main_stage_dict[4] == "ablation_studies"
        print("✓ main_stage_dict configured correctly")

        # Verify main_stage_goals
        assert 1 in manager.main_stage_goals
        assert "エンドツーエンド" in manager.main_stage_goals[1]
        assert "バリエーション" in manager.main_stage_goals[2]
        assert "創造的" in manager.main_stage_goals[3]
        assert "貢献度" in manager.main_stage_goals[4]
        print("✓ main_stage_goals configured correctly (MASist format)")

        print("✅ Test 4 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_curate_task_desc():
    """Test 5: _curate_task_desc() method"""
    print("\n" + "=" * 80)
    print("Test 5: _curate_task_desc() Method")
    print("=" * 80)

    config = create_test_config()

    try:
        task_desc_str = json.dumps(TPGG_TASK_DESC, ensure_ascii=False, indent=2)

        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )

        # Get curated task_desc
        curated = manager._curate_task_desc(manager.current_stage)

        # Verify it's a string
        assert isinstance(curated, str), "curated task_desc should be a string"
        print(f"✓ Curated task_desc is a string ({len(curated)} chars)")

        # Verify it contains key information
        assert "Title" in curated, "Should contain Title section"
        assert "Abstract" in curated, "Should contain Abstract section"
        assert "Short Hypothesis" in curated, "Should contain Short Hypothesis section"
        print("✓ Contains required sections: Title, Abstract, Short Hypothesis")

        # Verify it contains actual content from TPGG
        assert "TPGG" in curated or "しきい値公共財ゲーム" in curated or "Threshold" in curated
        print("✓ Contains TPGG-related content")

        print("✅ Test 5 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_invalid_json_format():
    """Test 6: Invalid JSON format validation"""
    print("\n" + "=" * 80)
    print("Test 6: Invalid JSON Format Validation")
    print("=" * 80)

    config = create_test_config()

    try:
        # Test with invalid JSON string
        try:
            manager = AgentManager(
                task_desc="not a valid json",
                cfg=config,
                workspace_dir=Path(config.workspace_dir),
            )
            print("❌ Should have raised JSONDecodeError")
            return False
        except json.JSONDecodeError as e:
            print(f"✓ Correctly raised JSONDecodeError for invalid JSON")

        # Test with valid JSON but wrong type
        try:
            manager = AgentManager(
                task_desc=json.dumps(["list", "not", "dict"]),
                cfg=config,
                workspace_dir=Path(config.workspace_dir),
            )
            print("❌ Should have raised error for list instead of dict")
            return False
        except (TypeError, AttributeError, ValueError) as e:
            print(f"✓ Correctly raised error for non-dict JSON")

        print("✅ Test 6 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def test_experiments_fixture_integration():
    """Test 7: Integration with experiments.py fixtures"""
    print("\n" + "=" * 80)
    print("Test 7: Integration with experiments.py Fixtures")
    print("=" * 80)

    config = create_test_config()

    try:
        # Test all available experiments
        from fixtures.experiments import list_experiments

        for exp_name in list_experiments():
            exp = get_experiment(exp_name)
            task_desc_str = json.dumps(exp["task_desc"], ensure_ascii=False, indent=2)

            manager = AgentManager(
                task_desc=task_desc_str,
                cfg=config,
                workspace_dir=Path(config.workspace_dir),
            )

            # Verify compat keys were generated
            assert "Abstract" in manager.task_desc
            assert "Short Hypothesis" in manager.task_desc
            print(f"✓ {exp_name}: AgentManager initialized successfully")

        print("✅ Test 7 PASSED")
        cleanup_test_dirs(config)
        return True

    except Exception as e:
        print(f"❌ Test 7 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_test_dirs(config)
        return False


def main():
    """Run all tests"""
    import argparse
    parser = argparse.ArgumentParser(description="AgentManager Tests")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Run specific test by number")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AgentManager Tests")
    print("=" * 80)

    test_map = {
        1: ("Initialization", test_agent_manager_init),
        2: ("Required Keys Validation", test_masist_required_keys),
        3: ("_generate_compat_keys()", test_generate_compat_keys),
        4: ("Stage Transitions", test_stage_transitions),
        5: ("_curate_task_desc()", test_curate_task_desc),
        6: ("Invalid JSON Format", test_invalid_json_format),
        7: ("Experiments Fixture Integration", test_experiments_fixture_integration),
    }

    results = []

    if args.test:
        name, func = test_map[args.test]
        results.append((name, func()))
    else:
        for num, (name, func) in test_map.items():
            results.append((name, func()))

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
