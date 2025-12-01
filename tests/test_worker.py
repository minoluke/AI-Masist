"""
Test script for worker node processor
Tests single node processing through all phases
"""
import sys
import json
import logging
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

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

from masist.treesearch.node_processor import process_node_wrapper
from masist.treesearch import Node, Journal
from masist.treesearch.utils.config import _load_cfg, prep_cfg, Config
import shutil

# 実験設定を fixtures からインポート
from fixtures.experiments import get_experiment

# TPGG 実験を使用
_exp = get_experiment("tpgg")
SAMPLE_TASK_DESC = json.dumps(_exp["task_desc"], ensure_ascii=False, indent=2)
SAMPLE_METRICS = _exp["metrics"]

# テスト用設定ファイルパス
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"


def create_test_config(workspace_name: str = "test_worker") -> Config:
    """Create a test configuration from YAML file (AI-Scientist-v2準拠)"""
    cfg = _load_cfg(TEST_CONFIG_PATH)
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.goal = "Test goal for worker"
    cfg.exec.timeout = 300  # 5 minutes for worker tests
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


def test_draft_node_processing():
    """Test processing a draft node through all phases"""
    print("=" * 80)
    print("Testing Draft Node Processing - TPGG Simulation")
    print("=" * 80)

    # Setup
    config = create_test_config()
    task_desc = SAMPLE_TASK_DESC

    # prep_cfg already creates workspace dirs

    try:
        # Process a draft node (parent_node_data = None)
        print("\n[TEST] Processing TPGG draft node...")
        evaluation_metrics = SAMPLE_METRICS
        result_data = process_node_wrapper(
            node_data=None,
            task_desc=task_desc,
            cfg=config,
            evaluation_metrics=evaluation_metrics,
            memory_summary=None
        )

        # Verify result
        print("\n[TEST] Verifying result...")
        assert isinstance(result_data, dict), "Result should be a dict"
        assert "code" in result_data, "Result should contain 'code'"
        assert "plan" in result_data, "Result should contain 'plan'"
        assert "id" in result_data, "Result should contain 'id'"
        assert "metric" in result_data, "Result should contain 'metric'"
        assert "is_buggy" in result_data, "Result should contain 'is_buggy'"

        # Recreate node from result
        node = Node.from_dict(result_data, journal=None)
        print(f"\n[TEST] Recreated node: {node.id}")
        print(f"  - Plan: {node.plan[:100]}...")
        print(f"  - Code length: {len(node.code)} chars")
        print(f"  - Metric: {node.metric}")
        print(f"  - Is buggy: {node.is_buggy}")
        print(f"  - Has plots: {len(node.plots) if node.plots else 0}")

        # Test with Journal
        print("\n[TEST] Testing with Journal...")
        journal = Journal()
        journal.append(node)
        print(f"  - Journal length: {len(journal)}")
        print(f"  - Good nodes: {len(journal.good_nodes)}")
        print(f"  - Buggy nodes: {len(journal.buggy_nodes)}")

        print("\n" + "=" * 80)
        print("✅ Draft node processing test PASSED")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Draft node processing test FAILED")
        print(f"Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_draft_node_processing()
    sys.exit(0 if success else 1)
