"""
Test script for LLM-based _generate_aggregation_code
Tests that the aggregation code generation works correctly with LLM.

Test cases:
- Test 1: Basic aggregation code generation (LLM call)
- Test 2: Generated code contains required elements
- Test 3: Generated code is executable
- Test 4: Multiple seed nodes with plot_code references
- Test 5: Mock test without LLM (structure validation)
"""
import sys
import json
import logging
import os
import tempfile
import numpy as np
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

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from masist.treesearch import ParallelAgent, Journal, Node
from masist.treesearch.utils.metric import MetricValue
from masist.treesearch.utils.config import _load_cfg, prep_cfg, Config
import shutil

# 実験設定
from fixtures.experiments import get_experiment

EXPERIMENT = get_experiment("tpgg")
# task_desc は dict なので JSON 文字列に変換
TASK_DESC = json.dumps(EXPERIMENT["task_desc"], ensure_ascii=False, indent=2)
METRICS = EXPERIMENT["metrics"]

# テスト用設定ファイルパス
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"


def create_test_config(workspace_name: str = "test_aggregation") -> Config:
    """Create a test configuration from YAML file (AI-Scientist-v2準拠)"""
    cfg = _load_cfg(TEST_CONFIG_PATH)
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.goal = "Test goal for aggregation"
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


def create_mock_seed_nodes(config: Config, num_seeds: int = 3) -> tuple:
    """Create mock seed nodes with experiment data and plot code"""
    temp_dir = os.path.join(config.workspace_dir, "test_mock_seeds")
    os.makedirs(temp_dir, exist_ok=True)

    parent_node = Node(
        plan="Parent node for aggregation test",
        code="# parent code"
    )
    parent_node.is_buggy = False
    parent_node.metric = MetricValue(0.85)

    seed_nodes = []
    for i in range(num_seeds):
        # Create experiment data directory
        exp_dir = os.path.join(temp_dir, f"seed_{i}")
        os.makedirs(exp_dir, exist_ok=True)

        # Create mock experiment data with varying metrics
        exp_data = {
            'scenarios': {
                'scenario_A': {
                    'messages': [f'msg_{i}_1', f'msg_{i}_2'],
                    'events': [],
                    'metrics': {
                        'cooperation_rate': 0.7 + (i * 0.05) + np.random.uniform(-0.02, 0.02),
                        'average_contribution': 5.0 + (i * 0.3) + np.random.uniform(-0.2, 0.2),
                    },
                    'config': {'num_players': 4},
                },
                'scenario_B': {
                    'messages': [f'msg_{i}_3'],
                    'events': [],
                    'metrics': {
                        'cooperation_rate': 0.5 + (i * 0.03) + np.random.uniform(-0.02, 0.02),
                        'average_contribution': 4.0 + (i * 0.2) + np.random.uniform(-0.2, 0.2),
                    },
                    'config': {'num_players': 4},
                },
            },
            'metrics': {
                'overall_cooperation': 0.6 + (i * 0.04),
            },
        }

        np.savez_compressed(
            os.path.join(exp_dir, 'experiment_data.npz'),
            experiment_data=np.array(exp_data, dtype=object)
        )

        # Create seed node with plot_code
        seed_node = Node(
            plan=f"Seed {i} evaluation",
            code="# seed code",
            parent=parent_node,
        )
        seed_node.is_seed_node = True
        seed_node.is_buggy = False
        seed_node.exp_results_dir = exp_dir
        seed_node.plot_code = f'''
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Load data
npz_path = os.path.join(working_dir, '../seed_{i}/experiment_data.npz')
data = np.load(npz_path, allow_pickle=True)
exp_data = data['experiment_data'].item()

# Plot cooperation rate by scenario
scenarios = exp_data['scenarios']
scenario_names = list(scenarios.keys())
coop_rates = [scenarios[s]['metrics']['cooperation_rate'] for s in scenario_names]

plt.figure(figsize=(8, 6))
plt.bar(scenario_names, coop_rates)
plt.ylabel('Cooperation Rate')
plt.title('Cooperation Rate by Scenario (Seed {i})')
plt.savefig(os.path.join(working_dir, 'cooperation_rate_seed{i}.png'))
plt.close()

# Plot average contribution
avg_contribs = [scenarios[s]['metrics']['average_contribution'] for s in scenario_names]
plt.figure(figsize=(8, 6))
plt.bar(scenario_names, avg_contribs, color='orange')
plt.ylabel('Average Contribution')
plt.title('Average Contribution by Scenario (Seed {i})')
plt.savefig(os.path.join(working_dir, 'avg_contribution_seed{i}.png'))
plt.close()
'''
        seed_nodes.append(seed_node)

    return parent_node, seed_nodes, temp_dir


def test_aggregation_code_structure():
    """Test 1: Verify aggregation prompt structure (no LLM call)"""
    print("=" * 80)
    print("Test 1: Aggregation Prompt Structure Validation")
    print("=" * 80)

    config = create_test_config()
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        parent_node, seed_nodes, temp_dir = create_mock_seed_nodes(config)
        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        with ParallelAgent(
            task_desc=TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=METRICS,
        ) as agent:
            # Check _prompt_resp_fmt property
            resp_fmt = agent._prompt_resp_fmt
            assert "Response format" in resp_fmt
            assert "markdown code block" in resp_fmt["Response format"]
            print("  - _prompt_resp_fmt: OK")

            # Check that plan_and_code_query method exists
            assert hasattr(agent, 'plan_and_code_query')
            print("  - plan_and_code_query method: OK")

            # Check that _generate_aggregation_code method exists
            assert hasattr(agent, '_generate_aggregation_code')
            print("  - _generate_aggregation_code method: OK")

        print("\n Test 1 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_code_generation_mock():
    """Test 2: Mock LLM response for aggregation code generation"""
    print("\n" + "=" * 80)
    print("Test 2: Mock LLM Aggregation Code Generation")
    print("=" * 80)

    config = create_test_config()
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        parent_node, seed_nodes, temp_dir = create_mock_seed_nodes(config)
        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        # Mock LLM response
        mock_response = '''
I will create an aggregation script that loads experiment data from all seeds,
calculates mean and standard deviation for each metric, and creates plots with error bars.
The script will generate comparison plots for cooperation rate and average contribution.

```python
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Load data from all seeds
data_paths = ["seed_0", "seed_1", "seed_2"]
all_data = []

for path in data_paths:
    npz_path = os.path.join(os.getenv("MASIST_ROOT", os.getcwd()), path, 'experiment_data.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        all_data.append(data['experiment_data'].item())

# Aggregate metrics
scenarios = list(all_data[0]['scenarios'].keys())
metrics = {}

for scenario in scenarios:
    coop_rates = [d['scenarios'][scenario]['metrics']['cooperation_rate'] for d in all_data]
    avg_contribs = [d['scenarios'][scenario]['metrics']['average_contribution'] for d in all_data]
    metrics[scenario] = {
        'cooperation_rate': {'mean': np.mean(coop_rates), 'std': np.std(coop_rates)},
        'average_contribution': {'mean': np.mean(avg_contribs), 'std': np.std(avg_contribs)},
    }

# Plot with error bars
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    means = [metrics[s]['cooperation_rate']['mean'] for s in scenarios]
    stds = [metrics[s]['cooperation_rate']['std'] for s in scenarios]
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Aggregated Cooperation Rate (Mean +/- Std)')
    plt.savefig(os.path.join(working_dir, 'aggregated_cooperation_rate.png'))
    plt.close()
except Exception as e:
    print(f"Error: {e}")
    plt.close()

print("Aggregation complete")
```
'''

        with patch('masist.treesearch.parallel_agent.query') as mock_query:
            mock_query.return_value = mock_response

            with ParallelAgent(
                task_desc=TASK_DESC,
                cfg=config,
                journal=journal,
                evaluation_metrics=METRICS,
            ) as agent:
                code = agent._generate_aggregation_code(parent_node, seed_nodes)

                # Verify code was extracted
                assert code is not None and len(code) > 0
                print(f"  - Generated code length: {len(code)} chars")

                # Verify required elements
                assert "import matplotlib" in code or "matplotlib.pyplot" in code
                print("  - Contains matplotlib import: OK")

                assert "import numpy" in code or "numpy as np" in code
                print("  - Contains numpy import: OK")

                assert "working_dir" in code
                print("  - Contains working_dir: OK")

                assert "mean" in code.lower() or "np.mean" in code
                print("  - Contains mean calculation: OK")

                assert "std" in code.lower() or "np.std" in code
                print("  - Contains std calculation: OK")

                assert "savefig" in code
                print("  - Contains savefig: OK")

                # Verify LLM was called
                mock_query.assert_called_once()
                print("  - LLM query was called: OK")

        print("\n Test 2 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_prompt_content():
    """Test 3: Verify prompt content includes seed plot codes"""
    print("\n" + "=" * 80)
    print("Test 3: Aggregation Prompt Content Validation")
    print("=" * 80)

    config = create_test_config()
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        parent_node, seed_nodes, temp_dir = create_mock_seed_nodes(config)
        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        captured_prompt = None

        def capture_query(system_message, user_message, model, temperature):
            nonlocal captured_prompt
            captured_prompt = system_message
            return '''
Simple plan.

```python
import matplotlib.pyplot as plt
print("test")
```
'''

        with patch('masist.treesearch.parallel_agent.query', side_effect=capture_query):
            with ParallelAgent(
                task_desc=TASK_DESC,
                cfg=config,
                journal=journal,
                evaluation_metrics=METRICS,
            ) as agent:
                agent._generate_aggregation_code(parent_node, seed_nodes)

        # Verify prompt structure
        assert captured_prompt is not None
        print("  - Prompt was captured: OK")

        assert "Introduction" in captured_prompt
        print("  - Has Introduction: OK")

        assert "Instructions" in captured_prompt
        print("  - Has Instructions: OK")

        instructions = captured_prompt["Instructions"]

        assert "Plotting code reference" in instructions
        print("  - Has Plotting code reference: OK")

        assert "Experiment Data Path" in instructions
        print("  - Has Experiment Data Path: OK")

        # Verify seed plot codes are included
        plot_ref = instructions["Plotting code reference"]
        assert "plotting code 1" in plot_ref
        assert "plotting code 2" in plot_ref
        assert "plotting code 3" in plot_ref
        print("  - All seed plot codes referenced: OK")

        # Verify data paths are included
        data_paths = instructions["Experiment Data Path"]
        assert "seed_0" in data_paths
        assert "seed_1" in data_paths
        assert "seed_2" in data_paths
        print("  - All seed data paths included: OK")

        # Verify aggregation instructions
        intro = captured_prompt["Introduction"]
        assert "aggregate" in intro.lower()
        assert "mean" in intro.lower() or "error bar" in intro.lower()
        print("  - Aggregation instructions present: OK")

        print("\n Test 3 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_code_generation_real():
    """Test 4: Real LLM aggregation code generation (requires API key)"""
    print("\n" + "=" * 80)
    print("Test 4: Real LLM Aggregation Code Generation")
    print("=" * 80)

    # Check if API key is available
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping: No API key available (DEEPSEEK_API_KEY or OPENAI_API_KEY)")
        print("\n Test 4 SKIPPED")
        return True

    config = create_test_config(workspace_name="test_agg_real")
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        parent_node, seed_nodes, temp_dir = create_mock_seed_nodes(config)
        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        with ParallelAgent(
            task_desc=TASK_DESC,
            cfg=config,
            journal=journal,
            evaluation_metrics=METRICS,
        ) as agent:
            print("  - Calling LLM for aggregation code generation...")
            code = agent._generate_aggregation_code(parent_node, seed_nodes)

            assert code is not None and len(code) > 100
            print(f"  - Generated code length: {len(code)} chars")

            # Verify code quality
            assert "import" in code
            print("  - Has imports: OK")

            assert "plt" in code or "matplotlib" in code
            print("  - Uses matplotlib: OK")

            assert "savefig" in code
            print("  - Saves figures: OK")

            # Check for error bar related code
            has_error_bars = any(x in code.lower() for x in ['yerr', 'error', 'std', 'errorbar'])
            if has_error_bars:
                print("  - Contains error bar code: OK")
            else:
                print("  - Warning: No explicit error bar code found")

            # Print first 500 chars of generated code
            print("\n  Generated code preview:")
            print("  " + "-" * 40)
            for line in code[:500].split('\n'):
                print(f"  {line}")
            print("  ...")

        print("\n Test 4 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_code_executable():
    """Test 5: Verify generated code is syntactically valid"""
    print("\n" + "=" * 80)
    print("Test 5: Aggregation Code Syntax Validation")
    print("=" * 80)

    config = create_test_config()
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        parent_node, seed_nodes, temp_dir = create_mock_seed_nodes(config)
        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        # Mock a valid Python code response
        mock_code = '''
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

# Aggregation code
data_paths = ["path1", "path2", "path3"]
all_metrics = []

for path in data_paths:
    try:
        npz_path = os.path.join(path, 'experiment_data.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            all_metrics.append(data['experiment_data'].item())
    except Exception as e:
        print(f"Error: {e}")

# Calculate aggregated statistics
if all_metrics:
    mean_value = np.mean([0.8, 0.85, 0.82])
    std_value = np.std([0.8, 0.85, 0.82])

    try:
        plt.figure(figsize=(10, 6))
        plt.bar(['Metric'], [mean_value], yerr=[std_value], capsize=5)
        plt.title('Aggregated Results (Mean +/- Std)')
        plt.savefig(os.path.join(working_dir, 'aggregated.png'))
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")
        plt.close()

print("Done")
'''

        mock_response = f"Plan text here.\n\n```python\n{mock_code}\n```"

        with patch('masist.treesearch.parallel_agent.query') as mock_query:
            mock_query.return_value = mock_response

            with ParallelAgent(
                task_desc=TASK_DESC,
                cfg=config,
                journal=journal,
                evaluation_metrics=METRICS,
            ) as agent:
                code = agent._generate_aggregation_code(parent_node, seed_nodes)

                # Try to compile the code (syntax check)
                try:
                    compile(code, '<string>', 'exec')
                    print("  - Code syntax is valid: OK")
                except SyntaxError as e:
                    print(f"  - Syntax error: {e}")
                    raise

                # Verify it doesn't have obvious issues
                assert "import matplotlib" in code or "from matplotlib" in code
                print("  - Has matplotlib import: OK")

                assert "working_dir" in code
                print("  - Has working_dir: OK")

                assert "try:" in code and "except" in code
                print("  - Has error handling: OK")

        print("\n Test 5 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_empty_seed_nodes():
    """Test 6: Handle empty or missing plot_code in seed nodes"""
    print("\n" + "=" * 80)
    print("Test 6: Empty Seed Nodes Handling")
    print("=" * 80)

    config = create_test_config()
    # prep_cfg already creates workspace dirs
    journal = Journal()

    try:
        # Create seed nodes without plot_code
        parent_node = Node(plan="Parent", code="# code")
        parent_node.is_buggy = False

        seed_nodes = []
        for i in range(3):
            sn = Node(plan=f"Seed {i}", code="# code", parent=parent_node)
            sn.is_seed_node = True
            sn.exp_results_dir = f"/fake/path/seed_{i}"
            # No plot_code set
            seed_nodes.append(sn)

        journal.append(parent_node)
        for sn in seed_nodes:
            journal.append(sn)

        captured_prompt = None

        def capture_query(system_message, user_message, model, temperature):
            nonlocal captured_prompt
            captured_prompt = system_message
            return "Plan.\n\n```python\nprint('test')\n```"

        with patch('masist.treesearch.parallel_agent.query', side_effect=capture_query):
            with ParallelAgent(
                task_desc=TASK_DESC,
                cfg=config,
                journal=journal,
                evaluation_metrics=METRICS,
            ) as agent:
                code = agent._generate_aggregation_code(parent_node, seed_nodes)

                # Should still work even without plot_code
                assert code is not None
                print("  - Generated code without plot_code reference: OK")

                # Check that it handles missing plot_code gracefully
                plot_ref = captured_prompt["Instructions"]["Plotting code reference"]
                assert "No reference code available" in plot_ref
                print("  - Handles missing plot_code: OK")

        print("\n Test 6 PASSED")
        return True

    except Exception as e:
        print(f"\n Test 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    import argparse
    parser = argparse.ArgumentParser(description="Aggregation Code Generation Tests")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run specific test by number")
    parser.add_argument("--real", action="store_true",
                        help="Run real LLM test (requires API key)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Aggregation Code Generation Tests")
    print("=" * 80)

    test_map = {
        1: ("Prompt Structure", test_aggregation_code_structure),
        2: ("Mock LLM Generation", test_aggregation_code_generation_mock),
        3: ("Prompt Content", test_aggregation_prompt_content),
        4: ("Real LLM Generation", test_aggregation_code_generation_real),
        5: ("Syntax Validation", test_aggregation_code_executable),
        6: ("Empty Seed Nodes", test_empty_seed_nodes),
    }

    results = []

    if args.test:
        name, func = test_map[args.test]
        results.append((name, func()))
    else:
        # Run all tests except real LLM test by default
        for num, (name, func) in test_map.items():
            if num == 4 and not args.real:
                print(f"\n  Skipping Test {num} (Real LLM) - use --real to run")
                continue
            results.append((name, func()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    all_passed = True
    for name, passed in results:
        status = " PASSED" if passed else " FAILED"
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
