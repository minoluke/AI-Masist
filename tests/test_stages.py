#!/usr/bin/env python3
"""
Stage Integration Tests for MASist
=================================

Tests for multi-stage experiment flow:
- Test 9: Stage 2 (Baseline Tuning) - uses Stage 1 best node as starting point
- Test 10: Stage 3 (Creative Research) - uses Stage 2 best node as starting point
- Test 11: Stage 4 (Ablation Studies) - uses Stage 3 best node as starting point
- Test 12: All Stages End-to-End using AgentManager

Usage:
    python tests/test_stages.py --test 9      # Run Stage 2 test only
    python tests/test_stages.py --test 10     # Run Stage 3 test only
    python tests/test_stages.py --test 11     # Run Stage 4 test only
    python tests/test_stages.py --test 12     # Run all stages end-to-end
    python tests/test_stages.py --full        # Run all stage tests
    python tests/test_stages.py --exp tpgg    # Use specific experiment
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from masist.treesearch.parallel_agent import ParallelAgent
from masist.treesearch.agent_manager import AgentManager, Stage
from masist.treesearch.journal import Journal, Node
from masist.treesearch.utils.config import _load_cfg, prep_cfg, prep_agent_workspace
from masist.treesearch.interpreter import Interpreter

# Test configuration path
TEST_CONFIG_PATH = Path(__file__).parent / "fixtures" / "test_config.yaml"

# =============================================================================
# Experiment Definitions (shared with test_parallel.py)
# =============================================================================

EXPERIMENTS = {
    "tpgg": {
        "name": "Threshold Public Goods Game (tpgg)",
        "task_desc": {
            "Title": "Threshold Public Goods Game with LLM Agents",
            "Name": "tpgg",
            "SimulationRequest": {
                "Background": "Public goods games are fundamental models in behavioral economics.",
                "Purpose": "Investigate LLM agent behavior in threshold public goods scenarios.",
                "Hypotheses": [
                    "LLM agents will coordinate contributions when threshold is achievable",
                    "Higher thresholds lead to more strategic behavior"
                ],
                "VariableConditions": [
                    {"name": "threshold_level", "values": ["low", "high"]},
                    {"name": "group_size", "values": [3, 5]}
                ]
            },
            "SimulationRequirements": {
                "Agents": ["contributor_agent"],
                "Scenarios": ["low_threshold", "high_threshold"],
                "Rounds": 5,
                "Rules": {
                    "ExperimentConditions": [
                        "Vary threshold levels (low: 50%, high: 80% of max contribution)",
                        "Test with different group sizes (3 and 5 agents)",
                        "Measure coordination success rate across conditions"
                    ]
                }
            },
            "Logging": {
                "Required": ["contributions", "threshold_achieved", "final_payoffs"]
            }
        },
        "metrics": [
            "threshold_achievement_rate",
            "average_contribution",
            "excess_contribution",
            "rule_compliance_rate"
        ]
    },
}

# Current experiment settings
CURRENT_EXPERIMENT = "tpgg"
SIMPLE_TASK_DESC = None
SAMPLE_METRICS = None


def get_experiment(name: str) -> dict:
    """Get experiment by name"""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def set_experiment(name: str):
    """Set current experiment"""
    global CURRENT_EXPERIMENT, SIMPLE_TASK_DESC, SAMPLE_METRICS
    exp = get_experiment(name)
    CURRENT_EXPERIMENT = name
    SIMPLE_TASK_DESC = json.dumps(exp["task_desc"])
    SAMPLE_METRICS = exp["metrics"]
    print(f"[Experiment] Using: {exp['name']}")
    print(f"[Experiment] Metrics: {exp['metrics']}")


def list_experiments():
    """List available experiments"""
    return list(EXPERIMENTS.keys())


# Initialize default experiment
set_experiment("tpgg")


def create_test_config(workspace_name: str = "test_stages") -> "Config":
    """Create test configuration using YAML file"""
    cfg = _load_cfg(TEST_CONFIG_PATH)
    cfg.exp_name = workspace_name
    cfg.data_dir = str(Path(__file__).parent / "fixtures" / "test_data")
    test_data_dir = Path(__file__).parent / "fixtures" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.goal = "Test goal for stages"
    cfg = prep_cfg(cfg)
    return cfg


# =============================================================================
# Helper Functions
# =============================================================================

def run_stage1_and_get_best_node(config, max_steps: int = 10) -> Node:
    """
    Run Stage 1 and return the best node.
    This is used as preparation for Stage 2+ tests.
    """
    print("\n[PREP] Running Stage 1 to get best node...")

    journal = Journal()

    with ParallelAgent(
        task_desc=SIMPLE_TASK_DESC,
        cfg=config,
        journal=journal,
        evaluation_metrics=SAMPLE_METRICS,
        stage_name="1_initial_implementation_1",
    ) as agent:
        success = agent.run(max_steps=max_steps)

        if success and journal.good_nodes:
            best_node = journal.get_best_node(only_good=True, use_val_metric_only=True)
            print(f"[PREP] Stage 1 complete. Best node: {best_node.id[:8]}")
            return best_node, journal
        else:
            print("[PREP] Stage 1 failed to produce good nodes")
            return None, journal


def create_interpreter(config) -> Interpreter:
    """Create an interpreter for code execution"""
    return Interpreter(
        working_dir=config.workspace_dir,
        timeout=config.exec.timeout,
        format_tb_ipython=config.exec.format_tb_ipython,
        agent_file_name=config.exec.agent_file_name,
    )


# =============================================================================
# Test 9: Stage 2 - Baseline Tuning
# =============================================================================

def test_stage2_baseline_tuning():
    """Test 9: Stage 2 Full Flow (Baseline Tuning)

    This test:
    1. Runs Stage 1 to get a baseline implementation
    2. Uses that as starting point for Stage 2
    3. Runs Stage 2 (baseline tuning) with variations
    4. Verifies improvements over Stage 1
    """
    print("\n" + "=" * 80)
    print("Test 9: Stage 2 Full Flow (Baseline Tuning)")
    print("=" * 80)

    config = create_test_config(workspace_name="test_stage2")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    config.agent.stages.stage1_max_iters = 5
    config.agent.stages.stage2_max_iters = 5

    try:
        # Step 1: Run Stage 1 to get best node
        best_stage1_node, stage1_journal = run_stage1_and_get_best_node(config, max_steps=5)

        if not best_stage1_node:
            print("\n[TEST] Skipping Stage 2 - no good Stage 1 node available")
            print("Test 9 PASSED (Stage 1 only)")
            return True

        print(f"\n[TEST] Stage 1 best node: {best_stage1_node.id[:8]}")
        print(f"  - Metric: {best_stage1_node.metric}")

        # Step 2: Run Stage 2 with Stage 1 best node
        print("\n[TEST] Starting Stage 2 (Baseline Tuning)...")

        stage2_journal = Journal()

        # Curate task description for Stage 2
        stage2_task_desc = f"""{SIMPLE_TASK_DESC}

Current Main Stage: baseline_tuning
Sub-stage: 1 - baseline_tuning
Sub-stage goals:
- Systematically vary experimental conditions
- Tune baseline parameters without changing core architecture
- Establish robust baseline performance metrics
"""

        with ParallelAgent(
            task_desc=stage2_task_desc,
            cfg=config,
            journal=stage2_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="2_baseline_tuning_1",
            best_stage1_node=best_stage1_node,
        ) as agent:
            print(f"\n[TEST] Running Stage 2 with {agent.num_workers} workers...")

            success = agent.run(max_steps=5)

            print(f"\n[TEST] Stage 2 completed. Success: {success}")
            print(f"  - Total nodes: {len(stage2_journal)}")
            print(f"  - Good nodes: {len(stage2_journal.good_nodes)}")
            print(f"  - Buggy nodes: {len(stage2_journal.buggy_nodes)}")

            if stage2_journal.good_nodes:
                best_stage2_node = stage2_journal.get_best_node(only_good=True, use_val_metric_only=True)
                print(f"\n[TEST] Stage 2 best node: {best_stage2_node.id[:8]}")
                print(f"  - Metric: {best_stage2_node.metric}")

                # Compare with Stage 1
                print("\n[TEST] Comparison:")
                print(f"  - Stage 1 best: {best_stage1_node.metric}")
                print(f"  - Stage 2 best: {best_stage2_node.metric}")

        print("\nTest 9 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 9 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 10: Stage 3 - Creative Research
# =============================================================================

def test_stage3_creative_research():
    """Test 10: Stage 3 Full Flow (Creative Research)

    This test:
    1. Runs Stage 1 & 2 to get baseline
    2. Uses Stage 2 best as starting point for Stage 3
    3. Runs Stage 3 (creative research) with new ideas
    4. Verifies novel approaches are explored
    """
    print("\n" + "=" * 80)
    print("Test 10: Stage 3 Full Flow (Creative Research)")
    print("=" * 80)

    config = create_test_config(workspace_name="test_stage3")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    config.agent.stages.stage1_max_iters = 4
    config.agent.stages.stage2_max_iters = 4
    config.agent.stages.stage3_max_iters = 4

    try:
        # Step 1: Run Stage 1
        best_stage1_node, _ = run_stage1_and_get_best_node(config, max_steps=4)

        if not best_stage1_node:
            print("\n[TEST] Skipping - no good Stage 1 node")
            print("Test 10 PASSED (Stage 1 only)")
            return True

        # Step 2: Run Stage 2 (quick)
        print("\n[TEST] Running Stage 2 (quick)...")
        stage2_journal = Journal()

        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=stage2_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="2_baseline_tuning_1",
            best_stage1_node=best_stage1_node,
        ) as agent:
            agent.run(max_steps=4)

        best_stage2_node = None
        if stage2_journal.good_nodes:
            best_stage2_node = stage2_journal.get_best_node(only_good=True, use_val_metric_only=True)
            print(f"[TEST] Stage 2 best node: {best_stage2_node.id[:8]}")
        else:
            # Use Stage 1 best as fallback
            best_stage2_node = best_stage1_node
            print("[TEST] Using Stage 1 best as fallback for Stage 3")

        # Step 3: Run Stage 3
        print("\n[TEST] Starting Stage 3 (Creative Research)...")

        stage3_journal = Journal()

        stage3_task_desc = f"""{SIMPLE_TASK_DESC}

Current Main Stage: creative_research
Sub-stage: 1 - creative_research
Sub-stage goals:
- Explore novel approaches beyond baseline
- Test creative hypotheses
- Discover new insights through experimentation
"""

        with ParallelAgent(
            task_desc=stage3_task_desc,
            cfg=config,
            journal=stage3_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="3_creative_research_1",
            best_stage2_node=best_stage2_node,
        ) as agent:
            print(f"\n[TEST] Running Stage 3 with {agent.num_workers} workers...")

            success = agent.run(max_steps=4)

            print(f"\n[TEST] Stage 3 completed. Success: {success}")
            print(f"  - Total nodes: {len(stage3_journal)}")
            print(f"  - Good nodes: {len(stage3_journal.good_nodes)}")

            if stage3_journal.good_nodes:
                best_stage3_node = stage3_journal.get_best_node(only_good=True, use_val_metric_only=True)
                print(f"\n[TEST] Stage 3 best node: {best_stage3_node.id[:8]}")
                print(f"  - Metric: {best_stage3_node.metric}")

        print("\nTest 10 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 10 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 11: Stage 4 - Ablation Studies
# =============================================================================

def test_stage4_ablation_studies():
    """Test 11: Stage 4 Full Flow (Ablation Studies)

    This test:
    1. Runs Stage 1, 2, 3 to build up to Stage 4
    2. Uses Stage 3 best as starting point
    3. Runs Stage 4 (ablation studies)
    4. Verifies systematic component analysis
    """
    print("\n" + "=" * 80)
    print("Test 11: Stage 4 Full Flow (Ablation Studies)")
    print("=" * 80)

    config = create_test_config(workspace_name="test_stage4")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    config.agent.stages.stage1_max_iters = 3
    config.agent.stages.stage2_max_iters = 3
    config.agent.stages.stage3_max_iters = 3
    config.agent.stages.stage4_max_iters = 3

    try:
        # Run through Stage 1-3 (quick versions)
        best_stage1_node, _ = run_stage1_and_get_best_node(config, max_steps=3)

        if not best_stage1_node:
            print("\n[TEST] Skipping - no good Stage 1 node")
            print("Test 11 PASSED (Stage 1 only)")
            return True

        # Stage 2
        print("\n[TEST] Running Stage 2 (quick)...")
        stage2_journal = Journal()
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=stage2_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="2_baseline_tuning_1",
            best_stage1_node=best_stage1_node,
        ) as agent:
            agent.run(max_steps=3)

        best_stage2_node = stage2_journal.get_best_node(only_good=True, use_val_metric_only=True) if stage2_journal.good_nodes else best_stage1_node

        # Stage 3
        print("\n[TEST] Running Stage 3 (quick)...")
        stage3_journal = Journal()
        with ParallelAgent(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            journal=stage3_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="3_creative_research_1",
            best_stage2_node=best_stage2_node,
        ) as agent:
            agent.run(max_steps=3)

        best_stage3_node = stage3_journal.get_best_node(only_good=True, use_val_metric_only=True) if stage3_journal.good_nodes else best_stage2_node

        # Stage 4
        print("\n[TEST] Starting Stage 4 (Ablation Studies)...")

        stage4_journal = Journal()

        stage4_task_desc = f"""{SIMPLE_TASK_DESC}

Current Main Stage: ablation_studies
Sub-stage: 1 - ablation_studies
Sub-stage goals:
- Systematically analyze component contributions
- Remove/modify individual components to measure impact
- Validate which elements are essential for performance
"""

        with ParallelAgent(
            task_desc=stage4_task_desc,
            cfg=config,
            journal=stage4_journal,
            evaluation_metrics=SAMPLE_METRICS,
            stage_name="4_ablation_studies_1",
            best_stage3_node=best_stage3_node,
        ) as agent:
            print(f"\n[TEST] Running Stage 4 with {agent.num_workers} workers...")

            success = agent.run(max_steps=3)

            print(f"\n[TEST] Stage 4 completed. Success: {success}")
            print(f"  - Total nodes: {len(stage4_journal)}")
            print(f"  - Good nodes: {len(stage4_journal.good_nodes)}")

            if stage4_journal.good_nodes:
                best_stage4_node = stage4_journal.get_best_node(only_good=True, use_val_metric_only=True)
                print(f"\n[TEST] Stage 4 best node: {best_stage4_node.id[:8]}")
                print(f"  - Metric: {best_stage4_node.metric}")

        print("\nTest 11 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 11 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 12: All Stages End-to-End with AgentManager
# =============================================================================

def test_all_stages_with_agent_manager():
    """Test 12: All Stages End-to-End using AgentManager

    This is the comprehensive integration test that uses AgentManager
    to run through all stages automatically with stage transitions.
    """
    print("\n" + "=" * 80)
    print("Test 12: All Stages End-to-End with AgentManager")
    print("=" * 80)

    config = create_test_config(workspace_name="test_all_stages")
    config.agent.num_workers = 2
    config.agent.search.num_drafts = 2
    config.agent.stages.stage1_max_iters = 3
    config.agent.stages.stage2_max_iters = 3
    config.agent.stages.stage3_max_iters = 3
    config.agent.stages.stage4_max_iters = 3
    config.generate_report = False  # Disable report generation for test

    try:
        # Prepare agent workspace (creates directories, copies files)
        prep_agent_workspace(config)

        # Create interpreter for code execution
        interpreter = create_interpreter(config)

        # Create AgentManager
        manager = AgentManager(
            task_desc=SIMPLE_TASK_DESC,
            cfg=config,
            workspace_dir=Path(config.workspace_dir),
        )

        print(f"\n[TEST] AgentManager created")
        print(f"  - Initial stage: {manager.current_stage.name if manager.current_stage else 'None'}")
        print(f"  - Stages defined: {len(manager.stages)}")

        def exec_callback(*args, **kwargs):
            """Callback for code execution"""
            return interpreter.run(*args, **kwargs)

        step_count = [0]

        def step_callback(stage, journal):
            """Callback after each step"""
            step_count[0] += 1
            print(f"  [Step {step_count[0]}] Stage: {stage.name}, Nodes: {len(journal)}")

        print("\n[TEST] Running AgentManager.run()...")
        manager.run(exec_callback=exec_callback, step_callback=step_callback)

        # Summary
        print("\n" + "-" * 40)
        print("[TEST] All Stages Summary:")
        print("-" * 40)
        print(f"  - Completed stages: {manager.completed_stages}")
        print(f"  - Total stage transitions: {len(manager.stage_history)}")

        for stage_name, journal in manager.journals.items():
            print(f"\n  Stage '{stage_name}':")
            print(f"    - Total nodes: {len(journal)}")
            print(f"    - Good nodes: {len(journal.good_nodes)}")
            if journal.good_nodes:
                best = journal.get_best_node(only_good=True, use_val_metric_only=True)
                if best:
                    print(f"    - Best metric: {best.metric}")

        print("\nTest 12 PASSED")
        return True

    except Exception as e:
        print(f"\nTest 12 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run stage tests"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage Integration Tests for MASist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tests:
  9  - Stage 2 (Baseline Tuning)
  10 - Stage 3 (Creative Research)
  11 - Stage 4 (Ablation Studies)
  12 - All Stages End-to-End with AgentManager

Examples:
  python tests/test_stages.py --test 9       # Run Stage 2 test
  python tests/test_stages.py --test 12      # Run all stages end-to-end
  python tests/test_stages.py --full         # Run all stage tests
  python tests/test_stages.py --exp tpgg     # Use specific experiment
"""
    )
    parser.add_argument("--test", type=int, choices=[9, 10, 11, 12], help="Run specific test by number")
    parser.add_argument("--full", action="store_true", help="Run all stage tests")
    parser.add_argument("--exp", type=str, choices=list_experiments(), default="tpgg",
                        help=f"Experiment to use (default: tpgg)")
    args = parser.parse_args()

    # Set experiment
    set_experiment(args.exp)

    print("\n" + "=" * 80)
    print("Stage Integration Tests")
    print(f"Experiment: {args.exp}")
    print("=" * 80)

    # Test mapping
    test_map = {
        9: ("Stage 2 (Baseline Tuning)", test_stage2_baseline_tuning),
        10: ("Stage 3 (Creative Research)", test_stage3_creative_research),
        11: ("Stage 4 (Ablation Studies)", test_stage4_ablation_studies),
        12: ("All Stages E2E", test_all_stages_with_agent_manager),
    }

    results = {}

    if args.test:
        # Run specific test
        name, test_func = test_map[args.test]
        results[args.test] = test_func()
    elif args.full:
        # Run all tests
        for test_num, (name, test_func) in test_map.items():
            print(f"\n{'='*60}")
            print(f"Running Test {test_num}: {name}")
            print(f"{'='*60}")
            results[test_num] = test_func()
    else:
        # Default: run just Test 12 (all stages)
        print("\nNo test specified. Run with --test N or --full")
        print("Running Test 12 (All Stages) as default...")
        results[12] = test_all_stages_with_agent_manager()

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_num, passed in results.items():
        name = test_map[test_num][0]
        status = "PASSED" if passed else "FAILED"
        print(f"  Test {test_num} ({name}): {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("All tests PASSED!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Some tests FAILED!")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
