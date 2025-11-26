"""
Test where files are actually saved by Interpreter
実際の node_processor.py と同じ条件でテスト
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unitTest.executor.interpreter import Interpreter


def test_interpreter_cwd_real_path():
    """Test with actual workspace path like node_processor.py"""

    # 実際の node_processor.py と同じパス構造
    cfg_workspace_dir = "/Users/idenominoru/Desktop/tmp/workspaces/test_interpreter"
    process_id = "TestProcess-1"

    # node_processor.py:46-53 と同じ
    workspace = os.path.join(cfg_workspace_dir, f"process_{process_id}")
    os.makedirs(workspace, exist_ok=True)

    working_dir = os.path.join(workspace, "working")
    os.makedirs(working_dir, exist_ok=True)

    print("=" * 60)
    print("node_processor.py と同じ条件でテスト")
    print("=" * 60)
    print(f"[SETUP] cfg.workspace_dir: {cfg_workspace_dir}")
    print(f"[SETUP] workspace:         {workspace}")
    print(f"[SETUP] working_dir:       {working_dir}")

    # node_processor.py:57-62 と同じ
    process_interpreter = Interpreter(
        working_dir=workspace,  # ← node_processor.py と同じく workspace を渡す
        timeout=60,
        agent_file_name="agent.py",
    )
    print(f"[SETUP] Interpreter.working_dir: {process_interpreter.working_dir}")

    # 生成コードと同じパターン (code_generator.py:163-166)
    test_code = """
import os
import numpy as np

print(f"=== Inside Interpreter ===")
print(f"os.getcwd() = {os.getcwd()}")

working_dir = os.path.join(os.getcwd(), 'working')
print(f"working_dir = {working_dir}")
print(f"working_dir (absolute) = {os.path.abspath(working_dir)}")

os.makedirs(working_dir, exist_ok=True)

# テストファイル保存
with open(os.path.join(working_dir, 'test.txt'), 'w') as f:
    f.write('hello from interpreter')

# .npz ファイル保存 (実際の生成コードと同じ)
np.savez_compressed(
    os.path.join(working_dir, 'experiment_data_test.npz'),
    experiment_data={'test': 'data'}
)

print(f"Files saved to: {working_dir}")
print(f"Contents: {os.listdir(working_dir)}")
"""

    print("\n[EXEC] Running test code...")
    result = process_interpreter.run(test_code, reset_session=True)
    process_interpreter.cleanup_session()

    print("\n=== Execution Output ===")
    for line in result.term_out:
        print(f"  {line}")

    if result.exc_type:
        print(f"\n[ERROR] Exception: {result.exc_type}")
        print(f"[ERROR] Info: {result.exc_info}")

    print("\n=== Actual Directory Contents ===")
    workspace_path = Path(workspace)
    print(f"{workspace_path.name}/:")
    for f in workspace_path.iterdir():
        if f.is_file():
            print(f"  FILE: {f.name}")
        else:
            print(f"  DIR:  {f.name}/")
            for sub in f.iterdir():
                print(f"    - {sub.name}")

    # 検証: working_dir (workspace/working) にファイルがあるか
    working_path = Path(working_dir)
    expected_files = ['test.txt', 'experiment_data_test.npz']

    print("\n=== Verification ===")
    print(f"Expected save location: {working_dir}")

    all_found = True
    for fname in expected_files:
        fpath = working_path / fname
        if fpath.exists():
            print(f"  ✅ {fname} found in working_dir")
        else:
            print(f"  ❌ {fname} NOT found in working_dir")
            all_found = False

    if all_found:
        print("\n✅ PASS: All files correctly saved to workspace/working/")
    else:
        print("\n❌ FAIL: Files missing from workspace/working/")
        print("\n[DEBUG] Searching entire workspace...")
        for f in workspace_path.rglob("*"):
            if f.is_file():
                print(f"  Found: {f}")

    return all_found


if __name__ == "__main__":
    success = test_interpreter_cwd_real_path()
    sys.exit(0 if success else 1)
