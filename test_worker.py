"""
Test script for worker node processor
Tests single node processing through all phases
"""
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
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

from unitTest.worker import process_node_wrapper
from unitTest.core import Node, Journal

# サンプルタスク（MASist向け - TPGGシミュレーション）
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
    "threshold_achievement_rate",
    "average_contribution",
    "excess_contribution",
    "rule_compliance_rate",
]


@dataclass
class SimpleConfig:
    """Simple configuration class for testing"""

    @dataclass
    class ExecConfig:
        timeout: int = 300  # 5 minutes
        num_gpus: int = 0
        format_tb_ipython: bool = True
        agent_file_name: str = "agent.py"

    @dataclass
    class AgentConfig:
        @dataclass
        class CodeConfig:
            model: str = "gpt-4o-mini"
            temp: float = 1.0

        @dataclass
        class FeedbackConfig:
            model: str = "gpt-4o-mini"
            temp: float = 0.3

        @dataclass
        class VLMFeedbackConfig:
            model: str = "gpt-4o-mini"
            temp: float = 0.3

        k_fold_validation: int = 1
        data_preview: bool = False

        def __post_init__(self):
            self.code = self.CodeConfig()
            self.feedback = self.FeedbackConfig()
            self.vlm_feedback = self.VLMFeedbackConfig()

    @dataclass
    class ExperimentConfig:
        num_syn_datasets: int = 2

    workspace_dir: str = "/Users/idenominoru/Desktop/tmp/workspaces/test_worker"

    def __post_init__(self):
        self.exec = self.ExecConfig()
        self.agent = self.AgentConfig()
        self.experiment = self.ExperimentConfig()


def test_draft_node_processing():
    """Test processing a draft node through all phases"""
    print("=" * 80)
    print("Testing Draft Node Processing - TPGG Simulation")
    print("=" * 80)

    # Setup
    config = SimpleConfig()
    task_desc = SAMPLE_TASK_DESC

    # Create workspace
    os.makedirs(config.workspace_dir, exist_ok=True)

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
