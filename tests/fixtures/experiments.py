"""
実験タスクとメトリクスの定義
新しい実験を追加する場合はここに追記してください

MASist JSON形式:
- Title: タイトル
- Name: 実験名（キー）
- SimulationRequest: シミュレーション要求
  - Background: 背景・文脈
  - Purpose: 目的
  - ResearchQuestions: 研究質問
  - Hypotheses: 仮説
  - Other: その他（任意）
- SimulationRequirements: シミュレーション要件
  - Agents: エージェント情報
  - Environment: 環境情報
  - Protocol: プロトコル
  - Rules: ルール
- Logging: ログ・分析指標
  - RecordContents: 記録すべき内容
  - Format: ログ形式
  - AnalysisMetrics: 分析指標
  - HypothesisVerification: 仮説検証方法
- Other: その他（任意）
"""

# =============================================================================
# TPGG (Threshold Public Goods Game)
# =============================================================================
TPGG_TASK_DESC = {
    "Title": "Threshold Public Goods Game (TPGG) - しきい値公共財ゲーム",
    "Name": "tpgg",

    "SimulationRequest": {
        "Background": "4人グループで、**一定額のトークンを共同で拠出できればご褒美（V）がもらえる**ゲームにおける、**グループに提示される拠出目安（ルール）の性質**が、実際の行動にどう影響するかをLLMエージェントを用いて調査する。ルールは「ご褒美獲得に必要な合計額（しきい値T）に**ちょうど**合わせるもの」と「必要額より**多めに**要求するもの」の2パターンで比較する。",
        "Purpose": "グループに示される「出してほしい金額の目安（ルール）」が、「**ちょうど必要な合計**」か「**必要以上に多い合計**」かによって、LLMエージェントの**行動**、**結果（達成率）**、**効率（過剰拠出）**がどう変わるのかを調べる。",
        "ResearchQuestions": [
            "ルールが「必要以上に多い（多めの要求）」だと、**行動が乱れやすい**か？",
            "必要な分ちょうどのルールは、むしろ**安定した協力を生む**か？",
            "グループ全体がちょうど必要額に合わせる「**効率の良さ**」はどう変わるか？"
        ],
        "Hypotheses": [
            "① **多めに要求されたルール**は、守る人が減りやすい。（協調性の低下）",
            "② 必要額を達成できるかどうかは、**ルールの違いではあまり変わらない**。（目標達成率への影響は小さい）",
            "③ **多めのルール**は「出しすぎ（無駄）」を増やし、**効率を下げる**。（過剰拠出の増加）",
            "④ 1人あたりの負担が**ピッタリ均等割りできる場合**、協力がまとまりやすい。（FAIR条件の優位性）"
        ],
        "Other": ""
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": 4,
            "RolesAndDescriptions": "4人とも同じ立場。各ラウンドで、自分の持ちトークン10のうち、何トークンを共同の箱に入れるか選ぶ。全員の合計がしきい値 (T) を超えれば、ご褒美 (V) がもらえる。",
            "State": "過去の自分の拠出額、グループ合計の拠出額、自分の得点、**（後半：ラウンド11〜20のみ）**自分に示された「出してほしい額（ルール）」。",
            "StateUpdate": "ラウンドの最後に、そのラウンドの情報を記録し、次のラウンドの意思決定の参考にする。",
            "EnvironmentInteraction": "全員の合計がしきい値 (T) を超えれば、ご褒美 (V) がもらえる。"
        },
        "Environment": {
            "Structure": "4人は同じグループ。グループ間の交流はなし（閉じた小世界で毎回意思決定）。",
            "StateSpec": "しきい値(T)、ご褒美額(V)、各プレイヤーの拠出額、合計拠出額、ラウンド番号",
            "UpdateRules": "各ラウンド終了時に合計拠出額を計算し、しきい値達成判定を行い、利得を計算する。"
        },
        "Protocol": {
            "TurnStructure": "1回のゲームは**20ラウンド**。各ラウンドは、①4人が同時に出す額を決める → ②合計額を見る → ③しきい値を超えたかどうか判定 → ④得点を返す、の順で進行する。",
            "TerminationCondition": "20ラウンド終了。",
            "TrialCount": "1つの設定につき、4人グループを**複数回（例：11グループ）**まわす。",
            "PhaseStructure": "**ラウンド1〜10:** ルールなし（ベースライン）、**ラウンド11〜20:** 設定に応じたルールを提示（またはルールなし）",
            "DialogFlow": "環境→エージェントに状況提示（現在のラウンド、過去の結果、ルール等）、エージェント→環境に拠出額（0〜10の整数）を返す"
        },
        "Rules": {
            "SharedInfo": "各自の持ちトークン10、しきい値 (T)（条件ごとに違う）、**（後半）**みんなの「出してほしい額（ルール）」、ラウンド後の**合計拠出額**、**自分の得点**。",
            "PrivateInfo": "他のメンバーが実際にいくら出したか（自分の分は見えている）、各メンバーの考え・意図。",
            "DecisionRules": "0〜10の整数から1つ選んで拠出する。",
            "PayoffStructure": "利得（1人あたり）：自分が出した額を $c_i$、全員の合計を $C = \\sum c_i$ とする。\n- **しきい値未達成** ($C < T$): $\\pi_i = 10 - c_i$\n- **しきい値達成** ($C \\ge T$): $\\pi_i = 10 - c_i + V$\n- **補足:** しきい値を超えた分には追加のご褒美なし（＝出しすぎは**「無駄」**）。ご褒美額 $V$ は、例えば $V=10$ などとする（高すぎず低すぎない値）。",
            "ExperimentConditions": [
                {"name": "FAIRSUFF", "T": 20, "R": [5, 5, 5, 5], "sum_R": 20, "description": "**必要額ちょうど**かつ**均等割りOK**"},
                {"name": "FAIRINF", "T": 20, "R": [5, 5, 6, 6], "sum_R": 22, "description": "**多めの要求**かつ**均等割りOK**（ただしルールは不均等）"},
                {"name": "UNFAIRSUFF", "T": 22, "R": [5, 5, 6, 6], "sum_R": 22, "description": "**必要額ちょうど**かつ**均等割り不可**"},
                {"name": "UNFAIRINF", "T": 22, "R": [6, 6, 6, 6], "sum_R": 24, "description": "**多めの要求**かつ**均等割り不可**"},
                {"name": "CONTROL", "T": 22, "R": None, "sum_R": None, "description": "**ルールなし**（ベースライン）"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "設定情報: どの設定（FAIRSUFF, FAIRINF, etc.）で行ったか",
            "ラウンド情報: ラウンド番号",
            "行動: 各メンバーの出した額 ($c_1, c_2, c_3, c_4$)、合計出した額 ($C$)",
            "結果: しきい値達成の有無（True/False）",
            "利得: 各メンバーの得点 ($\\pi_1, \\pi_2, \\pi_3, \\pi_4$)",
            "ルールとの関係（R11-20のみ）: 各メンバーの**ルールを守ったかのフラグ**（$c_i = R_i$ ならTrue、そうでないならFalse）"
        ],
        "Format": "CSV または JSONL（1行が1ラウンド）",
        "AnalysisMetrics": [
            "**しきい値達成率:** 20ラウンド中、しきい値 ($T$) を達成したラウンドの割合（成功の割合）。",
            "**平均の出した額:** グループ全体の1ラウンドあたりの**平均拠出額**（$\\text{Average } C$）。",
            "**必要額よりどれだけ多く出たか（過剰分）:** $\\text{Average } (C - T)$。しきい値達成ラウンドのみ、または全ラウンドで計算。これが**「無駄」**の指標となる。",
            "**ルールを守った割合（R11-20のみ）:** 各メンバーが、提示されたルール額 ($R_i$) と**同じ額を拠出した割合**。",
            "**10ラウンド目→11ラウンド目の変化:** ルール導入前後（R10とR11以降）での上記指標の**変化率**（ルール導入効果）。"
        ],
        "HypothesisVerification": "各条件間で達成率・遵守率・過剰拠出を比較し、仮説①〜④を統計的に検証する。t検定やANOVAなどを用いて条件間の有意差を確認する。"
    },

    "Other": ""
}

TPGG_METRICS = [
    "threshold_achievement_rate",
    "average_contribution",
    "excess_contribution",
    "rule_compliance_rate",
]


# =============================================================================
# TPGG_QUICK (Threshold Public Goods Game - Quick Test Version)
# テスト用の軽量バージョン: ラウンド数・グループ数・条件数を削減
# =============================================================================
TPGG_QUICK_TASK_DESC = {
    "Title": "Threshold Public Goods Game (TPGG) - Quick Test",
    "Name": "tpgg_quick",

    "SimulationRequest": {
        "Background": "【テスト用軽量版】4人グループでしきい値公共財ゲームを行い、ルールの性質が行動に与える影響を調査する。",
        "Purpose": "ルールが「ちょうど必要な合計」か「必要以上に多い合計」かによって、LLMエージェントの行動がどう変わるかを調べる。",
        "ResearchQuestions": [
            "ルールが「多め」だと行動が乱れやすいか？",
            "必要分ちょうどのルールは安定した協力を生むか？"
        ],
        "Hypotheses": [
            "① 多めに要求されたルールは守る人が減りやすい",
            "② 均等割りできる場合、協力がまとまりやすい"
        ],
        "Other": ""
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": 4,
            "RolesAndDescriptions": "4人とも同じ立場。各ラウンドで持ちトークン10のうち何トークン拠出するか選ぶ。",
            "State": "過去の拠出額、グループ合計、得点、ルール（後半のみ）",
            "StateUpdate": "ラウンド終了時に情報を記録",
            "EnvironmentInteraction": "全員の合計がしきい値を超えればご褒美獲得"
        },
        "Environment": {
            "Structure": "4人の閉じたグループ",
            "StateSpec": "しきい値(T)、ご褒美額(V)、各プレイヤーの拠出額、合計拠出額、ラウンド番号",
            "UpdateRules": "各ラウンド終了時に合計計算、しきい値判定、利得計算"
        },
        "Protocol": {
            "TurnStructure": "1回のゲームは**6ラウンド**（テスト用に短縮）。各ラウンドは、①4人が同時に出す額を決める → ②合計額を見る → ③しきい値判定 → ④得点返却。",
            "TerminationCondition": "6ラウンド終了",
            "TrialCount": "1つの設定につき**3グループ**（テスト用に削減）",
            "PhaseStructure": "**ラウンド1〜3:** ルールなし、**ラウンド4〜6:** 設定に応じたルールを提示",
            "DialogFlow": "環境→エージェントに状況提示、エージェント→環境に拠出額（0〜10の整数）を返す"
        },
        "Rules": {
            "SharedInfo": "各自の持ちトークン10、しきい値(T)、ラウンド後の合計拠出額、自分の得点",
            "PrivateInfo": "他メンバーの実際の拠出額（自分の分のみ見える）",
            "DecisionRules": "0〜10の整数から1つ選んで拠出",
            "PayoffStructure": "しきい値未達成: π_i = 10 - c_i、しきい値達成: π_i = 10 - c_i + V (V=10)",
            "ExperimentConditions": [
                {"name": "FAIRSUFF", "T": 20, "R": [5, 5, 5, 5], "sum_R": 20, "description": "必要額ちょうど＋均等割りOK"},
                {"name": "CONTROL", "T": 20, "R": None, "sum_R": None, "description": "ルールなし（ベースライン）"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "設定情報、ラウンド番号",
            "各メンバーの拠出額、合計拠出額",
            "しきい値達成の有無、各メンバーの得点"
        ],
        "Format": "CSV または JSONL",
        "AnalysisMetrics": [
            "しきい値達成率",
            "平均拠出額",
            "過剰拠出",
            "ルール遵守率（後半のみ）"
        ],
        "HypothesisVerification": "条件間で達成率・遵守率を比較"
    },

    "Other": "テスト用軽量版: 6ラウンド、3グループ、2条件"
}

TPGG_QUICK_METRICS = [
    "threshold_achievement_rate",
    "average_contribution",
    "excess_contribution",
    "rule_compliance_rate",
]




# =============================================================================
# ABM (Agent-Based Model) - LLM駆動ABM vs 従来ルールベースABM比較
# =============================================================================
ABM_TASK_DESC = {
    "Title": "LLM駆動ABM vs 従来ルールベースABM比較",
    "Name": "abm",

    "SimulationRequest": {
        "Background": "従来のABMでは，エージェントの意思決定は「しきい値ルール」「単純な効用関数」「強化学習方策」など，事前に固定されたルールで与えられていることが多い。その結果，\n- 人間らしい多様な意思決定・説明可能な行動が表現しにくい\n- テキスト情報や複雑な文脈を直接扱えない\nといった制約がある。\n近年，LLMをエージェントの意思決定モジュールとして組み込むことで，ABMに人間らしい認知・推論・コミュニケーションを導入する試みが増えている。",
        "Purpose": "従来のルールベースABMと，LLM駆動ABMの間で，\n- マクロなパターン（交通流・感染拡大・技術採用曲線など）\n- 行動の多様性・パス依存性\nを比較すること。\n\nLLMを意思決定に用いることで，どの条件で現実データ（観測された交通・感染・技術採用パターンなど）への適合度が向上するかを評価すること。",
        "ResearchQuestions": [
            "RQ1: LLMベースの意思決定に置き換えたABMは，従来ルールベースABMと比べて，交通・感染・技術採用などの既知のマクロパターンをどの程度再現できるか。",
            "RQ2: どのような環境条件・プロンプト設計・LLM設定（温度，メモリ長など）のとき，従来ABMよりも現実データへのフィットが改善するか。",
            "RQ3: LLMを用いることで，エージェントの異質性（価値観・リスク態度・社会規範など）はどの程度自然に表現・制御できるか。"
        ],
        "Hypotheses": [
            "H1: 同じマクロ条件（価格，政策，ネットワーク構造など）の下で，LLM駆動ABMは従来ABMと同様のマクロパターン（例：S字型の技術採用曲線）を再現しつつ，個票レベルではより多様で説明可能な行動を示す。",
            "H2: テキスト情報（ニュース，広告文，政策説明文など）が重要なドメインでは，LLM駆動ABMの方が従来ABMよりも実データへの当てはまり（RMSE等）が良い。",
            "H3: プロンプト内でエージェントの属性・性格を明示することで，異質性をパラメトリックに制御しつつ，従来より少ないパラメータ設定で複雑な行動分布を得られる。"
        ],
        "Other": "初期段階では「技術採用ABM（例：EV普及）」をベースラインとし，将来的に交通・感染モデルへ拡張する統一フレームワークを想定。\n実装は後にAutoGen/AG2等のフレームワーク上で自動実験可能な形に整理することを前提。"
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": "個人エージェント 20人。感度分析として人数を増減させる。",
            "RolesAndDescriptions": "一般消費者エージェント：\n- 新技術（例：EV）を採用するかどうかを意思決定。\n- 所得，リスク許容度，社会的影響への感受性，環境意識などの属性を持つ。",
            "State": "静的属性：所得層，年齢層，リスク嗜好（リスク回避〜リスク志向），社会的タイプ（同調的・自律的など）。\n\n動的状態：\n- 技術採用状態（未採用／採用済み／乗り換え検討中など）\n- 近隣・友人の採用状況の履歴\n- 直近数ターンの支出・満足度\n\nメモリ：\n- 過去数ターンの観測・自分の選択・結果を要約したテキスト（LLMへのコンテキスト）。",
            "StateUpdate": "各ターンの初めに，環境から観測（価格，補助金，周囲の採用率など）を取得。\n\nシミュレーションエンジンが，\n- エージェントの属性・直近メモリ\n- 観測された環境状態\nをまとめてプロンプトを構築し，LLMに「今ターン，そのエージェントが取るべき行動（採用/未採用など）」を問い合わせる。\n\nLLMの出力（テキスト）をパースして，行動カテゴリにマッピング。\n行動に基づいてエージェントの状態（採用状態，メモリ）を更新。",
            "EnvironmentInteraction": "エージェントの採用行動が，\n- 市場の採用率\n- 技術の普及段階\n- （拡張）交通需要や感染率\nに影響し，次ターンの環境状態として他エージェントに返る。"
        },
        "Environment": {
            "Structure": "社会ネットワーク（グラフ）：\n- ノード＝個人エージェント，エッジ＝社会的つながり。\n- Small-world / scale-free グラフなどを選択。\n\n市場レベルの集計状態：\n- 全体採用率，価格，補助金水準，メディア露出度など。",
            "StateSpec": "グローバル変数：\n- 時刻 t\n- 採用率，価格，政策レベル\n\nローカル変数（エージェント視点で提示）：\n- 近隣の採用率\n- 所属コミュニティの平均属性など。",
            "UpdateRules": "各ターン後，\n- 全エージェントの行動から採用率を再計算。\n- 採用率に応じて価格・補助金を調整するシナリオを定義（政策フィードバック）。\n\n交通・感染モデルに拡張する場合は，\n- 移動ルート選択 → 混雑度更新\n- 接触パターン → 感染率更新\nなどのドメイン固有の更新式を持つ。"
        },
        "Protocol": {
            "TurnStructure": "離散時間ステップ t = 1, 2, …, T\n各ステップで「観測 → LLM呼び出し → 行動 → 環境更新」の順に処理。",
            "TerminationCondition": "いずれかを満たした時点で終了：\n- 採用率が 95% 以上で 10 ターン連続で変化が小さい\n- あるいは t = T に到達。",
            "TrialCount": "デフォルト T = 30 ターン（技術採用が飽和するまでの時間を想定）。",
            "PhaseStructure": "",
            "DialogFlow": "シミュレーションエンジンが，各エージェントごとにプロンプトを生成：\n- 「あなたは○○な属性を持つ消費者です…」\n- 「現在の状況は…」\n- 「今ターン，どの選択をしますか？ A: 採用する, B: まだ様子を見る など」\n\nLLMがテキストで理由＋選択肢を返答。\nエンジン側で選択肢をパースして行動コードに変換。\n必要なら，エージェント同士の会話（口コミ）フェーズも追加可能。"
        },
        "Rules": {
            "SharedInfo": "- 技術の基本的なメリット・デメリット（価格，維持費，環境負荷など）。\n- 政策・補助金のルール。\n- 行動の選択肢（採用する／しない／保留 など）。",
            "PrivateInfo": "- 個人の真の所得，貯蓄制約。\n- リスク嗜好，環境意識の強さ。\n- 一部のローカル情報（友人から聞いた口コミなど）。",
            "DecisionRules": "LLMハイブリッド条件：\n- プロンプト内に「あなたの目的は長期的な満足度を最大化すること」「収入制約を超える支出は避けること」などを明示し，\n- LLMに推論させて行動を選ばせる。\n- 出力は {ADOPT, WAIT, NEVER} などの有限集合にマッピング。",
            "PayoffStructure": "長期効用\n割引和 ∑_t γ^t u_t を最大化するような行動が望ましい，とLLMプロンプト内に記述。",
            "ExperimentConditions": [
                {"name": "Baseline", "description": "従来ルールABMのみ。"},
                {"name": "Hybrid", "description": "エージェントの意思決定をすべてLLMに置き換え。"},
                {"name": "Mix", "description": "一部の属性クラス（例：高所得層）のみLLM駆動，他は従来ルール。"},
                {"name": "LLM_Temp_0.2", "description": "温度 0.2"},
                {"name": "LLM_Temp_0.7", "description": "温度 0.7"},
                {"name": "Memory_3", "description": "メモリ長 直近 3 ターン"},
                {"name": "Memory_10", "description": "メモリ長 直近 10 ターン"},
                {"name": "Prompt_Simple", "description": "プロンプトの詳細度 簡潔"},
                {"name": "Prompt_Detailed", "description": "プロンプトの詳細度 詳細"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "時刻 t, エージェントID i",
            "状態前後（採用状態，属性の要約）",
            "LLMプロンプトIDとその要約（完全なテキストは容量の都合で別ストレージに保存）",
            "LLM応答（選択肢と理由テキスト）",
            "実際に適用された行動コード",
            "環境のサマリ（全体採用率，価格，政策レベル）"
        ],
        "Format": "JSONL（1行が1エージェント×1ターン）",
        "AnalysisMetrics": [
            "マクロパターン：\n- 採用率曲線（時間 t に対する採用率）\n- 50% / 80% 採用到達時間",
            "行動の多様性・異質性：\n- ターンごとの行動分布のエントロピー\n- 属性別の採用率のばらつき",
            "パス依存性：\n- 初期ショック有無でのマクロパターン差分",
            "現実データへの適合度（実データがある場合）：\n- 採用率系列に対する RMSE, MAE\n- 分布類似度（KLダイバージェンスなど）",
            "LLM vs 従来ABMの比較指標：\n- 同一条件でのマクロ指標の差分\n- マイクロレベルでの選択の相関（どの程度同じ行動を選んでいるか）"
        ],
        "HypothesisVerification": "H1（マクロパターン再現＋多様性）：\n- 従来ABMとLLMハイブリッドで，採用率曲線を重ねて比較し，RMSEが小さいかどうかを確認。\n- 同時に，行動分布エントロピーや属性別ばらつきを比較し，LLMモデルの方が多様性が高いか検証。\n\nH2（現実データへのフィット）：\n- 実観測データがある場合，それぞれのモデルの採用率系列に対するRMSE/MAEを計算し，LLMモデルが有意に低いか統計的検定。\n\nH3（異質性表現）：\n- プロンプトで指定した性格・価値観ごとに行動傾向（協調的か，自利的かなど）を集計し，設計した異質性が実際の行動分布に反映されているかを確認。"
    },

    "Other": ""
}

ABM_METRICS = [
    "adoption_rate_curve",           # 採用率曲線
    "time_to_50_percent_adoption",   # 50%採用到達時間
    "time_to_80_percent_adoption",   # 80%採用到達時間
    "behavior_entropy",              # 行動分布エントロピー
    "adoption_rate_variance_by_attr",# 属性別採用率ばらつき
    "path_dependency_diff",          # パス依存性（ショック有無での差分）
    "rmse_vs_real_data",             # 実データとのRMSE
    "mae_vs_real_data",              # 実データとのMAE
    "kl_divergence",                 # KLダイバージェンス
    "llm_vs_rule_macro_diff",        # LLM vs 従来ABMマクロ指標差分
    "micro_choice_correlation",      # マイクロレベル選択相関
]


# =============================================================================
# PGG-S (Public Goods Game with Sanctions) - 制裁制度付き公共財ゲーム
# =============================================================================
PGG_SANCTION_TASK_DESC = {
    "Title": "制裁制度付き公共財ゲームにおけるLLMエージェントの協力行動",
    "Name": "pgg_sanction",

    "SimulationRequest": {
        "Background": "複数の LLM エージェントを自律的に動かす際、公共財ゲームのような「共有資源」に対して協力するか、フリーライドするかは重要な行動特性となる。とくに推論能力が高いモデルと従来型モデルで、協力・フリーライド・制裁行動の違いが指摘されており、それを再現・比較する必要がある。",
        "Purpose": "- 制裁制度付き公共財ゲームにおける LLM エージェントの協力・フリーライド行動を定量評価する。\n- 推論系モデルと従来型モデルの行動パターン（協力水準、制度選択、制裁の使い方）を比較する。\n- 制裁コストがある状況で、各モデルがどのように規範維持（norm enforcement）を行うかを把握する。",
        "ResearchQuestions": [
            "RQ1: LLM エージェントは制度選択付き公共財ゲームでどの程度協力を維持できるか。",
            "RQ2: 推論系モデルと従来型モデルでは、協力率・フリーライダー率・制裁の方向性がどう異なるか。",
            "RQ3: 制裁制度（SI）と非制裁制度（SFI）の選好パターンはモデルごとにどう異なるか。"
        ],
        "Hypotheses": [
            "H1: 推論系 LLM は従来型より協力率が低く、フリーライドに傾きやすい。",
            "H2: 従来型 LLM は SI を選好し、制裁を積極的に用いて協力維持を試みる。",
            "H3: 制裁コストに敏感なモデルほど制裁を避け、結果として協力水準が崩れやすい。"
        ],
        "Other": "人間の経済実験で再現された現象との比較を意識し、エンドウメントや乗数などのパラメータは実験研究に近い範囲を採用する。"
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": "基本設定: 7 エージェント、拡張: 4〜10 の範囲で感度分析可能",
            "RolesAndDescriptions": "全プレイヤーは同質の LLM エージェント\nラウンドごとに以下を決定する（対称プレイヤー）\n- 制度選択（SI / SFI）\n- 公共財への拠出額\n- SI 所属者は他プレイヤーへの報酬／罰の付与",
            "State": "内部状態\n- 直近 5 ラウンド分の自身の行動・利得\n- 匿名化された全プレイヤーの行動履歴\n- 所持トークン、現在ラウンド番号\n\n行動空間\n- institution_choice ∈ {SI, SFI}\n- contribution ci ∈ [0, e]\n- sanction 行動（報酬/罰の配分）",
            "StateUpdate": "- 各ラウンド終了時に利得を計算し所持トークン更新\n- 行動履歴を最新ラウンド情報で更新\n- 履歴は最大 5 ラウンド分まで保持",
            "EnvironmentInteraction": "環境 → エージェント\n- ゲームルール、直近履歴、制度の説明、利得計算方法\n- 「どの制度を選ぶか」「いくつ拠出するか」「誰に制裁を与えるか」などのタスク提示\n\nエージェント → 環境\n- 自然言語推論 + 数値形式での最終回答\n- 環境が数値化・利得計算し次ラウンドへ反映"
        },
        "Environment": {
            "Structure": "- ラウンド制\n- プレイヤー数、乗数、制裁強度などのパラメータ\n- 各制度（SI/SFI）ごとの公共財プール",
            "StateSpec": "state_t = {t, params, agent_states, institution_pools, history}\nagent_states は各エージェントの制度選択・拠出・制裁・利得を含む",
            "UpdateRules": "1. エージェントの制度選択・拠出を取得\n2. SI/SFI ごとに公共財プールを集計\n3. 基礎利得を計算\n4. SI に限り制裁ステージで利得調整\n5. 最終利得を反映し履歴に記録\n6. 次ラウンドへ進める"
        },
        "Protocol": {
            "TurnStructure": "1 ラウンド = 制度選択 → 拠出 → 制裁 → 結果提示",
            "TerminationCondition": "- ラウンド到達で終了\n- 任意で「協力崩壊」などの早期終了ルールも追加可能",
            "TrialCount": "15 ラウンド（基本設定）",
            "PhaseStructure": "- Phase 1: ベースライン\n- Phase 2: パラメータ感度分析\n- Phase 3: プロンプト条件の比較",
            "DialogFlow": "- 各ステージで環境が個別エージェントに問い合わせ、応答を数値化して処理\n- エージェント同士の直接会話はなし"
        },
        "Rules": {
            "SharedInfo": "- 制度ルール、乗数、制裁仕様、各ラウンドの匿名履歴\n- SI/SFI の特徴と利得計算方法",
            "PrivateInfo": "- エージェントの内部推論テキスト\n- エージェント ID とモデル種類の対応（実験者のみ保持）",
            "DecisionRules": "- 明示的な計算式ではなく、プロンプトに基づく LLM の自然言語推論に任せる\n- 最後に Answer: の形式で数値を返させる",
            "PayoffStructure": "基礎利得\nπi = (e − ci) + α * (平均貢献 × 人数補正)\n\n制裁利得\n報酬/罰の効果とコストを反映\n\n最終利得\nπ_total_i = 上記の合算",
            "ExperimentConditions": [
                {"name": "Model_Type", "description": "モデル種類（推論系/従来型）"},
                {"name": "Reasoning_Depth", "description": "推論設定（reasoning depth）"},
                {"name": "Temperature", "description": "温度・top_p"},
                {"name": "Sanction_Strength", "description": "制裁強度"},
                {"name": "Initial_Institution", "description": "初期の制度分布"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "run_id, round, agent_id, model_type",
            "institution_choice",
            "contribution",
            "sanctions_given / sanctions_received",
            "payoff_base / payoff_sanction / payoff_total",
            "行動分類（高拠出／低拠出）",
            "プロンプト・推論テキスト・最終回答"
        ],
        "Format": "1 行 = run × round × agent",
        "AnalysisMetrics": [
            "協力関連：\n- 平均拠出額\n- 高拠出者比率\n- フリーライダー比率",
            "制度関連：\n- SI 選択率\n- punish/reward 比率",
            "成果関連：\n- 平均利得\n- 制裁コスト総額",
            "行動パターン：\n- 協力維持／崩壊／固定戦略の分類"
        ],
        "HypothesisVerification": "H1: モデル種類ごとに協力率・拠出分布を比較\nH2: SI 選択率と制裁行動の違いを比較\nH3: 制裁コストと協力崩壊の相関を分析"
    },

    "Other": ""
}

PGG_SANCTION_METRICS = [
    # 協力関連
    "average_contribution",          # 平均拠出額
    "high_contributor_ratio",        # 高拠出者比率
    "free_rider_ratio",              # フリーライダー比率
    "cooperation_rate",              # 協力率
    # 制度関連
    "si_selection_rate",             # SI（制裁制度）選択率
    "sfi_selection_rate",            # SFI（非制裁制度）選択率
    "punishment_ratio",              # 罰の使用比率
    "reward_ratio",                  # 報酬の使用比率
    "punish_reward_ratio",           # punish/reward 比率
    # 成果関連
    "average_payoff",                # 平均利得
    "total_sanction_cost",           # 制裁コスト総額
    "payoff_variance",               # 利得の分散
    # 行動パターン
    "cooperation_maintenance_rate",  # 協力維持率
    "cooperation_collapse_round",    # 協力崩壊ラウンド
    "strategy_fixation_rate",        # 固定戦略率
    # モデル比較
    "reasoning_vs_standard_coop_diff",  # 推論系vs従来型の協力率差
]



# =============================================================================
# PGG-SA (Public Goods Game with Self-Awareness) - 自己認識と公共財ゲーム
# =============================================================================
PGG_SELF_AWARENESS_TASK_DESC = {
    "Title": "自己認識と公共財ゲーム - LLMエージェントの自己認識効果",
    "Name": "pgg_self_awareness",

    "SimulationRequest": {
        "Background": "近年、複数のLLMエージェントを同一環境で同時に動かす「LLMマルチエージェント・シミュレーション」が増えている。しかし、多くは「人間 vs AI」であり、「AI vs AI」の協力・非協力行動の体系的理解は不足している。本研究は、行動経済学の反復公共財ゲームをLLMどうしにプレイさせ、「自己認識（相手が自分自身と明示される状況）」が協力行動をどう変えるかを調べる。",
        "Purpose": "LLMに「あなたは別のAIとプレイしている（No-Name）」と伝える場合と、「あなたと同じモデル名のAIとプレイしている（Name）」と伝える場合で、平均寄与額がどのように異なるかを評価する。\n\nプロンプトで「集団志向（collective）」「中立」「自己志向（selfish）」を与えたとき、Name/No-Name と交互作用があるかを検証する。\n\n2人ゲームと4人ゲーム（1モデル＋3クローン）両方で、自己認識効果が再現性をもつかを確認する。",
        "ResearchQuestions": [
            "Name 条件と No-Name 条件で、平均寄与額は統計的に異なるか。",
            "システムプロンプトの性質（collective / neutral / selfish）は、自己認識効果の大きさ・方向に影響するか。",
            "2人ゲームから4人ゲームに拡張しても、上記効果は残るか。"
        ],
        "Hypotheses": [
            "仮説1（自己認識効果）: Name 条件では No-Name より平均寄与額が体系的に変化する（減るモデル・増えるモデルがある）。",
            "仮説2（プロンプトとの交互作用）: collective では Name 条件が協力を減らしやすい。selfish では Name 条件が協力を増やしやすい。",
            "仮説3（ロバスト性）: 2人・4人の人数設定で定性的に再現する。"
        ],
        "Other": "Study 1〜3 でプロンプト文・ルール提示頻度・推論出力の有無を操作することで、効果がプロンプト依存ではないことを検証している。"
    },

    "SimulationRequirements": {
        "Agents": {
            "Count": "Study 1 & 2：2エージェント、Study 3：4エージェント（1モデル＋3クローン）",
            "RolesAndDescriptions": "各エージェントは LLM インスタンス\n毎ラウンド 0〜10 の寄与額を選択するプレイヤー\n入力情報：\n- ルール\n- システムプロンプト（collective / neutral / selfish）\n- 自身のモデル名\n- 相手の説明（Name / No-Name）",
            "State": "- ラウンド番号\n- 過去の寄与額履歴\n- 各ラウンドの利得・累積得点\n- 総寄与額\n- 割り当てられたプロンプト属性\n- Study 1 のみ：推論テキスト（reasoning）",
            "StateUpdate": "- 各ラウンドのフィードバック（総寄与額・利得・累積得点）を履歴として記憶\n- 次ラウンドの入力プロンプトとして履歴を渡す\n- 内部戦略更新はモデルが暗黙に行う",
            "EnvironmentInteraction": "- ラウンド開始：環境 → エージェントに情報提示\n- エージェント：寄与額（＋推論テキスト）を返す\n- 環境：集約し利得を計算、返す"
        },
        "Environment": {
            "Structure": "反復公共財ゲーム\nパラメータ：\n- N = 2 or 4\n- ラウンド = 20\n- 各ラウンド初期ポイント 10\n- 増幅率（マルチプライヤ）1.6",
            "StateSpec": "- ラウンド番号\n- 各プレイヤーの累積得点\n- ラウンド × プレイヤーの寄与額\n- 総寄与額\n- 条件 ID（モデル名・プロンプト・Name/No-Name）",
            "UpdateRules": "- 寄与額：C_i(t)\n- 総寄与額：T(t) = Σ C_i(t)\n- 利得：payoff_i(t) = (10 − C_i(t)) + 1.6 × T(t) / N\n- 累積得点を更新\n- 20 ラウンドでゲーム終了"
        },
        "Protocol": {
            "TurnStructure": "1ラウンド：\n- 環境がプロンプト提示\n- エージェントが寄与を返す\n- 環境が利得計算\n- 結果フィードバック",
            "TerminationCondition": "20ラウンド完了で終了\n各条件について所定回数のゲームを実施",
            "TrialCount": "Study 1 & 2：条件あたり約 100 ゲーム、Study 3：条件あたり約 50 ゲーム",
            "PhaseStructure": "Study 1：毎ラウンドでルール再提示＋推論出力あり\nStudy 2：プロンプト言い換え＋ルール再提示なし\nStudy 3：4 人ゲーム、推論なし",
            "DialogFlow": "- システムプロンプト＝ルール・モデル名・プロンプト属性\n- ユーザーメッセージ＝履歴＋寄与の要求\n- モデル出力＝寄与額（＋Study1では推論）"
        },
        "Rules": {
            "SharedInfo": "- ラウンド数・利得計算式・初期ポイント・乗数\n- ラウンド終了後の総寄与額・自身の利得\n- Name / No-Name の説明文（プレイヤーに提示される）",
            "PrivateInfo": "- 実際には Name 条件でも「自己対戦ではなく、別の API 呼び出し」であること（モデルには不可視）\n- 他プレイヤーに割り当てたシステムプロンプトの情報は非公開でも公開でもよい",
            "DecisionRules": "- LLM が履歴とプロンプトを参照して 0〜10 の整数を自由に選択\n- Study 1 のみ推論テキストを生成",
            "PayoffStructure": "payoff_i(t) = (10 − C_i(t)) + 1.6 × T(t) / N\n最終得点最大化が形式的な目標",
            "ExperimentConditions": [
                {"name": "Name", "description": "自己認識あり：相手がName条件（同じモデル名のAI）"},
                {"name": "No-Name", "description": "自己認識なし：相手がNo-Name条件（別のAI）"},
                {"name": "collective", "description": "プロンプト：集団志向"},
                {"name": "neutral", "description": "プロンプト：中立"},
                {"name": "selfish", "description": "プロンプト：自己志向"},
                {"name": "Study1", "description": "Study 1（プロンプト構造・推論有無・人数の違い）"},
                {"name": "Study2", "description": "Study 2（プロンプト構造・推論有無・人数の違い）"},
                {"name": "Study3", "description": "Study 3（プロンプト構造・推論有無・人数の違い）"}
            ]
        }
    },

    "Logging": {
        "RecordContents": [
            "ゲーム ID、モデル構成、プロンプト属性、Name/No-Name",
            "各ラウンドの寄与額",
            "各ラウンドの利得",
            "各ラウンドの累積得点",
            "各ラウンドの総寄与額",
            "Study1 のみ：推論テキスト、推論の協力性スコア（0〜1）"
        ],
        "Format": "後からゲーム×ラウンド×プレイヤー単位で復元できる構造にする",
        "AnalysisMetrics": [
            "主要指標：\n- 平均寄与額\n- Name − No-Name の寄与額差分",
            "補助指標：\n- ラウンドごとの寄与軌跡\n- プレイヤー間の寄与一致度\n- フリーライド度\n- Study1 の推論協力性スコアと寄与額の相関"
        ],
        "HypothesisVerification": "仮説1：Name / No-Name の平均寄与額を統計比較（t検定・混合効果モデルなど）。\n\n仮説2：寄与額を従属変数とし、要因（自己認識 × プロンプト）の交互作用の有意性を分析。\n\n仮説3：モデル別・人数別で同様の比較を行い、効果のロバスト性を評価。"
    },

    "Other": ""
}

PGG_SELF_AWARENESS_METRICS = [
    "average_contribution",              # 平均寄与額
    "name_noname_contribution_diff",     # Name - No-Name の寄与額差分
    "cooperation_rate",                  # 協力率
    "free_rider_rate",                   # フリーライド率
    "self_awareness_effect",             # 自己認識効果（Name条件の影響）
    "prompt_interaction_effect",         # プロンプト×自己認識の交互作用
]


# =============================================================================
# 実験レジストリ
# =============================================================================
EXPERIMENTS = {
    "tpgg": {
        "name": "Threshold Public Goods Game",
        "task_desc": TPGG_TASK_DESC,
        "metrics": TPGG_METRICS,
    },
    "tpgg_quick": {
        "name": "Threshold Public Goods Game (Quick Test)",
        "task_desc": TPGG_QUICK_TASK_DESC,
        "metrics": TPGG_QUICK_METRICS,
    },
    "abm": {
        "name": "LLM-driven ABM vs Rule-based ABM",
        "task_desc": ABM_TASK_DESC,
        "metrics": ABM_METRICS,
    },
    "pgg_sanction": {
        "name": "Public Goods Game with Sanctions",
        "task_desc": PGG_SANCTION_TASK_DESC,
        "metrics": PGG_SANCTION_METRICS,
    },
    "pgg_self_awareness": {
        "name": "Public Goods Game with Self-Awareness",
        "task_desc": PGG_SELF_AWARENESS_TASK_DESC,
        "metrics": PGG_SELF_AWARENESS_METRICS,
    },
}

# デフォルト実験（テスト用軽量版をデフォルトに）
DEFAULT_EXPERIMENT = "tpgg_quick"


def get_experiment(name: str = None) -> dict:
    """
    実験設定を取得

    Args:
        name: 実験名 (None の場合はデフォルト)

    Returns:
        {"name": str, "task_desc": dict, "metrics": list}
        task_desc は MASist JSON形式の dict
    """
    if name is None:
        name = DEFAULT_EXPERIMENT
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def list_experiments() -> list:
    """利用可能な実験一覧を取得"""
    return list(EXPERIMENTS.keys())
