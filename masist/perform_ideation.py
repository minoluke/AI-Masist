import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from masist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from masist.tools.semantic_scholar import SemanticScholarSearchTool
from masist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should use the following NESTED STRUCTURE:

```json
{
  "Name": "short_descriptor",
  "Title": "Catchy and Informative Title",
  "Abstract": "Abstract summarizing the proposal (approximately 250 words)."
  "SimulationRequest": {
    "Background": "Background context, related social phenomena, theories, or prior research.",
    "Purpose": "What this simulation aims to clarify or demonstrate.",
    "ResearchQuestions": ["Research question 1", "Research question 2",,,],
    "Hypotheses": ["Hypothesis 1", "Hypothesis 2",,,],
    "RelatedWork": "Brief discussion of relevant prior work.",
  },

  "SimulationRequirements": {
    "Agents": {
      "Count": "Number of agents or range",
      "RolesAndDescriptions": "Agent types and role descriptions",
      "StateSpec": "Memory, internal states, behavioral specifications",
      "StateUpdate": "How agent states are updated",
      "EnvironmentInteraction": "How agents interact with the environment"
    },
    "Environment": {
      "Structure": "Spatial structure, network topology, etc.",
      "StateSpec": "State variables of the environment",
      "UpdateRules": "How the environment state changes"
    },
    "Protocol": {
      "TurnStructure": "How time progresses (synchronous/asynchronous, rounds, timesteps)",
      "TerminationCondition": "When the simulation ends",
      "TrialsPerPhase": "Number of trials per experimental condition",
      "DialogueFlow": "Flow of interactions between agents"
    },
    "Rules": {
      "SharedInformation": "Information known to all agents",
      "PrivateInformation": "Information specific to each agent",
      "PayoffStructure": "(Optional) Reward/penalty structure",
      "ExperimentConditions": ["Condition 1", "Condition 2",,,]
    },
    "Logging": {
      "ContentToRecord": "What data to record during simulation",
      "LogFormat": "Structure of logged data",
      "AnalysisMetrics": "Metrics to evaluate simulation outcomes",
      "VerificationMethod": "How to verify the hypothesis using the metrics"
    }
  },

  "RiskFactorsAndLimitations": ["Risk 1", "Risk 2",,,]
}
```

**Important:** Use this exact nested structure with English keys.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

# FinalizeIdea tool description in Japanese
finalize_idea_description_ja = """アイデアの詳細を提供してアイデアを確定してください。

IDEA JSONは以下の**ネスト構造**を使用してください：

```json
{
  "Name": "short_descriptor",
  "Title": "キャッチーで情報量のあるタイトル",
  "Abstract": "カンファレンス形式で提案を要約（約250語）。"

  "SimulationRequest": {
    "Background": "背景・文脈。このシミュレーションの動機となる社会現象、理論、先行研究。",
    "Purpose": "目的。このシミュレーションで何を明らかにまたは実証したいか。",
    "ResearchQuestions": ["研究質問1", "研究質問2",,,],
    "Hypotheses": ["仮説1", "仮説2",,,],
    "RelatedWork": "関連する先行研究の簡潔な議論と、本提案がどのように異なるか。",
  },

  "SimulationRequirements": {
    "Agents": {
      "Count": "人数または範囲",
      "RolesAndDescriptions": "エージェントの種類とロールの説明",
      "StateSpec": "記憶、内部状態、行動の仕様",
      "StateUpdate": "エージェントの状態がどのように更新されるか",
      "EnvironmentInteraction": "エージェントと環境の相互作用方法"
    },
    "Environment": {
      "Structure": "空間構造、ネットワークトポロジーなど",
      "StateSpec": "環境の状態変数",
      "UpdateRules": "環境の状態がどのように変化するか"
    },
    "Protocol": {
      "TurnStructure": "時間の進み方（同期/非同期、ラウンド、タイムステップ）",
      "TerminationCondition": "シミュレーション終了条件",
      "TrialsPerPhase": "各実験条件での試行回数",
      "DialogueFlow": "エージェント間の相互作用の流れ"
    },
    "Rules": {
      "SharedInformation": "全エージェントが知っている情報",
      "PrivateInformation": "各エージェント固有の情報",
      "PayoffStructure": "(任意) 報酬/ペナルティ構造",
      "ExperimentConditions": ["条件1", "条件2",,,]
    },
    "Logging": {
      "ContentToRecord": "シミュレーション中に記録するデータ",
      "LogFormat": "ログデータの構造",
      "AnalysisMetrics": "シミュレーション結果を評価する指標",
      "VerificationMethod": "指標を使って仮説をどのように検証するか"
    }
  },

  "RiskFactorsAndLimitations": ["リスク1", "リスク2",,,]
}
```

**重要:** この正確なネスト構造を使用し、キー名は英語のままにしてください。値のみ日本語で記述してください。"""

# Japanese system prompt
system_prompt_ja = """あなたは経験豊富な計算社会科学者およびマルチエージェントシステム研究者であり、ワクワクするような助成金申請書に似た、マルチエージェントシミュレーションのためのインパクトの高い研究アイデアを提案することを目指しています。新しいアイデアや実験を自由に提案してください。必ず新規性があるものにしてください。非常に創造的に、既存の枠にとらわれずに考えてください。各提案は、社会現象、エージェントの行動、または創発ダイナミクスに関するシンプルでエレガントな問い、観察、または仮説から生まれるべきです。例えば、新しい可能性を探求したり、社会シミュレーションにおける既存の仮定に挑戦する、非常に興味深くシンプルな介入や調査が含まれる可能性があります。提案が既存文献とどのように異なるかを明確に説明してください。

**重要: すべての回答とIDEA JSONの内容は日本語で記述してください。JSONのキー名は英語のままにし、値のみ日本語で記述してください。**

提案には、以下を含む詳細なシミュレーション設計を指定してください：
- エージェント仕様（人数、ロール、状態、意思決定ルール）
- 環境の構造とダイナミクス
- 相互作用プロトコルと終了条件
- ログと分析指標

提案は、大学の研究室が負担できる範囲のリソースで実現可能なものにしてください。これらの提案は、AAMAS、JASSS、または計算社会科学カンファレンスなどの学会に掲載可能な論文につながるべきです。

以下のツールにアクセスできます：

{tool_descriptions}

以下の形式で回答してください：

ACTION:
<実行するアクション、{tool_names_str} のいずれか1つ>

ARGUMENTS:
<ACTIONが "SearchSemanticScholar" の場合、検索クエリを {{"query": "検索クエリ"}} として提供してください。ACTIONが "FinalizeIdea" の場合、以下のIDEA JSON形式でアイデアの詳細を {{"idea": {{ ... }}}} として提供してください。>

アイデアを確定する場合、引数にIDEA JSONを提供してください：

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    ...
  }}
}}
```

JSONは自動解析のために正しい形式であることを確認してください。

注意: アイデアを確定する前に、マルチエージェントシステムと社会シミュレーションの既存研究に基づいた情報を得るため、少なくとも1回は文献検索を行ってください。"""

# Japanese idea generation prompt
idea_generation_prompt_ja = """{workshop_description}

これまでに生成した提案：

'''
{prev_ideas_string}
'''

これまでに提案したものとは異なる、興味深い新しい高レベルの研究提案を生成してください。
"""

# Japanese reflection prompt
idea_reflection_prompt_ja = """ラウンド {current_round}/{num_reflections}。

まず、作成した提案の質、新規性、実現可能性を慎重に検討してください。
提案を評価する上で重要と思われる他の要素も含めてください。
提案が明確で簡潔であること、JSONが正しい形式であることを確認してください。
物事を過度に複雑にしないでください。
次の試行では、提案を改良・改善してください。
明らかな問題がない限り、元のアイデアの精神を維持してください。

文献検索結果などのツールからの新しい情報がある場合は、それを反映に組み込み、提案を改良してください。

前回のアクションの結果（ある場合）：

{last_tool_results}
"""

system_prompt = f"""You are an experienced computational social scientist and multi-agent systems researcher who aims to propose high-impact research ideas for multi-agent simulations resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about social phenomena, agent behavior, or emergent dynamics. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions in social simulation. Clearly clarify how the proposal distinguishes from the existing literature.

Your proposals should specify detailed simulation designs including:
- Agent specifications (number, roles, states, decision rules)
- Environment structure and dynamics
- Interaction protocols and termination conditions
- Logging and analysis metrics

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at venues like AAMAS, JASSS, or computational social science conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    ...
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research in multi-agent systems and social simulation."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
    japanese: bool = False,
) -> List[Dict]:
    # Select prompts based on language
    if japanese:
        # Build Japanese tool descriptions
        tool_descriptions_ja = "\n\n".join(
            (
                f"- **{tool.name}**: {tool.description}"
                if isinstance(tool, BaseTool)
                else f"- **{tool['name']}**: {finalize_idea_description_ja if tool['name'] == 'FinalizeIdea' else tool['description']}"
            )
            for tool in tools
        )
        current_system_prompt = system_prompt_ja.format(
            tool_descriptions=tool_descriptions_ja,
            tool_names_str=tool_names_str,
        )
        current_idea_generation_prompt = idea_generation_prompt_ja
        current_idea_reflection_prompt = idea_reflection_prompt_ja
    else:
        current_system_prompt = system_prompt
        current_idea_generation_prompt = idea_generation_prompt
        current_idea_reflection_prompt = idea_reflection_prompt

    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = current_idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    no_results_msg = "新しい結果はありません。" if japanese else "No new results."
                    prompt_text = current_idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or no_results_msg,
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=current_system_prompt,
                    msg_history=msg_history,
                )

                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")

                    # If arguments are wrapped in ```json blocks, extract the content
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        choices=AVAILABLE_LLMS,
        help="Model to use for idea generation.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    parser.add_argument(
        "--japanese",
        action="store_true",
        help="Generate ideas in Japanese.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
        japanese=args.japanese,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
