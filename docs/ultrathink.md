# AI Scientist v2 Stage 2-4 特有プロンプト集

AI Scientist v2 における Stage 2, 3, 4 の特有プロンプトをまとめた文書。

---

## 1. Stage Goals（各ステージの目標）

### Stage 2: Baseline Tuning
```
- Change hyperparameters such as learning rate, number of epochs, batch size, etc. to improve the performance
- DO NOT change the model architecture from the previous stage
- Introduce TWO more new datasets from HuggingFace test the model. Try very hard to think what Huggingface datasets can be used here for testing.
```

### Stage 3: Creative Research
```
- Explore novel improvements
- Come up with experiments to reveal new insights
- Be creative and think outside the box
- MAKE SURE you use THREE HuggingFace dataset in total to test your models
```

### Stage 4: Ablation Studies
```
- Conduct systematic component analysis that reveals the contribution of each part
- Use the same datasets you used from the previous stage
```

---

## 2. Stage 2: Hyperparameter Tuning プロンプト

### 2.1 アイデア生成プロンプト (`_generate_hyperparam_tuning_idea`)

```python
hyperparam_tuning_prompt = {
    "Introduction": (
        "You are an AI researcher conducting hyperparameter tuning for baseline experiments. "
        "Based on the current implementation and previous hyperparameter tuning attempts (if any), "
        "propose ONE new hyperparameter tuning idea to see if it improves the performance."
        "You should first check if simply training longer (more epochs) improves the performance."
        "Then try tuning common hyperparameters such as learning rate, batch size, etc."
        "Only propose algorithm-specific and/or model-specific hyperparameters after you have tried the above."
    ),
    "Base code you are working on": wrap_code(self.best_stage1_node.code),
    "Previous Hyperparam Tuning Attempts": {
        "Has been tried": tried if tried else "Nothing has been tried yet.",
    },
    "Instructions": {
        "Requirements": [
            "1. Identify ONE specific hyperparameter to tune",
            "2. Ensure the hyperparameter is different from previous attempts",
        ]
    },
    "Response format": (
        "Your response should start with 'HYPERPARAM NAME: <hyperparam name>' on the first line to represent the name of the hyperparameter."
        "The second line should start with 'DESCRIPTION: <description>', a brief description of what hyperparameter is being tuned and why (3-5 sentences). "
    ),
}
```

### 2.2 ノード生成プロンプト (`_generate_hyperparam_tuning_node`)

```python
prompt = {
    "Introduction": (
        "You are an experienced AI researcher. You are provided with a previously developed "
        "baseline implementation. Your task is to implement hyperparameter tuning for the following idea: "
        + hyperparam_idea.name
        + ". "
        + hyperparam_idea.description
    ),
    "Base code you are working on": wrap_code(parent_node.code),
    "Instructions": {
        "Implementation guideline": [
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
            "Data saving requirements:",
            "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
            "- Use the following naming convention for saved files:",
            "  ```python",
            "  # At the start of your code",
            "  experiment_data = {",
            "      'hyperparam_tuning_type_1': {",
            "          'dataset_name_1': {",
            "              'metrics': {'train': [], 'val': []},",
            "              'losses': {'train': [], 'val': []},",
            "              'predictions': [],",
            "              'ground_truth': [],",
            "              # Add other relevant data",
            "          },",
            "          # Add additional datasets as needed:",
            "      },",
            "      # Add additional hyperparam tuning types as needed",
            "  }",
            "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
        ]
    },
    "Response format": (
        "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
        "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including hyperparameter tuning. "
        "There should be no additional headings or text in your response. Do not omit any part of the code, "
        "Your generated code should be complete and executable."
        "Make sure to write concise code."
    ),
}
```

### 2.3 Stage 2 完了条件評価プロンプト

```python
eval_prompt = f"""
Evaluate if stage 2 (baseline tuning) is complete based on the following evidence:

1. Figure Analysis:
{vlm_feedback}

2. Datasets Tested: {best_node.datasets_successfully_tested}

Requirements for completion:
1. Training curves should show stable convergence
2. Results should be tested on at least two datasets
3. No major instabilities or issues in the plots

Provide a detailed evaluation of completion status.
"""
```

---

## 3. Stage 3: Creative Research プロンプト

### 3.1 task_desc への追加情報

Stage 3では、`task_desc`に以下の情報が追加される：

```python
if stage.name.startswith("3_"):
    # Experiments（実験計画）を追加
    task_desc += "Experiment Plan: " + experiment_str + "\n"
```

### 3.2 改善プロンプト (`_improve`)

```python
prompt = {
    "Introduction": (
        "You are an experienced AI researcher. You are provided with a previously developed "
        "implementation. Your task is to improve it based on the current experimental stage."
    ),
    "Research idea": self.task_desc,
    "Memory": self.memory_summary if self.memory_summary else "",
    "Feedback based on generated plots": parent_node.vlm_feedback_summary,
    "Feedback about execution time": parent_node.exec_time_feedback,
    "Previous solution": {
        "Code": wrap_code(parent_node.code),
    },
    "Instructions": {},
}
prompt["Instructions"] |= self._prompt_resp_fmt
prompt["Instructions"] |= self._prompt_impl_guideline
```

### 3.3 実行時間フィードバック

Stage 3では、実行時間が短すぎる場合に以下のフィードバックが追加される：

```python
if exec_time_minutes < self.cfg.exec.timeout / 60 / 2:
    exec_time_feedback = (
        f"Implementation works but runs too quickly ({exec_time_minutes:.2f} minutes)."
        "We have up to 60 minutes available for each experiment."
        "Make sure to scale up the experiment "
        "by increasing the number of epochs, using a larger model, or working with bigger datasets."
        "Given that the current execution time is {exec_time_minutes:.2f} minutes, think about how changing the number of epochs to run, or using a larger model, or working with bigger datasets to run"
        "will affect the execution time, and make sure to scale up the experiment accordingly."
    )
```

### 3.4 サブステージ目標生成プロンプト (`_generate_substage_goal`)

```python
prompt = f"""
Based on the current experimental progress, generate focused goals for the next sub-stage.

Main Stage Goals:
{main_stage_goal}

Current Progress:
- Total attempts: {metrics['total_nodes']}
- Successful implementations: {metrics['good_nodes']}
- Best performance: {metrics['best_metric']['value'] if metrics['best_metric'] else 'N/A'}
- Convergence status: {progress['convergence_status']}

Current Issues:
{json.dumps(issues, indent=2)}

Recent Changes:
{json.dumps(progress['recent_changes'], indent=2)}

Generate specific, actionable sub-stage goals that:
1. Address current issues and limitations
2. Build on recent progress
3. Move towards main stage goals
4. Are concrete and measurable
"""
```

---

## 4. Stage 4: Ablation Studies プロンプト

### 4.1 アイデア生成プロンプト (`_generate_ablation_idea`)

```python
ablation_prompt = {
    "Introduction": (
        "You are an AI researcher conducting ablation studies. "
        "Based on the current implementation and previous ablations (if any), "
        "propose ONE new ablation study that tests a different aspect of the model."
    ),
    "Base code you are working on": wrap_code(self.best_stage3_node.code),
    "Previous Ablations": {
        "Has been tried": (
            completed if completed else "Nothing has been tried yet."
        ),
    },
    "Instructions": {
        "Requirements": [
            "1. Identify ONE specific component/feature to ablate",
            "2. Ensure the ablation is different from previous completed or running attempts",
            "3. The ablation should be a new idea, not a variation of previous ideas",
            "4. If you have only used a single synthetic dataset throughout the experiment, one of your ablations should be to use multiple synthetic datasets (at least 3 different datasets)",
        ]
    },
    "Response format": (
        "Your response should start with 'ABLATION NAME: <ablation name>' on the first line to represent the name of the ablation."
        "The second line should start with 'ABLATION DESCRIPTION: <description>', a brief description of what component is being ablated and why (3-5 sentences), "
    ),
}
```

### 4.2 ノード生成プロンプト (`_generate_ablation_node`)

```python
prompt = {
    "Introduction": (
        "You are an experienced AI researcher. You are provided with a previously developed "
        "baseline implementation. Your task is to implement the ablation study for the following idea: "
        + ablation_idea.name
        + ". "
        + ablation_idea.description
    ),
    "Base code you are working on": wrap_code(parent_node.code),
    "Instructions": {
        "Implementation guideline": [
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
            "Data saving requirements:",
            "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
            "- Use the following naming convention for saved files:",
            "  ```python",
            "  # At the start of your code",
            "  experiment_data = {",
            "      'ablation_type_1': {",
            "          'dataset_name_1': {",
            "              'metrics': {'train': [], 'val': []},",
            "              'losses': {'train': [], 'val': []},",
            "              'predictions': [],",
            "              'ground_truth': [],",
            "              # Add other relevant data",
            "          },",
            "          # Add additional datasets as needed:",
            "          'dataset_name_2': {",
            "              'metrics': {'train': [], 'val': []},",
            "              'losses': {'train': [], 'val': []},",
            "              'predictions': [],",
            "              'ground_truth': [],",
            "              # Add other relevant data",
            "          },",
            "      },",
            "      # Add additional ablation types as needed",
            "  }",
            "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
        ]
    },
    "Response format": (
        "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
        "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the ablation study. "
        "There should be no additional headings or text in your response. Do not omit any part of the code, "
        "Your generated code should be complete and executable."
        "Make sure to write concise code."
    ),
}
```

### 4.3 task_desc への追加情報

Stage 4では、`task_desc`に以下の情報が追加される：

```python
if stage.name.startswith("4_"):
    # Risk Factors and Limitations を追加
    task_desc += "Risk Factors and Limitations: " + risk_factors_str + "\n"
```

### 4.4 プロット生成時の追加指示

```python
if self.stage_name.startswith("4_"):
    prompt_guideline.extend([
        "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
        plot_code_from_prev_stage,
        "2. Add comparison plots between ablation and baseline results",
        "3. Add ablation-specific visualizations if needed",
        "4. Include clear labels indicating which plots are from ablation vs baseline",
    ])
```

---

## 5. ステージ進行評価プロンプト

### 5.1 ステージ完了評価 (`stage_completion_eval_spec`)

```python
stage_completion_eval_spec = FunctionSpec(
    name="evaluate_stage_completion",
    description="Evaluate if the current stage is complete",
    json_schema={
        "type": "object",
        "properties": {
            "is_complete": {
                "type": "boolean",
                "description": "Whether the current stage is complete",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the decision",
            },
            "missing_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria still needed",
            },
        },
        "required": ["is_complete", "reasoning", "missing_criteria"],
    },
)
```

### 5.2 ステージ進行評価 (`_evaluate_stage_progression`)

```python
eval_prompt = f"""
Evaluate whether the current experimental stage should progress to the next stage.
Consider all available evidence holistically:

Current Stage Information:
- Name: {current_stage.name}
- Description: {current_stage.description}
- Goals: {', '.join(current_stage.goals) if isinstance(current_stage.goals, list) else current_stage.goals}

Performance Metrics:
{json.dumps(previous_results.get('metrics', {}), indent=2)}

Identified Issues:
{json.dumps(previous_results.get('issues', []), indent=2)}

Progress Analysis:
{json.dumps(previous_results.get('progress', {}), indent=2)}

Expected Stage Progression:
1. Initial Implementation: Focus on basic working implementation
2. Baseline Tuning: Systematic optimization of core parameters
3. Creative Research: Novel improvements and approaches
4. Ablation Studies: Systematic component analysis

Consider factors like:
- Progress toward stage goals
- Performance trends and stability
- Quality and reliability of results
- Understanding of the problem
- Presence of systematic issues
- Convergence indicators
- Readiness for next stage challenges

Provide a holistic evaluation of whether the experiment should:
1. Progress to next stage
2. Continue current stage with specific focus
3. Extend current stage with modifications
"""
```

---

## 6. 各ステージの状態管理

### 6.1 Stage 2 状態 (`_hyperparam_tuning_state`)

```python
self._hyperparam_tuning_state = {
    "tried_hyperparams": set(),  # 試行済みのハイパーパラメータ名
}
```

### 6.2 Stage 4 状態 (`_ablation_state`)

```python
self._ablation_state = {
    "completed_ablations": set(),  # 完了したアブレーション名
}
```

---

## 7. ノード継承関係

| Stage | 継承元 | 使用するベストノード |
|-------|--------|---------------------|
| Stage 2 | Stage 1 | `best_stage1_node` |
| Stage 3 | Stage 2 | `best_stage2_node` |
| Stage 4 | Stage 3 | `best_stage3_node` |

---

## 8. FunctionSpec 定義

### 8.1 ステージ設定生成 (`stage_config_spec`)

```python
stage_config_spec = FunctionSpec(
    name="generate_stage_config",
    description="Generate configuration for the next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Brief, descriptive name for the stage",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the stage's purpose",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific, measurable goals for this stage",
            },
            "max_iterations": {
                "type": "integer",
                "description": "Maximum number of iterations to run in this stage",
            },
        },
        "required": ["name", "description", "goals", "max_iterations"],
    },
)
```

### 8.2 ステージ進行評価 (`stage_progress_eval_spec`)

```python
stage_progress_eval_spec = FunctionSpec(
    name="evaluate_stage_progression",
    description="Evaluate readiness to progress to next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "ready_for_next_stage": {
                "type": "boolean",
                "description": "Whether the experiment is ready to progress to next stage",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the progression decision",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific recommendations for current or next stage",
            },
            "suggested_focus": {
                "type": "string",
                "description": "Key areas to focus on in the next iterations",
            },
        },
        "required": ["ready_for_next_stage", "reasoning", "recommendations"],
    },
)
```

---

## 9. ソースファイル参照

- `agent_manager.py`: ステージ管理、目標定義、遷移評価
- `parallel_agent.py`: ノード生成、アイデア生成、実行制御

---

*このドキュメントは AI-Scientist-v2 のソースコードから抽出されたプロンプト集です。*
