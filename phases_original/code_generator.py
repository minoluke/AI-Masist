"""
Code Generator for Draft Phase
Extracts and adapts code from parallel_agent.py MinimalAgent class
"""
import random
from typing import Any, Tuple
import humanize

from ..core.node import Node
from ..llm.backend import query
from ..utils.response import extract_code, extract_text_up_to_code


class CodeGenerator:
    """Generate Python code from research ideas using LLM"""

    def __init__(self, task_desc: str, evaluation_metrics: list, cfg: Any, memory_summary: str = None):
        self.task_desc = task_desc
        self.evaluation_metrics = evaluation_metrics
        self.cfg = cfg
        self.memory_summary = memory_summary
        self.data_preview = None

    def generate(self) -> Node:
        """Generate a draft implementation (equivalent to _draft())"""
        prompt: Any = {
            "Introduction": (
                "You are an AI researcher who is looking to publish a paper that will contribute significantly to the field."
                "Your first task is to write a python code to implement a solid baseline based on your research idea provided below, "
                "from data preparation to model training, as well as evaluation and visualization. "
                "Focus on getting a simple but working implementation first, before any sophisticated improvements. "
                "We will explore more advanced variations in later stages."
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design. ",
                "The solution sketch should be 6-10 sentences. ",
                "Don't suggest to do EDA.",
                "Make sure to create synthetic data if needed.",
                "",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        print("[cyan]--------------------------------[/cyan]")
        print("[cyan]self.task_desc[/cyan]")
        print("[cyan]" + self.task_desc + "[/cyan]")
        print("[cyan]--------------------------------[/cyan]")

        print("CodeGenerator: Getting plan and code")
        plan, code = self.plan_and_code_query(prompt)
        print("CodeGenerator: Draft complete")
        return Node(plan=plan, code=code)

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "albumentations",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:",
            "  - At the start of your code, add these lines to handle GPU/CPU:",
            "    ```python",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "    print(f'Using device: {device}')",
            "    ```",
            "  - ALWAYS move models to device using the `.to(device)` method",
            "  - ALWAYS move input tensors to device using the `.to(device)` method",
            "  - ALWAYS move model related tensors to device using the `.to(device)` method",
            "  - For optimizers, create them AFTER moving model to device",
            "  - When using DataLoader, move batch tensors to device in training loop: `batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}`",
            "CRITICAL MODEL INPUT GUIDELINES:",
            "  - Always pay extra attention to the input to the model being properly normalized",
            "  - This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output",
        ]
        if hasattr(self.cfg.experiment, "num_syn_datasets"):
            num_syn_datasets = self.cfg.experiment.num_syn_datasets
            if num_syn_datasets > 1:
                impl_guideline.extend(
                    [
                        f"You MUST evaluate your solution on at least {num_syn_datasets} different synthetic datasets to ensure robustness:",
                        "  - Use standard benchmark datasets when available",
                        f"  - If using synthetic data, generate at least {num_syn_datasets} variants with different characteristics",
                        "  - Report metrics separately for each dataset",
                        "  - Compute and report the average metric across all datasets",
                    ]
                )
        impl_guideline.extend(
            [
                "For generative modeling tasks, you must:",
                "  - Generate a set of samples from your model",
                "  - Compare these samples with ground truth data using appropriate visualizations",
                "  - When saving plots, always use the 'working_dir' variable that will be defined at the start of the script",
                "  - Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.",
                "Important code structure requirements:",
                "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block",
                "  - All code should be at the global scope or in functions that are called from the global scope",
                "  - The script should execute immediately when run, without requiring any special entry point",
                "The code should start with:",
                "  import os",
                "  working_dir = os.path.join(os.getcwd(), 'working')",
                "  os.makedirs(working_dir, exist_ok=True)",
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Your response should only contain a single code block.",
                f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
                'You can also use the "./working" directory to store any temporary files that your code needs to create.',
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'dataset_name_1': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "      # Add additional datasets as needed:",
                "      'dataset_name_2': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          # Add other relevant data",
                "      },",
                "  }",
                "  # During training/evaluation:",
                "  experiment_data['dataset_name_1']['metrics']['train'].append(train_metric)",
                "  ```",
                "- Include timestamps or epochs with the saved metrics",
                "- For large datasets, consider saving in chunks or using np.savez_compressed()",
                "CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:",
                "  1. Track and print validation loss at each epoch or at suitable intervals:",
                "     ```python",
                "     print(f'Epoch {{epoch}}: validation_loss = {{val_loss:.4f}}')",
                "     ```",
                "  2. Track and update ALL these additional metrics: "
                + str(self.evaluation_metrics),
                "  3. Update metrics at EACH epoch:",
                "  4. Save ALL metrics at the end:",
                "     ```python",
                "     np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
                "     ```",
            ]
        )

        if self.cfg.agent.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.cfg.agent.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Make sure to write concise code."
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> Tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore
