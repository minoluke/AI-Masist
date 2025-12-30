"""
BFTS (Best-First Tree Search) ユーティリティ関数
AI-Scientist-v2/ai_scientist/treesearch/bfts_utils.py から移植
"""
import os
import os.path as osp
import shutil
import yaml


def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    Convert a dictionary into a markdown file.
    Supports nested structures with recursive formatting.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """

    def write_value(f, value, indent_level=0):
        """Recursively write values with proper formatting"""
        indent = "  " * indent_level

        if isinstance(value, str):
            # Handle multi-line strings with proper indentation
            lines = value.split('\n')
            for line in lines:
                f.write(f"{indent}{line}\n")
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    f.write(f"{indent}- \n")
                    write_value(f, item, indent_level + 1)
                elif isinstance(item, (list, tuple)):
                    f.write(f"{indent}- \n")
                    write_value(f, item, indent_level + 1)
                else:
                    f.write(f"{indent}- {item}\n")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list, tuple)):
                    f.write(f"{indent}**{sub_key}**:\n")
                    write_value(f, sub_value, indent_level + 1)
                else:
                    f.write(f"{indent}**{sub_key}**: {sub_value}\n")
        else:
            f.write(f"{indent}{value}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # Handle different value types
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        f.write("- \n")
                        write_value(f, item, 1)
                    else:
                        f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n\n")
                    write_value(f, sub_value, 0)
                    f.write("\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            # Assert that the code file exists before trying to open it
            assert os.path.exists(load_code), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
            f.write("## Code To Potentially Use\n\n")
            f.write("Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")


def edit_masist_config_file(config_path: str, idea_dir: str, idea_path: str) -> str:
    """
    Edit the masist_config.yaml file to point to the idea.md file

    Args:
        config_path: Path to the masist_config.yaml file
        idea_dir: Directory where the idea.md file is located
        idea_path: Path to the idea.md file

    Returns:
        Path to the edited masist_config.yaml file
    """
    run_config_path = osp.join(idea_dir, "masist_config.yaml")
    shutil.copy(config_path, run_config_path)
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # make an empty data directory
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
