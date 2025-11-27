"""
Shared configuration classes for unitTest.
Follows AI-Scientist-v2 configuration patterns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast
import shutil

try:
    from omegaconf import OmegaConf, MISSING
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    OmegaConf = None
    MISSING = None


def get_project_root() -> Path:
    """Get the project root directory (unitTest/)"""
    # masist/treesearch/utils/config.py -> unitTest/
    return Path(__file__).parent.parent.parent.parent.resolve()


def get_default_workspace_dir(name: str = "default") -> Path:
    """Get default workspace directory under project root"""
    return get_project_root() / "workspaces" / name


def get_default_log_dir(name: str = "default") -> Path:
    """Get default log directory under project root"""
    return get_project_root() / "logs" / name


@dataclass
class SearchConfig:
    """Search configuration for tree search"""
    num_drafts: int = 4
    debug_prob: float = 0.5
    max_debug_depth: int = 3


@dataclass
class CodeConfig:
    """Configuration for code generation LLM"""
    model: str = "gpt-4o-mini"
    temp: float = 1.0


@dataclass
class FeedbackConfig:
    """Configuration for feedback LLM"""
    model: str = "gpt-4o-mini"
    temp: float = 0.3


@dataclass
class VLMFeedbackConfig:
    """Configuration for VLM feedback"""
    model: str = "gpt-4o-mini"
    temp: float = 0.3


@dataclass
class MultiSeedEvalConfig:
    """Configuration for multi-seed evaluation"""
    enabled: bool = False  # Disabled by default
    num_seeds: int = 3


@dataclass
class ExecConfig:
    """Execution configuration"""
    timeout: int = 120  # seconds
    num_gpus: int = 0
    format_tb_ipython: bool = True
    agent_file_name: str = "agent.py"


@dataclass
class AgentConfig:
    """Agent configuration"""
    num_workers: int = 2
    k_fold_validation: int = 1
    data_preview: bool = False
    code: CodeConfig = field(default_factory=CodeConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    vlm_feedback: VLMFeedbackConfig = field(default_factory=VLMFeedbackConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    multi_seed_eval: MultiSeedEvalConfig = field(default_factory=MultiSeedEvalConfig)


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    num_syn_datasets: int = 2


@dataclass
class Config:
    """
    Main configuration class.

    Usage:
        # Default paths (under unitTest/)
        config = Config()

        # Custom workspace name
        config = Config(workspace_name="my_experiment")

        # Fully custom paths
        config = Config(
            workspace_dir=Path("/custom/workspace"),
            log_dir=Path("/custom/logs")
        )
    """
    # Directory paths - can be overridden
    workspace_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    workspace_name: str = "default"

    # Sub-configurations
    exec: ExecConfig = field(default_factory=ExecConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        """Set default paths if not provided"""
        if self.workspace_dir is None:
            self.workspace_dir = get_default_workspace_dir(self.workspace_name)
        elif isinstance(self.workspace_dir, str):
            self.workspace_dir = Path(self.workspace_dir)

        if self.log_dir is None:
            self.log_dir = get_default_log_dir(self.workspace_name)
        elif isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

    def ensure_dirs(self):
        """Create workspace and log directories if they don't exist"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


def prep_workspace(cfg: Config, data_dir: Optional[Path] = None, copy_data: bool = False):
    """
    Setup the agent's workspace directory structure.
    Following AI-Scientist-v2's prep_agent_workspace pattern.

    Args:
        cfg: Configuration object
        data_dir: Optional source data directory to copy into workspace/input
        copy_data: If True, copy data files; if False, create symlinks (default)
    """
    # Create workspace subdirectories
    input_dir = cfg.workspace_dir / "input"
    working_dir = cfg.workspace_dir / "working"

    input_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink data if provided
    if data_dir is not None and data_dir.exists():
        if copy_data:
            # Copy entire directory tree
            if input_dir.exists():
                shutil.rmtree(input_dir)
            shutil.copytree(data_dir, input_dir)
        else:
            # Create symlinks for each file/directory
            for item in data_dir.iterdir():
                target = input_dir / item.name
                if not target.exists():
                    target.symlink_to(item.resolve())

    return cfg


# ============================================================================
# OmegaConf + YAML Configuration Support
# ============================================================================

def load_config(path: Optional[Path] = None, cli_overrides: bool = True) -> Config:
    """
    Load configuration from YAML file with optional CLI overrides.

    Args:
        path: Path to config.yaml file. If None, uses default config.
        cli_overrides: If True, merge CLI arguments (e.g., `model=gpt-4`)

    Returns:
        Config object

    Example:
        # Load from YAML
        cfg = load_config(Path("config.yaml"))

        # With CLI overrides: python script.py agent.code.model=gpt-4
        cfg = load_config(Path("config.yaml"), cli_overrides=True)
    """
    if not HAS_OMEGACONF:
        raise ImportError(
            "omegaconf is required for YAML config loading. "
            "Install with: pip install omegaconf"
        )

    if path is None:
        # Return default config
        return Config()

    # Load YAML file
    cfg = OmegaConf.load(path)

    # Merge with CLI arguments if requested
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # Create structured config for validation
    cfg_schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def save_config(cfg: Config, path: Path):
    """
    Save configuration to YAML file.

    Args:
        cfg: Configuration object
        path: Output path for YAML file
    """
    if not HAS_OMEGACONF:
        raise ImportError(
            "omegaconf is required for YAML config saving. "
            "Install with: pip install omegaconf"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path)


def print_config(cfg: Config):
    """Pretty print configuration as YAML"""
    if not HAS_OMEGACONF:
        # Fallback: simple dict print
        import pprint
        pprint.pprint(cfg.__dict__)
        return

    try:
        from rich import print as rprint
        from rich.syntax import Syntax
        yaml_str = OmegaConf.to_yaml(cfg)
        rprint(Syntax(yaml_str, "yaml", theme="monokai"))
    except ImportError:
        # Fallback without rich
        print(OmegaConf.to_yaml(cfg))


# Backwards compatibility aliases
SimpleConfig = Config
TestConfig = Config
