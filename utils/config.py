"""Configuration management for the protein design project.

This module provides utilities for loading and managing configuration from
both command-line arguments and YAML files, following best practices for
reproducible research.
"""

import argparse
import yaml
from pathlib import Path
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass, asdict, field
import json


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> config = load_yaml_config(Path('configs/train.yaml'))
        >>> print(config['model']['hidden_dim'])
        128
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_yaml_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Dictionary containing configuration parameters.
        output_path: Path where the YAML file will be saved.

    Example:
        >>> config = {'model': {'hidden_dim': 128}, 'batch_size': 32}
        >>> save_yaml_config(config, Path('outputs/config.yaml'))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Values in override_config take precedence over base_config.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration dictionary with override values.

    Returns:
        Merged configuration dictionary.

    Example:
        >>> base = {'model': {'dim': 128, 'layers': 4}, 'lr': 0.001}
        >>> override = {'model': {'dim': 256}, 'lr': 0.0001}
        >>> merged = merge_configs(base, override)
        >>> print(merged['model']['dim'])  # 256 (overridden)
        >>> print(merged['model']['layers'])  # 4 (from base)
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


@dataclass
class TrainingConfig:
    """Configuration for training the protein design model.

    This dataclass encapsulates all training hyperparameters and settings,
    making it easy to serialize, version, and track experiments.

    Attributes:
        # Model architecture
        hidden_dim: Hidden dimension size for the model.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

        # Training hyperparameters
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_epochs: Number of training epochs.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        gradient_clip: Maximum gradient norm for clipping.

        # Data
        data_dir: Directory containing training data.
        val_split: Fraction of data to use for validation.
        max_seq_len: Maximum sequence length.

        # Experiment tracking
        experiment_name: Name of the experiment for logging.
        wandb_project: Weights & Biases project name.
        wandb_enabled: Whether to enable W&B logging.

        # Checkpointing
        checkpoint_dir: Directory to save model checkpoints.
        save_every: Save checkpoint every N epochs.
        keep_last_n: Keep only the last N checkpoints.

        # System
        seed: Random seed for reproducibility.
        device: Device to use ('cuda' or 'cpu').
        num_workers: Number of data loading workers.

    Example:
        >>> config = TrainingConfig(hidden_dim=256, batch_size=64)
        >>> print(config.hidden_dim)
        256
    """

    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Data
    data_dir: Path = field(default_factory=lambda: Path('./data'))
    val_split: float = 0.1
    max_seq_len: int = 500

    # Experiment tracking
    experiment_name: str = 'protein_design'
    wandb_project: str = 'protein-design'
    wandb_enabled: bool = True

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path('./checkpoints'))
    save_every: int = 10
    keep_last_n: int = 5

    # System
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 4

    def __post_init__(self) -> None:
        """Convert path strings to Path objects after initialization."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        config_dict = asdict(self)
        # Convert Path objects to strings for serialization
        config_dict['data_dir'] = str(self.data_dir)
        config_dict['checkpoint_dir'] = str(self.checkpoint_dir)
        return config_dict

    def save(self, output_path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            output_path: Path where the config will be saved.

        Example:
            >>> config = TrainingConfig(hidden_dim=256)
            >>> config.save(Path('outputs/config.yaml'))
        """
        save_yaml_config(self.to_dict(), output_path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create a TrainingConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            TrainingConfig instance.

        Example:
            >>> config_dict = {'hidden_dim': 256, 'batch_size': 64}
            >>> config = TrainingConfig.from_dict(config_dict)
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'TrainingConfig':
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance.

        Example:
            >>> config = TrainingConfig.from_yaml(Path('configs/train.yaml'))
        """
        config_dict = load_yaml_config(yaml_path)
        return cls.from_dict(config_dict)


def create_training_parser() -> argparse.ArgumentParser:
    """Create an argument parser for training configuration.

    This parser can be used standalone or extended with additional arguments.
    Command-line arguments can override YAML configuration values.

    Returns:
        ArgumentParser configured with training arguments.

    Example:
        >>> parser = create_training_parser()
        >>> args = parser.parse_args(['--hidden-dim', '256', '--batch-size', '64'])
        >>> print(args.hidden_dim)
        256
    """
    parser = argparse.ArgumentParser(
        description='Train protein design model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to YAML configuration file. CLI args override config file values.'
    )

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--hidden-dim', type=int, default=128,
                           help='Hidden dimension size')
    model_group.add_argument('--num-layers', type=int, default=3,
                           help='Number of layers')
    model_group.add_argument('--num-heads', type=int, default=4,
                           help='Number of attention heads')
    model_group.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout probability')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--batch-size', type=int, default=32,
                           help='Training batch size')
    train_group.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                           help='Initial learning rate')
    train_group.add_argument('--num-epochs', type=int, default=100,
                           help='Number of training epochs')
    train_group.add_argument('--warmup-steps', type=int, default=1000,
                           help='Number of warmup steps for LR scheduler')
    train_group.add_argument('--gradient-clip', type=float, default=1.0,
                           help='Maximum gradient norm for clipping')

    # Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data-dir', type=Path, default=Path('./data'),
                          help='Directory containing training data')
    data_group.add_argument('--val-split', type=float, default=0.1,
                          help='Fraction of data for validation')
    data_group.add_argument('--max-seq-len', type=int, default=500,
                          help='Maximum sequence length')

    # Experiment tracking
    exp_group = parser.add_argument_group('Experiment Tracking')
    exp_group.add_argument('--experiment-name', type=str, default='protein_design',
                         help='Name of the experiment for logging')
    exp_group.add_argument('--wandb-project', type=str, default='protein-design',
                         help='Weights & Biases project name')
    exp_group.add_argument('--wandb-enabled', action='store_true', default=True,
                         help='Enable W&B logging')
    exp_group.add_argument('--no-wandb', dest='wandb_enabled', action='store_false',
                         help='Disable W&B logging')

    # Checkpointing
    ckpt_group = parser.add_argument_group('Checkpointing')
    ckpt_group.add_argument('--checkpoint-dir', type=Path, default=Path('./checkpoints'),
                          help='Directory to save model checkpoints')
    ckpt_group.add_argument('--save-every', type=int, default=10,
                          help='Save checkpoint every N epochs')
    ckpt_group.add_argument('--keep-last-n', type=int, default=5,
                          help='Keep only the last N checkpoints')

    # System
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument('--seed', type=int, default=42,
                         help='Random seed for reproducibility')
    sys_group.add_argument('--device', type=str, default='cuda',
                         choices=['cuda', 'cpu'],
                         help='Device to use for training')
    sys_group.add_argument('--num-workers', type=int, default=4,
                         help='Number of data loading workers')

    return parser


def parse_training_config(args: Optional[list[str]] = None) -> TrainingConfig:
    """Parse training configuration from command line and/or YAML file.

    Command-line arguments take precedence over YAML configuration values.

    Args:
        args: List of command-line arguments. If None, uses sys.argv.

    Returns:
        TrainingConfig instance with merged configuration.

    Example:
        >>> # With YAML file
        >>> config = parse_training_config(['--config', 'configs/train.yaml'])
        >>>
        >>> # With CLI args only
        >>> config = parse_training_config(['--hidden-dim', '256', '--batch-size', '64'])
        >>>
        >>> # YAML + CLI overrides
        >>> config = parse_training_config([
        ...     '--config', 'configs/train.yaml',
        ...     '--learning-rate', '0.0001'
        ... ])
    """
    parser = create_training_parser()
    parsed_args = parser.parse_args(args)

    # Start with default config
    config_dict = asdict(TrainingConfig())

    # Load from YAML if provided
    if parsed_args.config is not None:
        yaml_config = load_yaml_config(parsed_args.config)
        config_dict = merge_configs(config_dict, yaml_config)

    # Override with command-line arguments
    cli_config = {}
    for key, value in vars(parsed_args).items():
        if key != 'config':  # Skip the config file path itself
            # Convert hyphenated names to underscored
            key_normalized = key.replace('-', '_')
            cli_config[key_normalized] = value

    config_dict = merge_configs(config_dict, cli_config)

    return TrainingConfig.from_dict(config_dict)
