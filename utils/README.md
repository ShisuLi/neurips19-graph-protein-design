# Utils Package - Engineering Standards

This package implements the mandatory engineering standards for the protein design project.

## Overview

The `utils/` package provides:
- **Centralized logging** with console and file output
- **Configuration management** using YAML and argparse
- **Type hints** throughout all code
- **Google-style docstrings** with shape annotations
- **Path handling** using `pathlib.Path`

## Quick Start

### Basic Logger Usage

```python
from utils import setup_logger, get_logger

# Set up logger in your main script
logger = setup_logger(
    name='training',
    log_dir=Path('./logs'),
    log_level=logging.INFO
)

logger.info('Training started')
logger.debug('Batch processed')

# In other modules, retrieve the same logger
logger = get_logger('training')
```

### Tensor Shape Logging

```python
import torch
from utils import setup_logger, TensorShapeLogger

logger = setup_logger('model')
shape_logger = TensorShapeLogger(logger)

x = torch.randn(32, 100, 512)
shape_logger.log_shape('embeddings', x, ['Batch', 'SeqLen', 'HiddenDim'])
# Output: Tensor 'embeddings' shape: [32, 100, 512] (Batch, SeqLen, HiddenDim)
```

### Configuration Management

#### Using YAML Configuration

```python
from pathlib import Path
from utils import TrainingConfig

# Load from YAML
config = TrainingConfig.from_yaml(Path('configs/default_training.yaml'))

print(f'Hidden dim: {config.hidden_dim}')
print(f'Batch size: {config.batch_size}')

# Save configuration
config.save(Path('outputs/experiment_config.yaml'))
```

#### Using Command-Line Arguments

```python
from utils import parse_training_config

# Parse from command line
config = parse_training_config()

# Or parse specific arguments
config = parse_training_config([
    '--hidden-dim', '256',
    '--batch-size', '64',
    '--learning-rate', '0.0001'
])
```

#### Combining YAML and CLI (Recommended)

```python
# CLI arguments override YAML values
config = parse_training_config([
    '--config', 'configs/default_training.yaml',
    '--learning-rate', '0.0001',  # Override LR from YAML
    '--experiment-name', 'custom_experiment'
])
```

### Complete Training Script Example

```python
from pathlib import Path
import logging
import torch
from utils import (
    setup_logger,
    get_logger,
    TensorShapeLogger,
    parse_training_config
)

def main():
    # Parse configuration
    config = parse_training_config()

    # Set up logging
    logger = setup_logger(
        name='training',
        log_dir=Path('./logs'),
        log_level=logging.INFO
    )
    shape_logger = TensorShapeLogger(logger)

    # Log configuration
    logger.info(f'Starting experiment: {config.experiment_name}')
    logger.info(f'Config: {config.to_dict()}')

    # Save config for reproducibility
    config_output_dir = config.checkpoint_dir / 'configs'
    config_output_dir.mkdir(parents=True, exist_ok=True)
    config.save(config_output_dir / 'config.yaml')

    # Initialize W&B (gracefully handle if disabled)
    if config.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.to_dict()
            )
            logger.info('W&B initialized successfully')
        except Exception as e:
            logger.warning(f'W&B initialization failed: {e}')

    # Model training loop
    for epoch in range(config.num_epochs):
        # Dummy batch
        batch = torch.randn(config.batch_size, 100, config.hidden_dim)

        shape_logger.log_shape('batch', batch, ['Batch', 'Seq', 'Hidden'])
        logger.info(f'Epoch {epoch+1}/{config.num_epochs}')

        # Your training code here...

        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f'model_epoch_{epoch+1}.pt'
            logger.info(f'Saving checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    main()
```

## Engineering Standards Checklist

When writing new code, ensure:

- [ ] **Type hints** on all function signatures
- [ ] **Google-style docstrings** with Args, Returns, and Shape info
- [ ] **`pathlib.Path`** for all path operations (no string concatenation)
- [ ] **`logging`** instead of `print()`
- [ ] **No hardcoded hyperparameters** in model code
- [ ] **W&B integration** for experiment tracking
- [ ] **Proper error handling** with informative messages

## File Structure

```
utils/
├── __init__.py          # Package exports
├── logger.py            # Logging utilities
├── config.py            # Configuration management
└── README.md            # This file

configs/
└── default_training.yaml  # Default configuration

logs/                    # Created automatically
└── training_*.log       # Timestamped log files
```

## Dependencies

All required dependencies are listed in `requirements.txt`:
- `pyyaml` - YAML configuration files
- `wandb` - Experiment tracking
- `torch` - Deep learning framework

Install with:
```bash
pip install -r requirements.txt
```
