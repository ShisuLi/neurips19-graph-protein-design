"""Central logging configuration for the protein design project.

This module provides a centralized logging setup that outputs to both console
and file, following best practices for research code.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "protein_design",
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """Set up a logger with console and file handlers.

    This function creates a logger that outputs formatted messages to both
    the console (stdout) and a log file. The log file is timestamped to
    prevent overwrites across different runs.

    Args:
        name: Name of the logger. Defaults to 'protein_design'.
        log_dir: Directory where log files will be saved. If None, defaults to
            './logs'. Will be created if it doesn't exist.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.
        console_output: Whether to output logs to console. Defaults to True.
        file_output: Whether to output logs to file. Defaults to True.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger(name='training', log_level=logging.DEBUG)
        >>> logger.info('Training started')
        >>> logger.debug('Batch shape: [32, 100, 512]')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate logs

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_dir is None:
            log_dir = Path('./logs')
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f'Log file created at: {log_file}')

    return logger


def get_logger(name: str = "protein_design") -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    This is a convenience function to retrieve loggers in different modules
    without having to reconfigure them.

    Args:
        name: Name of the logger to retrieve.

    Returns:
        Logger instance.

    Example:
        >>> # In one module
        >>> logger = setup_logger('training')
        >>>
        >>> # In another module
        >>> logger = get_logger('training')  # Retrieves the same logger
    """
    logger = logging.getLogger(name)

    # If logger hasn't been configured yet, set it up with defaults
    if not logger.hasHandlers():
        return setup_logger(name)

    return logger


class TensorShapeLogger:
    """Helper class for logging tensor shapes in a structured way.

    This class provides utilities for logging tensor shapes with clear
    dimension labels, which is crucial for debugging deep learning models.

    Example:
        >>> shape_logger = TensorShapeLogger(logger)
        >>> shape_logger.log_shape('input', x, ['Batch', 'Length', 'Features'])
        INFO - Tensor 'input' shape: [32, 100, 512] (Batch, Length, Features)
    """

    def __init__(self, logger: logging.Logger):
        """Initialize the TensorShapeLogger.

        Args:
            logger: Logger instance to use for output.
        """
        self.logger = logger

    def log_shape(
        self,
        name: str,
        tensor,
        dim_names: Optional[list[str]] = None,
        level: int = logging.DEBUG
    ) -> None:
        """Log the shape of a tensor with optional dimension names.

        Args:
            name: Name/description of the tensor.
            tensor: The tensor object (must have a .shape attribute).
            dim_names: Optional list of dimension names for clarity.
            level: Logging level to use. Defaults to DEBUG.

        Example:
            >>> import torch
            >>> x = torch.randn(32, 100, 512)
            >>> shape_logger.log_shape('embeddings', x, ['Batch', 'Seq', 'Dim'])
        """
        shape = list(tensor.shape)
        shape_str = f"[{', '.join(map(str, shape))}]"

        if dim_names:
            if len(dim_names) != len(shape):
                self.logger.warning(
                    f"Dimension names count ({len(dim_names)}) doesn't match "
                    f"shape dimensions ({len(shape)}) for tensor '{name}'"
                )
            dim_str = f" ({', '.join(dim_names)})"
        else:
            dim_str = ""

        self.logger.log(level, f"Tensor '{name}' shape: {shape_str}{dim_str}")
