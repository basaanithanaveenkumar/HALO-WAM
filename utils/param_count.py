"""
Utility functions for counting and logging model parameters.
"""

from typing import Optional

import torch.nn as nn

from loguru import logger


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def log_module_parameters(
    model: nn.Module,
    model_name: str = "Model",
    logger_fn = None,
) -> dict[str, dict[str, int]]:
    """
    Log parameter counts for each top-level sub-module of a model.

    Args:
        model:      The nn.Module to inspect.
        model_name: Display name for the model header.
        logger_fn:  Logger instance; uses module-level logger if None.

    Returns:
        Dictionary mapping module name → {"total": ..., "trainable": ...}
    """
    log = logger_fn or logger

    total_all = count_parameters(model, trainable_only=False)
    trainable_all = count_parameters(model, trainable_only=True)

    log.info("=" * 60)
    log.info("  {} — Parameter Summary", model_name)
    log.info("=" * 60)
    log.info("  {:<30s} {:>12s} {:>12s}", "Module", "Total", "Trainable")
    log.info("-" * 60)

    stats: dict[str, dict[str, int]] = {}

    for name, child in model.named_children():
        total = count_parameters(child, trainable_only=False)
        trainable = count_parameters(child, trainable_only=True)
        stats[name] = {"total": total, "trainable": trainable}
        log.info(
            "  {:<30s} {:>10.2f}M {:>10.2f}M",
            name,
            total / 1e6,
            trainable / 1e6,
        )

    log.info("-" * 60)
    log.info(
        "  {:<30s} {:>10.2f}M {:>10.2f}M",
        "TOTAL",
        total_all / 1e6,
        trainable_all / 1e6,
    )
    log.info("=" * 60)

    stats["TOTAL"] = {"total": total_all, "trainable": trainable_all}
    return stats
