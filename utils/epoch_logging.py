from __future__ import annotations

import math

from pytorch_lightning import Callback


class EpochMetricsPrinter(Callback):
    def __init__(self, metric_prefixes: tuple[str, ...] = ("train_", "val_", "test_")):
        self.metric_prefixes = metric_prefixes
        self._last_printed_validation_epoch = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._print_metrics(trainer, prefix="epoch")
        self._last_printed_validation_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if self._will_validate_this_epoch(trainer):
            return
        if self._last_printed_validation_epoch == trainer.current_epoch:
            return
        self._print_metrics(trainer, prefix="epoch")

    def on_test_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, prefix="test")

    def _print_metrics(self, trainer, prefix: str):
        metrics = self._collect_metrics(trainer.callback_metrics)
        if not metrics:
            return

        metric_text = " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        print(f"[{prefix} {trainer.current_epoch + 1}] {metric_text}")

    def _collect_metrics(self, callback_metrics):
        collected = {}
        for name, value in sorted(callback_metrics.items()):
            if name.endswith("_step"):
                continue
            if not name.startswith(self.metric_prefixes):
                continue

            scalar = self._to_float(value)
            if scalar is not None:
                collected[name] = scalar
        return collected

    @staticmethod
    def _will_validate_this_epoch(trainer) -> bool:
        check_every = getattr(trainer, "check_val_every_n_epoch", None)
        if not getattr(trainer, "enable_validation", False) or not check_every:
            return False
        return (trainer.current_epoch + 1) % check_every == 0

    @staticmethod
    def _to_float(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item") and getattr(value, "numel", lambda: 1)() == 1:
            value = value.item()

        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None
