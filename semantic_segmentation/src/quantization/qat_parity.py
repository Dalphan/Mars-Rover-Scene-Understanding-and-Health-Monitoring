from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FP32IOFP16Fallback(nn.Module):
    """Keep FP32 I/O while executing fallback operators with FP16 weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.to(torch.float16)).float()


def parity(reference, candidate):
    error = np.abs(candidate.astype(np.float32) - reference.astype(np.float32))
    return {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "prediction_agreement": float(
            np.mean(candidate.argmax(1) == reference.argmax(1))
        ),
    }
