from __future__ import annotations


def log_batch_shapes(batch, prefix=""):
    """
    Print tensor shapes for debugging.

    Expected segmentation batch:
        image: [B, 3, 512, 512]
        mask:  [B, 512, 512]
    """
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{prefix}{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
