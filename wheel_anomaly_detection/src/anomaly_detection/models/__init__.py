"""Anomaly-detection model integrations."""

from .base import AnomalyDetector, AnomalyPrediction
from .factory import build_model
from .patchcore import PatchCore
from .supersimplenet import SuperSimpleNet
from .tinyglass import TinyGLASS
from .trainable import EfficientAD

__all__ = [
    "AnomalyDetector", "AnomalyPrediction", "EfficientAD", "PatchCore",
    "SuperSimpleNet", "TinyGLASS", "build_model",
]
