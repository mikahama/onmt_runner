"""Inference-only compatibility layer for legacy OpenNMT-py checkpoints."""

from .translator import Prediction, TensorFlowTranslator, Translator, UnsupportedCheckpointError

__all__ = ["Prediction", "Translator", "TensorFlowTranslator", "UnsupportedCheckpointError"]
