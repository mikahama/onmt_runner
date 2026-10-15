"""Inference-only compatibility layer for legacy OpenNMT-py checkpoints."""

from .translator import TensorFlowTranslator, Translator, UnsupportedCheckpointError

__all__ = ["Translator", "TensorFlowTranslator", "UnsupportedCheckpointError"]
