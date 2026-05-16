"""H1 — SoftCAM-Transformer.

Intrinsic XAI variants of HuggingFace's TimeSeriesTransformer for FaaS
workload forecasting.
"""

from .softcam_transformer import (
    SoftCAMTransformerConfig,
    SoftCAMTransformerForPrediction,
    SoftCAMTSPredictionOutput,
)

__all__ = [
    "SoftCAMTransformerConfig",
    "SoftCAMTransformerForPrediction",
    "SoftCAMTSPredictionOutput",
]
