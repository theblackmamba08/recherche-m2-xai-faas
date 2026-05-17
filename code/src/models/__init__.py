"""H1 — SoftCAM-Transformer.

Intrinsic XAI variants of HuggingFace's TimeSeriesTransformer for FaaS
workload forecasting.

v1 — original (NO-GO on 2026-05-17, Test R²=-6.16).
v2 — diagnostic-friendly: ``use_evidence_layer`` toggle + ``evidence_mix``.
"""

from .softcam_transformer import (
    SoftCAMTransformerConfig,
    SoftCAMTransformerForPrediction,
    SoftCAMTSPredictionOutput,
)
from .softcam_transformer_v2 import (
    SoftCAMTransformerV2Config,
    SoftCAMTransformerV2ForPrediction,
    SoftCAMV2TSPredictionOutput,
)

__all__ = [
    "SoftCAMTransformerConfig",
    "SoftCAMTransformerForPrediction",
    "SoftCAMTSPredictionOutput",
    "SoftCAMTransformerV2Config",
    "SoftCAMTransformerV2ForPrediction",
    "SoftCAMV2TSPredictionOutput",
]
