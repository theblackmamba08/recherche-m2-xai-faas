"""H1 — SoftCAM-Transformer.

Intrinsic XAI variants of HuggingFace's TimeSeriesTransformer for FaaS
workload forecasting.

v1 — original (NO-GO on 2026-05-17, Test R²=-6.16).
v2 — diagnostic-friendly: ``use_evidence_layer`` toggle + ``evidence_mix``.
v3 — Fix #4: LayerNorm on h_evidence before mix (Run B2 R²=-1.97 → target ≥0.30).
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
from .softcam_transformer_v3 import (
    SoftCAMTransformerV3Config,
    SoftCAMTransformerV3ForPrediction,
)

__all__ = [
    "SoftCAMTransformerConfig",
    "SoftCAMTransformerForPrediction",
    "SoftCAMTSPredictionOutput",
    "SoftCAMTransformerV2Config",
    "SoftCAMTransformerV2ForPrediction",
    "SoftCAMV2TSPredictionOutput",
    "SoftCAMTransformerV3Config",
    "SoftCAMTransformerV3ForPrediction",
]
