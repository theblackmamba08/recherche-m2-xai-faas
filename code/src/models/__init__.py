"""H1 — SoftCAM-Transformer.

Intrinsic XAI variants of HuggingFace's TimeSeriesTransformer for FaaS
workload forecasting.

v1 — original (NO-GO on 2026-05-17, Test R²=-6.16).
v2 — diagnostic-friendly: ``use_evidence_layer`` toggle + ``evidence_mix``.
v3 — Fix #4: LayerNorm on h_evidence before mix (Run B2 R²=-1.97 → target ≥0.30).
v4 — gating par produit : h_ev = dec_output ⊙ (1 + tanh(gate_proj(LN(bmm(M, enc))))).
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
from .softcam_transformer_v4 import (
    SoftCAMTransformerV4Config,
    SoftCAMTransformerV4ForPrediction,
)
from .softcam_transformer_v5 import (
    SoftCAMTransformerV5Config,
    SoftCAMTransformerV5ForPrediction,
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
    "SoftCAMTransformerV4Config",
    "SoftCAMTransformerV4ForPrediction",
    "SoftCAMTransformerV5Config",
    "SoftCAMTransformerV5ForPrediction",
]
