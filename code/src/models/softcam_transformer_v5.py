"""SoftCAM-Transformer v5 — Produit borné entre carte d'évidence et décodeur.

Architecture :
    M           = softmax(score(dec, enc))                    # carte d'évidence
    h_evidence  = LayerNorm(bmm(M, enc_hidden))               # sortie de la carte
    h_ev        = dec_output * (1 + alpha * tanh(h_evidence)) # produit borné

Deux modes (paramètre ``m_mode`` du config) :
    - "mlp"          : M = softmax(Linear(dec_output))   (paramètres = 7920 sur D=32, C=240)
    - "dot_product"  : M = softmax(Q @ K^T / sqrt(D))    (paramètres = 2048)

Comparaison avec v3 et v4 :
    v3 : h_ev = (1-mix) * dec + mix * LN(bmm(M, enc))     additif, mix ∈ [0, 1]
    v4 : h_ev = dec ⊙ (1 + s * tanh(Linear(LN(bmm))))    multiplicatif, s ∈ [0, 1]
    v5 : h_ev = dec * (1 + α * tanh(LN(bmm(M, enc))))    multiplicatif borné

Différences cruciales v5 vs v4 (qui a échoué C1/C2/C3) :
    1. Plus de Linear sur h_context : on utilise directement la sortie de M.
    2. alpha typiquement petit (~0.1) au lieu de 1.0 → gate ∈ (0.9, 1.1).
    3. Peut combiner avec dot product pour M (mode="dot_product").
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from src.models.evidence_layer_v5 import EvidenceLayerV5
from src.models.softcam_transformer_v2 import SoftCAMV2TSPredictionOutput
from src.models.softcam_transformer_v3 import (
    SoftCAMTransformerV3Config,
    SoftCAMTransformerV3ForPrediction,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SoftCAMTransformerV5Config(SoftCAMTransformerV3Config):
    """Configuration de :class:`SoftCAMTransformerV5ForPrediction`.

    Ajoute ``m_mode`` (str, "mlp" ou "dot_product") par rapport à v3.
    ``evidence_mix`` est conservé pour rétrocompatibilité de config mais ignoré.
    """

    model_type = "softcam_transformer_v5"

    def __init__(self, m_mode: str = "dot_product", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.m_mode = m_mode


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SoftCAMTransformerV5ForPrediction(SoftCAMTransformerV3ForPrediction):
    """v5 : produit borné entre sortie de la carte d'évidence et dec_output.

    Hérité de v3 sans modification :
        - ``generate()`` Fix #5
        - ``predict_with_M_override()`` (H1.F / H1.G)
        - ``explain()``, ``forward()`` avec décomposition de la loss
        - ``_regularization_terms()`` (ElasticNet + entropie sur M)
        - ``_encoder_forward_hook``

    Override :
        - ``output_params`` : utilise ``ev_layer_v5`` pour h_ev = dec * (1 + α * tanh(h_evidence))
        - ``_M_override`` recalcule h_evidence et le produit avec la M injectée.
    """

    config_class = SoftCAMTransformerV5Config

    def __init__(self, config: SoftCAMTransformerV5Config) -> None:
        super().__init__(config)
        self.ev_layer_v5 = EvidenceLayerV5(
            d_model=config.d_model,
            context_length=config.context_length,
            m_mode=getattr(config, "m_mode", "dot_product"),
        )

    # ---------------------------------------------------------- output_params

    def output_params(self, dec_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Produit borné : h_ev = dec * (1 + α * tanh(LN(bmm(M, enc)))).

        Si ``use_evidence_layer=False`` ou enc_hidden absent : comportement parent.
        Si ``_M_override`` défini   : recalcule h_evidence avec la M injectée.
        """
        if not self.use_evidence_layer:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        if self._current_enc_hidden is None:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        enc_hidden = self._current_enc_hidden

        # Batch mismatch : generate() avec num_parallel_samples > 1.
        if dec_output.shape[0] != enc_hidden.shape[0]:
            repeat_factor, remainder = divmod(
                dec_output.shape[0], enc_hidden.shape[0]
            )
            if remainder != 0:
                raise RuntimeError(
                    "Evidence Layer v5 batch dim mismatch: "
                    f"dec={dec_output.shape[0]}, enc={enc_hidden.shape[0]}."
                )
            enc_hidden = enc_hidden.repeat_interleave(repeat_factor, dim=0)

        if self._M_override is not None:
            # H1.F / H1.G : M arbitraire, on recalcule le produit en aval.
            M = self._M_override
            h_evidence = self.ev_layer_v5.layer_norm(torch.bmm(M, enc_hidden))
            h_ev = dec_output * (
                1.0 + self.ev_layer_v5.alpha * torch.tanh(h_evidence)
            )
        else:
            h_ev, M = self.ev_layer_v5(dec_output, enc_hidden)

        self._last_evidence_map = M
        return self.parameter_projection(h_ev)

    # ------------------------------------------------------ modulation_strength

    @torch.no_grad()
    def modulation_strength(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Mesure |gate - 1| moyen. Borné par alpha."""
        self.eval()
        out = self.forward(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            future_values=future_values,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
        )
        M = out.evidence_map
        if M is None or self._current_enc_hidden is None:
            return 0.0
        enc_hidden = self._current_enc_hidden
        if M.shape[0] != enc_hidden.shape[0]:
            repeat_factor = M.shape[0] // enc_hidden.shape[0]
            enc_hidden = enc_hidden.repeat_interleave(repeat_factor, dim=0)
        h_evidence = self.ev_layer_v5.layer_norm(torch.bmm(M, enc_hidden))
        gate_minus_1 = self.ev_layer_v5.alpha * torch.tanh(h_evidence)
        return gate_minus_1.abs().mean().item()
