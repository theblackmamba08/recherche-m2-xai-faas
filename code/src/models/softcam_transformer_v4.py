"""SoftCAM-Transformer v4 — Gating par produit (dot product).

Remplace le mélange additif de v3 :
    v3 : h_ev = (1 - mix) * dec_output + mix * LN(bmm(M, enc_hidden))
    v4 : gate = 1 + tanh(Linear(LN(bmm(M, enc_hidden))))
         h_ev = dec_output * gate

Justifications théoriques (retours encadreurs 2026-05-25) :
    - Vaswani et al. 2017   : M = softmax(Linear(dec_output)) = attention normalisée
    - Hu et al. 2018 SE-Net : gating par produit élément-par-élément sur les features
    - Dauphin et al. 2017 GLU : sélectivité par multiplication (gate ∈ (0,2))
    - Ba et al. 2016        : LayerNorm sur h_context pour stabilité numérique
    - Srivastava et al. 2015 : gate centré sur 1 = identité à l'init (Highway Networks)

Avantages sur v3 :
    - gate = 1 à l'init (std=0.01) → h_ev ≈ dec_output → checkpoint B5 préservé
    - gate ∈ (0, 2) → peut amplifier (>1) ou atténuer (<1) par dimension et par pas
    - plus de hyperparamètre evidence_mix à régler en inférence
    - modulation adaptative : chaque dimension D et chaque horizon t a son propre gate

Compatibilité checkpoint B5 :
    - Poids Transformer HF (partagés) : chargent avec strict=False.
    - evidence_linear + evidence_norm (v3) : ignorés à la lecture (absent de v4).
    - ev_layer_v4 (nouveau) : initialisé avec gate_proj ~ N(0, 0.01) → gate ≈ 1.
    - Résultat : les premières epochs de fine-tune partent du même point que B5.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from src.models.evidence_layer_v4 import EvidenceLayerV4
from src.models.softcam_transformer_v2 import SoftCAMV2TSPredictionOutput
from src.models.softcam_transformer_v3 import (
    SoftCAMTransformerV3Config,
    SoftCAMTransformerV3ForPrediction,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SoftCAMTransformerV4Config(SoftCAMTransformerV3Config):
    """Configuration de :class:`SoftCAMTransformerV4ForPrediction`.

    Identique à v3. ``evidence_mix`` est conservé pour rétrocompatibilité
    mais ignoré à l'inférence (le gate remplace le mix additif).
    """

    model_type = "softcam_transformer_v4"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SoftCAMTransformerV4ForPrediction(SoftCAMTransformerV3ForPrediction):
    """v4 : v3 + gating par produit à la place du mélange additif.

    Changements par rapport à v3 :

    - ``self.ev_layer_v4 = EvidenceLayerV4(d_model, context_length)``
      remplace ``evidence_linear + evidence_norm + mix``.
    - :meth:`output_params` calcule ``h_ev = dec_output ⊙ gate`` via
      ``ev_layer_v4`` et retourne ``parameter_projection(h_ev)``.
    - ``_M_override`` (H1.F / H1.G) : quand défini, court-circuite le
      calcul de M et recalcule le gate avec la M injectée via les
      sous-modules de ``ev_layer_v4``.
    - :meth:`gate_deviation` : métrique de contribution effective de M
      (0 si M uniforme, >0 si M concentrée).

    Hérité de v3 sans modification :
    - ``generate()`` Fix #5
    - ``predict_with_M_override()`` (faithfulness H1.F / H1.G)
    - ``explain()``
    - ``forward()`` avec décomposition de la loss
    - ``_regularization_terms()`` (ElasticNet + entropie sur M)
    - ``_encoder_forward_hook``
    """

    config_class = SoftCAMTransformerV4Config

    def __init__(self, config: SoftCAMTransformerV4Config) -> None:
        super().__init__(config)
        # Remplace evidence_linear + evidence_norm de v3
        self.ev_layer_v4 = EvidenceLayerV4(
            d_model=config.d_model,
            context_length=config.context_length,
        )

    # ---------------------------------------------------------- output_params

    def output_params(self, dec_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Gating par produit : h_ev = dec_output ⊙ (1 + tanh(gate_proj(h_ctx))).

        Si ``use_evidence_layer=False`` : comportement parent (identique FAYAM).
        Si ``_M_override`` est défini   : utilise la M injectée pour le gate
                                          (tests de fidélité H1.F / H1.G).
        """
        if not self.use_evidence_layer:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        if self._current_enc_hidden is None:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        enc_hidden = self._current_enc_hidden

        # Batch mismatch : generate() appelle output_params avec
        # batch_size × num_parallel_samples alors que enc_hidden a batch_size.
        if dec_output.shape[0] != enc_hidden.shape[0]:
            repeat_factor, remainder = divmod(
                dec_output.shape[0], enc_hidden.shape[0]
            )
            if remainder != 0:
                raise RuntimeError(
                    "Evidence Layer v4 batch dim mismatch: "
                    f"dec={dec_output.shape[0]}, enc={enc_hidden.shape[0]}."
                )
            enc_hidden = enc_hidden.repeat_interleave(repeat_factor, dim=0)

        if self._M_override is not None:
            # H1.F / H1.G : M arbitraire injectée par predict_with_M_override()
            # On recalcule h_context et gate avec la M fournie.
            # gate_strength miroite la valeur courante d'ev_layer_v4 (warm-up).
            M = self._M_override
            h_context = self.ev_layer_v4.layer_norm(torch.bmm(M, enc_hidden))
            gate = 1.0 + self.ev_layer_v4.gate_strength * torch.tanh(
                self.ev_layer_v4.gate_proj(h_context)
            )
            h_ev = dec_output * gate
        else:
            h_ev, M = self.ev_layer_v4(dec_output, enc_hidden)

        self._last_evidence_map = M
        return self.parameter_projection(h_ev)

    # --------------------------------------------------------- gate_deviation

    @torch.no_grad()
    def gate_deviation(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Déviation moyenne du gate par rapport à l'identité (1.0).

        Mesure la contribution effective de M :
        - ≈ 0 : M uniforme → gate ≈ 1 → v4 ≈ v3 avec mix→0
        - > 0  : M concentrée → gate module effectivement dec_output

        Utile pour diagnostiquer si le fine-tune à partir de B5 a bien
        appris un gate informatif.
        """
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
        h_context = self.ev_layer_v4.layer_norm(torch.bmm(M, enc_hidden))
        gate = 1.0 + self.ev_layer_v4.gate_strength * torch.tanh(
            self.ev_layer_v4.gate_proj(h_context)
        )
        return (gate - 1.0).abs().mean().item()
