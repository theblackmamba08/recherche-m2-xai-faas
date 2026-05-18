"""SoftCAM-Transformer v3 — Fix #4 : LayerNorm sur h_evidence avant le mix.

Diagnostic Run B2 (2026-05-18) :
  - Spearman = 0.80 ≈ gate (bon rang) mais R² = -1.97 (mauvaise échelle).
  - Cause : ``h_evidence = bmm(M, enc_hidden)`` est une moyenne pondérée brute
    d'états encodeurs, sans normalisation. ``dec_output`` a traversé plusieurs
    couches Cross-Attention + FFN + LayerNorm. Mélanger les deux sans alignement
    donne des features d'entrée à ``parameter_projection`` avec une distribution
    décalée → mauvais paramètres Student-T → R² effondré.

Fix v3 :
  Ajouter ``nn.LayerNorm(d_model)`` appliqué à ``h_evidence`` **avant** le mix :

  .. math::
      h = (1 - \\text{mix}) \\cdot \\text{dec\\_output}
        + \\text{mix} \\cdot \\text{LayerNorm}(\\text{bmm}(M, \\text{enc\\_hidden}))

  ``parameter_projection`` reçoit ainsi des features normalisées dans les deux
  branches → l'hypothèse est que le R² redevient positif.

Tout le reste (config, schedules, regularisation, explain()) est hérité de v2.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.softcam_transformer_v2 import (
    SoftCAMTransformerV2Config,
    SoftCAMTransformerV2ForPrediction,
    SoftCAMV2TSPredictionOutput,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SoftCAMTransformerV3Config(SoftCAMTransformerV2Config):
    """Configuration de :class:`SoftCAMTransformerV3ForPrediction`.

    Identique à v2. ``model_type`` mis à jour pour distinguer les checkpoints.
    """

    model_type = "softcam_transformer_v3"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SoftCAMTransformerV3ForPrediction(SoftCAMTransformerV2ForPrediction):
    """v3 : v2 + LayerNorm sur h_evidence avant le mix.

    Un seul changement par rapport à v2 :

    - ``self.evidence_norm = nn.LayerNorm(d_model)`` ajouté dans ``__init__``.
    - ``output_params`` applique ``evidence_norm`` à ``h_evidence`` avant le
      calcul de ``h``.

    Tous les autres comportements (schedules mix/gamma via attributs mutables,
    regularisation, explain(), batch mismatch handling) sont hérités de v2.
    """

    config_class = SoftCAMTransformerV3Config

    def __init__(self, config: SoftCAMTransformerV3Config) -> None:
        super().__init__(config)
        self.evidence_norm = nn.LayerNorm(config.d_model)

    def output_params(self, dec_output):  # type: ignore[override]
        if not self.use_evidence_layer:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        if self._current_enc_hidden is None:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        h_evidence, M = self._compute_evidence_map(
            dec_output, self._current_enc_hidden
        )
        self._last_evidence_map = M

        h_evidence = self.evidence_norm(h_evidence)  # ← FIX #4

        mix = self.evidence_mix
        if mix == 1.0:
            h = h_evidence
        elif mix == 0.0:
            h = dec_output
        else:
            h = (1.0 - mix) * dec_output + mix * h_evidence

        return self.parameter_projection(h)
