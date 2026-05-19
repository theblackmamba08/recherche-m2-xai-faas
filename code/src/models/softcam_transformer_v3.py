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

Extension 2026-05-19 (faithfulness tests H1.F / H1.G) :
  Mécanisme ``_M_override`` permettant d'injecter une matrice M arbitraire dans
  ``output_params`` à la place de la M apprise. Sert aux tests
  *comprehensiveness* (masquer top-k entrées de M) et *sufficiency* (ne garder
  que top-k entrées de M).

Fix #5 (2026-05-19) — generate() court-circuitait output_params :
  HuggingFace ``TimeSeriesTransformerForPrediction.generate()`` appelle
  ``self.parameter_projection(...)`` directement dans la boucle de décodage
  autorégressif (ligne 1679 du source HF), court-circuitant notre override
  ``output_params`` et donc l'Evidence Layer complète.
  Conséquence : toutes les évaluations via ``generate()`` (runs A … B7)
  mesuraient un Transformer sans Evidence Layer à l'inférence.
  Fix : override ``generate()`` dans v3 avec la seule modification
  ``output_params(dec_last_hidden[:, -1:])`` à la place de
  ``parameter_projection(dec_last_hidden[:, -1:])``.

Tout le reste (config, schedules, regularisation, explain()) est hérité de v2.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.models.time_series_transformer.modeling_time_series_transformer import (
    SampleTSPredictionOutput,  # noqa: F401 — used by generate() override
)

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
    """v3 : v2 + LayerNorm sur h_evidence avant le mix + override de M.

    Changements par rapport à v2 :

    - ``self.evidence_norm = nn.LayerNorm(d_model)`` dans ``__init__``.
    - ``output_params`` applique ``evidence_norm`` à ``h_evidence`` avant
      le mix.
    - Attribut ``_M_override`` qui, lorsqu'il est défini, remplace la M
      apprise par une M arbitraire (utile pour H1.F / H1.G).
    - Méthode :meth:`predict_with_M_override` qui fait un forward
      teacher-forced en utilisant une M injectée.

    Tous les autres comportements (schedules mix/gamma, regularisation,
    explain(), batch mismatch handling) sont hérités de v2.
    """

    config_class = SoftCAMTransformerV3Config

    def __init__(self, config: SoftCAMTransformerV3Config) -> None:
        super().__init__(config)
        self.evidence_norm = nn.LayerNorm(config.d_model)
        self._M_override: Optional[torch.Tensor] = None

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

        # Faithfulness tests (H1.F / H1.G) : remplace M apprise par une
        # M arbitraire et recalcule h_evidence avec la nouvelle M.
        if self._M_override is not None:
            M = self._M_override
            enc_hidden = self._current_enc_hidden
            if dec_output.shape[0] != enc_hidden.shape[0]:
                repeat_factor, remainder = divmod(
                    dec_output.shape[0], enc_hidden.shape[0]
                )
                if remainder != 0:
                    raise RuntimeError(
                        "M_override batch dim mismatch: "
                        f"dec={dec_output.shape[0]}, enc={enc_hidden.shape[0]}."
                    )
                enc_hidden = enc_hidden.repeat_interleave(repeat_factor, dim=0)
            h_evidence = torch.bmm(M, enc_hidden)

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

    # --------------------------------------------------- faithfulness helper

    # -------------------------------------------------- generate() Fix #5
    # HuggingFace TimeSeriesTransformerForPrediction.generate() appelle
    # self.parameter_projection(...) directement dans la boucle autorégresssive,
    # court-circuitant notre override output_params() et donc l'Evidence Layer.
    # On copie la boucle à l'identique et on remplace l'unique appel fautif.

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SampleTSPredictionOutput:
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_features.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []

        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )
            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            dec_output = decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=repeated_enc_last_hidden,
            )
            dec_last_hidden = dec_output.last_hidden_state

            # FIX #5 : output_params() à la place de parameter_projection()
            # pour que l'Evidence Layer soit active pendant generate().
            # _current_enc_hidden est déjà peuplé (batch_size) par le hook
            # lors du self() initial ; _compute_evidence_map gère le repeat
            # (batch_size × num_parallel_samples vs batch_size).
            params = self.output_params(dec_last_hidden[:, -1:])
            distr = self.output_distribution(
                params, loc=repeated_loc, scale=repeated_scale
            )
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (
                    repeated_past_values,
                    (next_sample - repeated_loc) / repeated_scale,
                ),
                dim=1,
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)
        return SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length)
                + self.target_shape,
            )
        )

    # --------------------------------------------------- faithfulness helper

    @torch.no_grad()
    def predict_with_M_override(
        self,
        M_override: torch.Tensor,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
    ) -> SoftCAMV2TSPredictionOutput:
        """Forward teacher-forced avec une M injectée à la place de la M apprise.

        Sert aux tests de comprehensiveness (H1.F) et sufficiency (H1.G) :
        l'appelant fournit une M modifiée (top-k masquée ou top-k conservée)
        et observe l'impact sur la prédiction.

        ``M_override`` doit avoir la forme ``(B, prediction_length, context_length)``
        et ses lignes doivent sommer à 1 (lignes softmax-like).
        """
        if not self.use_evidence_layer:
            raise RuntimeError(
                "predict_with_M_override() requires use_evidence_layer=True."
            )
        was_training = self.training
        self.eval()
        self._M_override = M_override
        try:
            outputs = self.forward(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                static_categorical_features=static_categorical_features,
                future_values=future_values,
                future_time_features=future_time_features,
                future_observed_mask=future_observed_mask,
            )
        finally:
            self._M_override = None
            self.train(was_training)
        return outputs
