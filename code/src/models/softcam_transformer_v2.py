"""SoftCAM-Transformer v2 — diagnostic-friendly variant of v1.

v1 replaced ``parameter_projection(dec_output)`` with
``parameter_projection(bmm(M, enc_hidden))``. The first Colab run
(2026-05-17) failed GATE H1.C with Test R²=-6.16 and Spearman=-0.87
(anti-correlation on all 5 functions of Cluster 4). The most likely
explanation is an **information bottleneck**: the decoder's processed
representation is *entirely discarded* and replaced by a linear
combination of *raw* encoder embeddings. ``parameter_projection`` —
initialized for decoder-output statistics — is then fed a tensor with
very different statistical properties, leading to degenerate forecasts.

v2 introduces two configuration knobs to *diagnose* this hypothesis
without rewriting the architecture:

1. ``use_evidence_layer: bool`` (default ``True``)
   When ``False``, the Evidence Layer is bypassed and ``output_params``
   falls back to the parent's standard ``parameter_projection(dec_output)``.
   This lets the *same notebook* run a sanity check ("does the parent
   converge in our setup?") and a v2 run side by side, varying a single
   flag.

2. ``evidence_mix: float in [0, 1]`` (default ``1.0``)
   When the Evidence Layer is active, the value fed to
   ``parameter_projection`` is

   .. math:: h = (1 - \\text{mix}) \\cdot \\text{dec\\_output}
       + \\text{mix} \\cdot \\text{bmm}(M, \\text{enc\\_hidden})

   - ``mix = 0.0``: identical to the parent (Evidence Layer computed
     for monitoring but does not affect the prediction).
   - ``mix = 1.0``: identical to v1 (decoder representation discarded).
   - ``0 < mix < 1``: hybrid — the decoder's information is preserved,
     and the Evidence Layer is an *additive* correction whose weight
     can be raised gradually.

The "fidelity by construction" argument needs slight revision when
``mix < 1``: ``M`` is no longer the *only* algebraic weight in the
prediction. Instead, ``M[t, s]`` is the **algebraic coefficient of the
encoder hidden state at past position s** in the prediction's residual
correction. This is still a faithful, exact decomposition of one
additive component of the forecast.

All other contracts (the encoder forward hook, the override of
``output_params``, the regularization terms, the test suite shapes)
are preserved from v1.

References
----------
- Djoumessi & Berens (2025), *SoftCAM*
- Vaswani et al. (2017), *Attention is All You Need*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.time_series_transformer.configuration_time_series_transformer import (
    TimeSeriesTransformerConfig,
)
from transformers.models.time_series_transformer.modeling_time_series_transformer import (
    Seq2SeqTSPredictionOutput,
    TimeSeriesTransformerForPrediction,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SoftCAMTransformerV2Config(TimeSeriesTransformerConfig):
    """Configuration for :class:`SoftCAMTransformerV2ForPrediction`.

    Adds two diagnostic knobs on top of v1's regularization hyperparameters:

    Parameters
    ----------
    use_evidence_layer
        Master switch. When ``False``, the model behaves exactly like the
        parent ``TimeSeriesTransformerForPrediction`` and the value of
        ``evidence_mix`` is ignored. Use this for the FAYAM-baseline
        sanity check inside the same notebook.
    evidence_mix
        Interpolation coefficient in ``[0.0, 1.0]``. Only consulted when
        ``use_evidence_layer=True``. ``0.0`` recovers the parent's
        behavior (Evidence Layer still computed for monitoring, but does
        not affect the prediction). ``1.0`` reproduces v1 (decoder
        output entirely replaced).
    alpha_l1, beta_l2, gamma_entropy
        Same as v1. Inert when ``use_evidence_layer=False``.
    """

    model_type = "softcam_transformer_v2"

    def __init__(
        self,
        use_evidence_layer: bool = True,
        evidence_mix: float = 1.0,
        alpha_l1: float = 0.0,
        beta_l2: float = 1e-3,
        gamma_entropy: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_evidence_layer = bool(use_evidence_layer)
        mix = float(evidence_mix)
        if not 0.0 <= mix <= 1.0:
            raise ValueError(
                f"evidence_mix must be in [0, 1], got {mix}."
            )
        self.evidence_mix = mix
        self.alpha_l1 = float(alpha_l1)
        self.beta_l2 = float(beta_l2)
        self.gamma_entropy = float(gamma_entropy)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class SoftCAMV2TSPredictionOutput(Seq2SeqTSPredictionOutput):
    """Output of :meth:`SoftCAMTransformerV2ForPrediction.forward`.

    Extends :class:`Seq2SeqTSPredictionOutput` with the evidence map ``M``
    (when computed) and a per-term breakdown of the loss for monitoring.

    ``M`` is ``None`` when ``use_evidence_layer=False`` or when the
    forward pass did not exercise ``output_params`` (e.g. ``future_values``
    not provided).
    """

    evidence_map: Optional[torch.FloatTensor] = None
    forecast_loss: Optional[torch.FloatTensor] = None
    elastic_loss: Optional[torch.FloatTensor] = None
    entropy_loss: Optional[torch.FloatTensor] = None
    evidence_mix: Optional[float] = None
    use_evidence_layer: Optional[bool] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


_ENTROPY_EPS: float = 1e-9


class SoftCAMTransformerV2ForPrediction(TimeSeriesTransformerForPrediction):
    """v2: diagnostic-friendly SoftCAM-Transformer.

    See module docstring for the rationale.

    The architecture differs from v1 only in :meth:`output_params`, which
    now supports two diagnostic modes:

    - ``use_evidence_layer=False`` → calls ``self.parameter_projection(dec_output)``
      verbatim. Forward hook still fires (encoder cache is populated) but
      the cache is never consumed. ``evidence_map`` in the output is
      ``None``.
    - ``use_evidence_layer=True`` and ``evidence_mix=mix`` →
      ``h = (1 - mix) * dec_output + mix * bmm(M, enc_hidden)``;
      ``params = parameter_projection(h)``. The evidence map ``M`` is
      stored and returned.
    """

    config_class = SoftCAMTransformerV2Config

    # ------------------------------------------------------------------ init

    def __init__(self, config: SoftCAMTransformerV2Config) -> None:
        super().__init__(config)

        self.evidence_linear = nn.Linear(
            in_features=config.d_model,
            out_features=config.context_length,
        )

        self.use_evidence_layer: bool = bool(getattr(config, "use_evidence_layer", True))
        self.evidence_mix: float = float(getattr(config, "evidence_mix", 1.0))
        self.alpha_l1: float = float(getattr(config, "alpha_l1", 0.0))
        self.beta_l2: float = float(getattr(config, "beta_l2", 0.0))
        self.gamma_entropy: float = float(getattr(config, "gamma_entropy", 0.0))

        self._current_enc_hidden: Optional[torch.Tensor] = None
        self._last_evidence_map: Optional[torch.Tensor] = None

        self._init_evidence_layer()

        self._encoder_hook_handle = self.model.encoder.register_forward_hook(
            self._encoder_forward_hook
        )

    # ----------------------------------------------------------------- utils

    def _init_evidence_layer(self) -> None:
        """Xavier-uniform init so that softmax(logits) ≈ uniform at iter 0."""
        nn.init.xavier_uniform_(self.evidence_linear.weight)
        if self.evidence_linear.bias is not None:
            nn.init.zeros_(self.evidence_linear.bias)

    def _encoder_forward_hook(
        self,
        module: nn.Module,  # noqa: ARG002
        inputs: Tuple[Any, ...],  # noqa: ARG002
        outputs: Any,
    ) -> None:
        if hasattr(outputs, "last_hidden_state"):
            self._current_enc_hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            self._current_enc_hidden = outputs[0]
        else:
            self._current_enc_hidden = outputs

    # ----------------------------------------------------------- evidence op

    def _compute_evidence_map(
        self,
        dec_output: torch.Tensor,
        enc_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h_evidence, M)`` for the current decoder output.

        ``h_evidence = bmm(M, enc_hidden)`` is the pure-evidence
        representation (used directly when ``evidence_mix=1.0``).
        """
        if dec_output.shape[0] != enc_hidden.shape[0]:
            repeat_factor, remainder = divmod(
                dec_output.shape[0], enc_hidden.shape[0]
            )
            if remainder != 0:
                raise RuntimeError(
                    "Evidence Layer batch dim mismatch: "
                    f"dec={dec_output.shape[0]}, enc={enc_hidden.shape[0]}; "
                    "expected dec to be an integer multiple of enc."
                )
            enc_hidden = enc_hidden.repeat_interleave(repeat_factor, dim=0)

        logits = self.evidence_linear(dec_output)            # (B, T, ctx)
        M = F.softmax(logits, dim=-1)                         # (B, T, ctx)
        h_evidence = torch.bmm(M, enc_hidden)                 # (B, T, d_model)
        return h_evidence, M

    # ---- override of the parent's output_params

    def output_params(self, dec_output: torch.Tensor) -> torch.Tensor:
        """Insert the Evidence Layer with optional residual mixing.

        Branches by ``self.use_evidence_layer`` and ``self.evidence_mix``:

        - ``use_evidence_layer=False``  → exactly the parent's behavior.
        - ``use_evidence_layer=True``   → ``h = (1 - mix) * dec_output
                                            + mix * bmm(M, enc_hidden)``;
                                          ``params = parameter_projection(h)``.

        ``self._last_evidence_map`` is populated only when the Evidence
        Layer is actually used (i.e. ``use_evidence_layer=True`` and the
        encoder hook has captured an encoder output).
        """
        if not self.use_evidence_layer:
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        if self._current_enc_hidden is None:
            # Defensive fallback (e.g. unit tests that bypass the encoder).
            self._last_evidence_map = None
            return self.parameter_projection(dec_output)

        h_evidence, M = self._compute_evidence_map(
            dec_output, self._current_enc_hidden
        )
        self._last_evidence_map = M

        mix = self.evidence_mix
        if mix == 1.0:
            h = h_evidence
        elif mix == 0.0:
            h = dec_output
        else:
            h = (1.0 - mix) * dec_output + mix * h_evidence

        return self.parameter_projection(h)

    # ------------------------------------------------------ regularizations

    @staticmethod
    def _row_entropy(M: torch.Tensor) -> torch.Tensor:
        """Mean row-wise Shannon entropy of ``M``."""
        log_M = torch.log(M.clamp_min(_ENTROPY_EPS))
        H = -(M * log_M).sum(dim=-1)
        return H.mean()

    def _regularization_terms(
        self,
        M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l1 = M.abs().mean()                     # constant under softmax
        l2 = (M ** 2).mean()
        elastic = self.alpha_l1 * l1 + self.beta_l2 * l2

        if self.gamma_entropy != 0.0:
            entropy = self.gamma_entropy * self._row_entropy(M)
        else:
            entropy = torch.zeros((), device=M.device, dtype=M.dtype)
        return elastic, entropy

    # --------------------------------------------------------------- forward

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # noqa: ARG002 — always True
    ) -> SoftCAMV2TSPredictionOutput:
        outputs = super().forward(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=True,
        )

        M: Optional[torch.Tensor] = self._last_evidence_map

        forecast_loss = outputs.loss
        elastic_loss: Optional[torch.Tensor] = None
        entropy_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = forecast_loss

        if forecast_loss is not None and M is not None:
            elastic_loss, entropy_loss = self._regularization_terms(M)
            total_loss = forecast_loss + elastic_loss + entropy_loss

        return SoftCAMV2TSPredictionOutput(
            loss=total_loss,
            params=outputs.params,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loc=outputs.loc,
            scale=outputs.scale,
            static_features=outputs.static_features,
            evidence_map=M,
            forecast_loss=forecast_loss,
            elastic_loss=elastic_loss,
            entropy_loss=entropy_loss,
            evidence_mix=self.evidence_mix if self.use_evidence_layer else None,
            use_evidence_layer=self.use_evidence_layer,
        )

    # ------------------------------------------------------- explain helper

    @torch.no_grad()
    def explain(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward returning the evidence map ``M``.

        Raises if called while ``use_evidence_layer=False`` — there is
        nothing to explain in that mode.
        """
        if not self.use_evidence_layer:
            raise RuntimeError(
                "explain() called on a model with use_evidence_layer=False; "
                "no evidence map is produced in that mode."
            )
        was_training = self.training
        self.eval()
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
            self.train(was_training)
        if outputs.evidence_map is None:
            raise RuntimeError(
                "evidence_map was not computed — check that future_values "
                "was provided and the forward pass executed the Evidence Layer."
            )
        return outputs.evidence_map
