"""SoftCAM-Transformer — intrinsic XAI variant of HuggingFace's TimeSeriesTransformer.

Replaces the final ``parameter_projection`` of HuggingFace's
``TimeSeriesTransformerForPrediction`` with an *Evidence Layer* that produces,
for every batched window and every future step ``t``, a non-negative weight
vector ``M[t, :]`` over the ``context_length`` past positions, summing to 1.

The prediction is then computed as a linear combination of the encoder's last
hidden states weighted by ``M``. Because ``M`` is *literally* the algebraic
weight used in the forward computation (not a post-hoc estimate), the
explanation it carries is **faithful by construction**.

Inspired by
-----------
- Djoumessi & Berens (2025), *SoftCAM: Soft Class Activation Maps*
- Vaswani et al. (2017), *Attention is All You Need* (the structural skeleton
  of the Evidence Layer is a single-head attention without the Q·Kᵀ detour)

Hypothesis H1 — M2 thesis (Cabrel, 2026)
----------------------------------------
- **H1.A** : ``M`` should highlight peak/trough hours of the FaaS load.
- **H1.B** : the most important past lags (e.g. 1440 / 2880 min) should be
  captured by ``M`` and by ``cross_attentions``.
- **H1.C** : ``R² ≥ 0.30`` and ``Spearman ≥ 0.85`` should be preserved when
  the Evidence Layer replaces the direct projection of FAYAM (non-degradation
  gate).

Design choices
--------------
- We *subclass* ``TimeSeriesTransformerForPrediction`` rather than rewriting
  the encoder / decoder. The encoder and decoder are inherited unchanged, so
  comparing H1 vs FAYAM only varies a single component.
- We register a ``forward_hook`` on the encoder to cache its last hidden state
  as a private attribute. This is the only state shared between methods. It
  lets us override ``output_params(dec_output)`` (whose signature does not
  expose the encoder output) without touching ``forward`` / ``generate``.
- ``output_params`` is the single point of intervention: both training
  (``forward``) and autoregressive inference (``generate``) call it. Once it
  is overridden, the Evidence Layer is active in both paths.
- The regularization on ``M`` is added to the loss inside our ``forward``.

Caveat on the L1 term
---------------------
After ``Softmax(dim=-1)``, every row of ``M`` sums to one and is non-negative,
so ``mean(|M|) = 1 / context_length`` is **constant** and contributes zero
gradient. We expose ``alpha_l1`` for parity with the SoftCAM paper and the
SVG diagram, but the actual sparsity-inducing term in our setup is the
optional **entropy** regularization ``gamma_entropy``: minimizing the
Shannon entropy of each row pushes ``M`` toward peaked distributions.
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


class SoftCAMTransformerConfig(TimeSeriesTransformerConfig):
    """Configuration class for :class:`SoftCAMTransformerForPrediction`.

    Extends :class:`TimeSeriesTransformerConfig` with three regularization
    hyperparameters acting on the evidence map ``M``:

    Parameters
    ----------
    alpha_l1
        Weight of the L1 term. Kept for parity with the SoftCAM paper.
        Because every row of ``M`` sums to 1 after the softmax, ``mean(|M|)``
        is constant w.r.t. the model weights and this term has zero gradient.
        Set ``alpha_l1=0.0`` (default) to make the inertness explicit.
    beta_l2
        Weight of the L2 term ``mean(M ** 2)``. Minimized when each row of
        ``M`` is uniform — i.e., promotes *smoothing* (the row is spread over
        many past positions rather than concentrated on one).
    gamma_entropy
        Weight of the row-wise Shannon entropy term. Set ``> 0`` to encourage
        the model to *concentrate* its evidence on a few positions
        (sparsity). The entropy of a row is ``-Σ_s M[t, s] log M[t, s]``.
        Adding ``+ gamma * H(M)`` to the loss pushes entropy down and
        therefore induces sparse, peaked rows.
    """

    model_type = "softcam_transformer"

    def __init__(
        self,
        alpha_l1: float = 0.0,
        beta_l2: float = 1e-3,
        gamma_entropy: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha_l1 = float(alpha_l1)
        self.beta_l2 = float(beta_l2)
        self.gamma_entropy = float(gamma_entropy)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class SoftCAMTSPredictionOutput(Seq2SeqTSPredictionOutput):
    """Output of :meth:`SoftCAMTransformerForPrediction.forward`.

    Extends :class:`Seq2SeqTSPredictionOutput` with the evidence map ``M`` and
    a per-term breakdown of the loss for monitoring.

    Attributes
    ----------
    evidence_map
        Tensor of shape ``(B, prediction_length, context_length)``.
        ``M[b, t, s]`` is the non-negative weight applied to encoder position
        ``s`` of batch element ``b`` when predicting future step ``t``.
        ``Σ_s M[b, t, s] == 1``. **This IS the explanation.**
    forecast_loss
        Scalar tensor — the negative log-likelihood term of the loss.
    elastic_loss
        Scalar tensor — the L1 + L2 regularization term on ``M``.
    entropy_loss
        Scalar tensor — the row-wise Shannon entropy term on ``M``.
    """

    evidence_map: Optional[torch.FloatTensor] = None
    forecast_loss: Optional[torch.FloatTensor] = None
    elastic_loss: Optional[torch.FloatTensor] = None
    entropy_loss: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


# Numerical safety floor for log(M) when computing entropy. Without it,
# zero-mass positions would yield -inf. 1e-9 keeps the gradient stable.
_ENTROPY_EPS: float = 1e-9


class SoftCAMTransformerForPrediction(TimeSeriesTransformerForPrediction):
    """TimeSeriesTransformer with an Evidence Layer for intrinsic XAI.

    The forward computation is identical to
    :class:`TimeSeriesTransformerForPrediction` *up to and including*
    the decoder's last hidden state. From there:

    .. code-block::

        decoder_last_hidden_state          (B, prediction_length, d_model)
            │
            ▼  Linear(d_model → context_length)
        evidence_logits                    (B, prediction_length, context_length)
            │
            ▼  Softmax(dim=-1)             rows sum to 1
        M  (evidence_map)                  (B, prediction_length, context_length)
            │
            ▼  bmm with encoder_last_hidden_state  (B, context_length, d_model)
        h                                  (B, prediction_length, d_model)
            │
            ▼  parameter_projection (reused, unchanged)
        params                             (B, prediction_length, num_distr_params)
            │
            ▼  StudentT.output_distribution → sampling
        prediction

    During training the loss is

    .. math:: \\mathcal L = \\mathcal L_\\mathrm{forecast}
        + \\alpha \\, \\overline{|M|}
        + \\beta \\, \\overline{M^2}
        + \\gamma \\, \\overline{H(M)}

    where ``H(M)`` is the row-wise Shannon entropy of ``M``.

    Notes
    -----
    - The encoder and decoder modules from the parent class are *not* touched.
    - A forward hook on the encoder caches ``encoder_last_hidden_state``
      automatically, so :meth:`output_params` always finds it (whether
      invoked from :meth:`forward` or :meth:`generate`).
    - During autoregressive generation, HuggingFace repeats the batch by
      ``num_parallel_samples``. We detect the mismatch in batch dimension and
      ``repeat_interleave`` the cached encoder hidden state on the fly.
    """

    config_class = SoftCAMTransformerConfig

    # ------------------------------------------------------------------ init

    def __init__(self, config: SoftCAMTransformerConfig) -> None:
        super().__init__(config)

        # The single new sub-module of the architecture.
        self.evidence_linear = nn.Linear(
            in_features=config.d_model,
            out_features=config.context_length,
        )

        # Pull regularization hyperparameters out of config for fast access.
        self.alpha_l1: float = float(getattr(config, "alpha_l1", 0.0))
        self.beta_l2: float = float(getattr(config, "beta_l2", 0.0))
        self.gamma_entropy: float = float(getattr(config, "gamma_entropy", 0.0))

        # Volatile per-call caches (NOT registered as parameters / buffers).
        # They are populated by a forward hook on the encoder and consumed by
        # the overridden output_params.
        self._current_enc_hidden: Optional[torch.Tensor] = None
        self._last_evidence_map: Optional[torch.Tensor] = None

        self._init_evidence_layer()

        # Register the encoder hook. Stored so it can be removed in tests.
        self._encoder_hook_handle = self.model.encoder.register_forward_hook(
            self._encoder_forward_hook
        )

    # ----------------------------------------------------------------- utils

    def _init_evidence_layer(self) -> None:
        """Xavier-uniform init for stable softmax outputs at iteration 0.

        With Xavier-uniform on ``Linear(d_model, context_length)`` and zero
        bias, the initial logits are small in magnitude, so the softmax
        output starts near-uniform (≈ ``1 / context_length``). This avoids a
        runaway "winner-take-all" at the very first batch and lets the
        forecast loss drive learning.
        """
        nn.init.xavier_uniform_(self.evidence_linear.weight)
        if self.evidence_linear.bias is not None:
            nn.init.zeros_(self.evidence_linear.bias)

    def _encoder_forward_hook(
        self,
        module: nn.Module,  # noqa: ARG002 — required by the hook signature
        inputs: Tuple[Any, ...],  # noqa: ARG002
        outputs: Any,
    ) -> None:
        """Cache ``encoder.last_hidden_state`` automatically.

        Triggered each time the encoder runs (once per ``forward`` /
        ``generate`` call), so the cache is always consistent with the most
        recent encoder pass.
        """
        if hasattr(outputs, "last_hidden_state"):
            self._current_enc_hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            self._current_enc_hidden = outputs[0]
        else:
            self._current_enc_hidden = outputs

    # ----------------------------------------------------------- evidence op

    def _evidence_layer(
        self,
        dec_output: torch.Tensor,
        enc_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``params`` and the evidence map ``M``.

        Parameters
        ----------
        dec_output
            Decoder hidden state, shape ``(B_dec, T, d_model)``.
        enc_hidden
            Encoder hidden state, shape ``(B_enc, context_length, d_model)``.
            During autoregressive generation, ``B_dec = B_enc * num_parallel_samples``.

        Returns
        -------
        params
            Distribution parameters, shape ``(B_dec, T, num_distr_params)``.
        M
            Evidence map, shape ``(B_dec, T, context_length)``. Each row sums
            to one.
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
        h = torch.bmm(M, enc_hidden)                          # (B, T, d_model)
        params = self.parameter_projection(h)                 # (B, T, k)
        return params, M

    # ---- override of the parent's output_params (single point of insertion)

    def output_params(self, dec_output: torch.Tensor) -> torch.Tensor:
        """Insert the Evidence Layer before ``parameter_projection``.

        Falls back to the parent's behaviour if no encoder cache is
        available — this is a defensive guard that should not be exercised
        in normal flow because the encoder hook always populates the cache
        before this method is invoked.
        """
        if self._current_enc_hidden is None:
            return self.parameter_projection(dec_output)

        params, M = self._evidence_layer(dec_output, self._current_enc_hidden)
        self._last_evidence_map = M
        return params

    # ------------------------------------------------------ regularizations

    @staticmethod
    def _row_entropy(M: torch.Tensor) -> torch.Tensor:
        """Mean row-wise Shannon entropy of ``M``.

        ``M`` has shape ``(B, T, ctx)`` with non-negative entries summing to
        one along the last axis. Returns a scalar tensor.
        """
        log_M = torch.log(M.clamp_min(_ENTROPY_EPS))
        # row entropies: (B, T)
        H = -(M * log_M).sum(dim=-1)
        return H.mean()

    def _regularization_terms(
        self,
        M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the (elastic, entropy) loss components from ``M``."""
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
        return_dict: Optional[bool] = None,
    ) -> SoftCAMTSPredictionOutput:
        """Forward pass with Evidence Layer + ElasticNet/entropy regularization.

        The signature is intentionally identical to the parent's
        :meth:`TimeSeriesTransformerForPrediction.forward` so that existing
        training loops (e.g. the FAYAM notebooks) work unchanged.
        """
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

        # The encoder hook has populated _current_enc_hidden and our
        # overridden output_params has populated _last_evidence_map (the
        # latter is only set when params is computed, which happens only when
        # future_values is provided — i.e., during training / teacher forcing).
        M: Optional[torch.Tensor] = self._last_evidence_map

        forecast_loss = outputs.loss
        elastic_loss: Optional[torch.Tensor] = None
        entropy_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = forecast_loss

        if forecast_loss is not None and M is not None:
            elastic_loss, entropy_loss = self._regularization_terms(M)
            total_loss = forecast_loss + elastic_loss + entropy_loss

        return SoftCAMTSPredictionOutput(
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
        """Run a teacher-forced forward pass and return the evidence map ``M``.

        Use this to obtain ``M (B, prediction_length, context_length)`` on the
        test set for visualization. The model is set to ``eval()`` and no
        gradients are computed.

        Returns
        -------
        M
            Evidence map tensor on the same device as ``past_values``.
        """
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
