"""Sanity tests for :class:`SoftCAMTransformerForPrediction`.

These tests are designed to run on Colab (or any environment with
``transformers`` and ``torch`` installed). They verify the contract of the
Evidence Layer:

- Shapes propagate correctly through forward and generate.
- ``M`` is non-negative and each row sums to 1 (softmax invariant).
- The loss is finite and decomposes as forecast + elastic + entropy.
- A single backward step does not produce ``NaN`` / ``Inf``.

Run from the repository root with::

    pytest code/tests/test_softcam_transformer.py -v
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gluonts.time_feature import (  # noqa: E402  — after importorskip
    get_lags_for_frequency,
    time_features_from_frequency_str,
)

from src.models.softcam_transformer import (  # noqa: E402
    SoftCAMTransformerConfig,
    SoftCAMTransformerForPrediction,
    SoftCAMTSPredictionOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FREQ = "1T"
PREDICTION_LENGTH = 12       # tiny for speed
CONTEXT_LENGTH = 24
D_MODEL = 16
N_SERIES = 3
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def lags():
    return get_lags_for_frequency(FREQ)


@pytest.fixture(scope="module")
def time_features():
    return time_features_from_frequency_str(FREQ)


@pytest.fixture
def config(lags, time_features) -> SoftCAMTransformerConfig:
    return SoftCAMTransformerConfig(
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        lags_sequence=lags,
        num_time_features=len(time_features) + 1,
        num_static_categorical_features=1,
        cardinality=[N_SERIES],
        embedding_dimension=[2],
        encoder_layers=2,
        decoder_layers=2,
        d_model=D_MODEL,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=D_MODEL,
        decoder_ffn_dim=D_MODEL,
        dropout=0.0,
        alpha_l1=1e-4,
        beta_l2=1e-3,
        gamma_entropy=1e-3,
    )


@pytest.fixture
def model(config) -> SoftCAMTransformerForPrediction:
    torch.manual_seed(0)
    return SoftCAMTransformerForPrediction(config).eval()


@pytest.fixture
def batch(config):
    """Build a minimal synthetic batch consistent with the config shapes."""
    past_len = config.context_length + max(config.lags_sequence)
    n_time_features = config.num_time_features
    torch.manual_seed(1)
    return {
        "past_values": torch.randn(BATCH_SIZE, past_len),
        "past_time_features": torch.randn(BATCH_SIZE, past_len, n_time_features),
        "past_observed_mask": torch.ones(BATCH_SIZE, past_len),
        "static_categorical_features": torch.randint(
            0, N_SERIES, (BATCH_SIZE, 1), dtype=torch.long
        ),
        "future_values": torch.randn(BATCH_SIZE, PREDICTION_LENGTH),
        "future_time_features": torch.randn(
            BATCH_SIZE, PREDICTION_LENGTH, n_time_features
        ),
        "future_observed_mask": torch.ones(BATCH_SIZE, PREDICTION_LENGTH),
    }


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_model_has_evidence_layer(model, config):
    assert hasattr(model, "evidence_linear"), "Evidence Layer missing"
    assert model.evidence_linear.in_features == config.d_model
    assert model.evidence_linear.out_features == config.context_length


def test_encoder_hook_registered(model):
    assert model._encoder_hook_handle is not None
    # An untouched model has no cache yet.
    assert model._current_enc_hidden is None


# ---------------------------------------------------------------------------
# Forward shapes
# ---------------------------------------------------------------------------


def test_forward_returns_softcam_output(model, batch):
    out = model(**batch)
    assert isinstance(out, SoftCAMTSPredictionOutput)


def test_forward_shapes(model, batch, config):
    out = model(**batch)
    assert out.evidence_map is not None
    assert out.evidence_map.shape == (
        BATCH_SIZE,
        config.prediction_length,
        config.context_length,
    )
    assert out.params is not None
    assert out.params.shape[:2] == (BATCH_SIZE, config.prediction_length)


def test_evidence_map_is_a_distribution(model, batch):
    out = model(**batch)
    M = out.evidence_map
    # Non-negative
    assert torch.all(M >= 0.0), "M must be non-negative after softmax"
    # Rows sum to one
    row_sums = M.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-5
    ), f"Each M[b, t] must sum to 1, got min={row_sums.min()}, max={row_sums.max()}"


# ---------------------------------------------------------------------------
# Loss decomposition
# ---------------------------------------------------------------------------


def test_loss_components_are_finite(model, batch):
    out = model(**batch)
    for name, t in [
        ("loss", out.loss),
        ("forecast_loss", out.forecast_loss),
        ("elastic_loss", out.elastic_loss),
        ("entropy_loss", out.entropy_loss),
    ]:
        assert t is not None, f"{name} is None"
        assert torch.isfinite(t), f"{name} is not finite: {t.item()}"


def test_total_loss_decomposes(model, batch):
    out = model(**batch)
    reconstructed = out.forecast_loss + out.elastic_loss + out.entropy_loss
    assert torch.allclose(out.loss, reconstructed, atol=1e-6), (
        "total loss must equal forecast + elastic + entropy"
    )


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


def test_backward_does_not_produce_nan(model, batch):
    out = model(**batch)
    out.loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_evidence_linear_receives_gradient(model, batch):
    out = model(**batch)
    out.loss.backward()
    grad = model.evidence_linear.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0.0, (
        "evidence_linear must receive a non-zero gradient"
    )


# ---------------------------------------------------------------------------
# Generate (inference path)
# ---------------------------------------------------------------------------


def test_generate_runs_with_evidence_layer(model, batch, config):
    """Autoregressive generation must work and use the Evidence Layer
    transparently. We don't assert anything on M here (generate doesn't
    return it), only that generation completes and returns the right shape."""
    # Set a small number of parallel samples for speed.
    model.config.num_parallel_samples = 5
    out = model.generate(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        future_time_features=batch["future_time_features"],
    )
    assert out.sequences.shape == (
        BATCH_SIZE,
        model.config.num_parallel_samples,
        config.prediction_length,
    )


# ---------------------------------------------------------------------------
# explain() helper
# ---------------------------------------------------------------------------


def test_explain_returns_M(model, batch, config):
    M = model.explain(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
        static_categorical_features=batch["static_categorical_features"],
    )
    assert M.shape == (
        BATCH_SIZE,
        config.prediction_length,
        config.context_length,
    )
    assert torch.allclose(M.sum(dim=-1), torch.ones_like(M.sum(dim=-1)), atol=1e-5)
