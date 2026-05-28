"""
Evidence Layer v5 — Produit entre carte d'évidence et décodeur.

Demande encadreurs (2026-05-28) :
    1. La carte M doit dire explicitement quels pas passés comptent pour la prédiction.
    2. La sortie de la carte d'évidence (= passé pondéré par M) doit MULTIPLIER dec_output.

Architecture :
    M           = softmax(score(dec_output, enc_hidden))         # [B, P, C]
    h_evidence  = LayerNorm(bmm(M, enc_hidden))                  # [B, P, D]
                  ↑ "sortie de la carte d'évidence"
    h_ev        = dec_output * (1 + alpha * tanh(h_evidence))    # produit borné

    alpha ∈ [0, 1] : amplitude du produit (réglable dynamiquement, warm-up obligatoire).
    tanh borne h_evidence dans [-1, 1] → gate dans (1-alpha, 1+alpha) → stable.

Deux variantes de calcul du score (paramètre ``m_mode``) :

    - "mlp"          : score[t, c] = MLP(dec_output[t])[c]
                       Paramètres : Linear(D → C) = D*C + C = 32*240+240 = 7920
                       Identique à v3/v4 : aveugle au contenu de enc_hidden[c].

    - "dot_product"  : score[t, c] = (dec_output[t] @ W_q) · (enc_hidden[c] @ W_k) / sqrt(D)
                       Paramètres : W_q + W_k = 2 * D*D = 2048
                       Attention Vaswani 2017 : compare RÉELLEMENT futur et passé.

Pourquoi un facteur alpha borné petit (≠ v4 qui avait gate_strength jusqu'à 1) :

    Les expériences C1/C2/C3 ont montré qu'un produit multiplicatif non borné
    rend l'autorégression instable (R²=-5). En limitant la modulation à
    (1 ± alpha) avec alpha petit (~0.1), on garde le signal dec_output
    quasi-intact tout en injectant l'information de M.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class EvidenceLayerV5(nn.Module):
    """
    Evidence Layer v5 — Produit borné entre carte d'évidence et décodeur.

    Args:
        d_model        : dimension des représentations (ex. 32).
        context_length : longueur du contexte encodeur (ex. 240).
        m_mode         : "mlp" (comme v3/v4) ou "dot_product" (attention Vaswani).
    """

    VALID_MODES = ("mlp", "dot_product")

    def __init__(
        self,
        d_model: int,
        context_length: int,
        m_mode: str = "dot_product",
    ) -> None:
        super().__init__()

        if m_mode not in self.VALID_MODES:
            raise ValueError(
                f"m_mode doit être dans {self.VALID_MODES}, reçu : {m_mode!r}"
            )

        self.d_model = d_model
        self.context_length = context_length
        self.m_mode = m_mode

        if m_mode == "mlp":
            # Identique v3 : M = softmax(Linear(dec_output))
            self.evidence_proj = nn.Linear(d_model, context_length)
        else:  # "dot_product"
            # Attention Vaswani 2017 : score = Q · K / sqrt(D)
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.scale = 1.0 / math.sqrt(d_model)
            # Init standard pour l'attention
            nn.init.xavier_uniform_(self.W_q.weight)
            nn.init.xavier_uniform_(self.W_k.weight)

        # Sortie de la carte d'évidence : passé pondéré, normalisé.
        self.layer_norm = nn.LayerNorm(d_model)

        # Amplitude du produit. 0 = identité (warm-up).
        # Réglable epoch par epoch depuis le notebook.
        self.alpha: float = 1.0

    # ----------------------------------------------------------------- compute_M

    def compute_M(
        self,
        dec_output: torch.Tensor,   # [B, P, D]
        enc_hidden: torch.Tensor,   # [B, C, D]
    ) -> torch.Tensor:
        """Calcule M selon le mode (mlp ou dot_product)."""
        if self.m_mode == "mlp":
            scores = self.evidence_proj(dec_output)            # [B, P, C]
        else:
            Q = self.W_q(dec_output)                            # [B, P, D]
            K = self.W_k(enc_hidden)                            # [B, C, D]
            scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, P, C]
        return torch.softmax(scores, dim=-1)

    # -------------------------------------------------------------------- forward

    def forward(
        self,
        dec_output: torch.Tensor,   # [B, P, D]
        enc_hidden: torch.Tensor,   # [B, C, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_ev : [B, P, D]  représentation enrichie → entrée de parameter_projection.
            M    : [B, P, C]  carte d'évidence → régularisation + visualisation.
        """
        M = self.compute_M(dec_output, enc_hidden)

        # Sortie de la carte d'évidence : combinaison pondérée du passé,
        # normalisée pour stabilité numérique.
        h_evidence = self.layer_norm(torch.bmm(M, enc_hidden))  # [B, P, D]

        # Produit borné. tanh garde h_evidence ∈ [-1, 1] → gate ∈ (1-α, 1+α).
        # alpha=0 → gate=1 → h_ev = dec_output (identité, warm-up).
        # alpha=0.1 → gate ∈ (0.9, 1.1) → modulation faible.
        h_ev = dec_output * (1.0 + self.alpha * torch.tanh(h_evidence))

        return h_ev, M

    # ---------------------------------------------------------- modulation_strength

    @torch.no_grad()
    def modulation_strength(
        self,
        dec_output: torch.Tensor,
        enc_hidden: torch.Tensor,
    ) -> float:
        """Mesure |gate - 1| moyen. 0 = pas de modulation, alpha-borné supérieurement."""
        _, M = self.forward(dec_output, enc_hidden)
        h_evidence = self.layer_norm(torch.bmm(M, enc_hidden))
        gate_minus_1 = self.alpha * torch.tanh(h_evidence)
        return gate_minus_1.abs().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Régularisation ElasticNet sur M (identique v3/v4)
# L_total = L_forecast + α‖M‖₁ + β‖M‖₂² + γH(M)
# ─────────────────────────────────────────────────────────────────────────────

def evidence_regularization_loss(
    M: torch.Tensor,
    alpha: float = 1e-4,
    beta: float = 1e-4,
    gamma: float = 1e-3,
) -> torch.Tensor:
    """Pénalise M pour la rendre sparse, stable et non-collapsée."""
    l1 = M.abs().mean()
    l2 = M.pow(2).mean()
    H = -(M * torch.log(M + 1e-8)).sum(dim=-1).mean()
    return alpha * l1 + beta * l2 + gamma * H
