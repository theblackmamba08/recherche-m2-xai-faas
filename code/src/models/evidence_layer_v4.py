"""
Evidence Layer v4 — Gating par produit (dot product).

Remplace le mélange additif de v3 :
    v3 : h_ev = (1 - mix) * dec_output + mix * LN(bmm(M, enc_hidden))
    v4 : gate = 1 + tanh(Linear(LN(bmm(M, enc_hidden))))
         h_ev = dec_output * gate

Justifications théoriques :
    - Vaswani et al. 2017  : M = softmax(Linear(dec_output)) = mécanisme d'attention
    - Hu et al. 2018 (SE-Net) : gating par produit élément-par-élément sur les features
    - Dauphin et al. 2017 (GLU) : sélectivité par multiplication (gate ∈ (0,1) ou (0,2))
    - Ba et al. 2016 (LayerNorm) : stabilité numérique de h_context
    - Srivastava et al. 2015 (Highway) : gate centré sur 1 = identité à l'init → stable

Pourquoi 1 + tanh plutôt que sigmoid :
    - sigmoid(0) = 0.5 → h_ev ≈ 0.5 * dec_output à l'init → brise le checkpoint B5
    - 1 + tanh(0) = 1  → h_ev = dec_output à l'init → préserve le signal pré-entraîné
"""

import torch
import torch.nn as nn


class EvidenceLayerV4(nn.Module):
    """
    Evidence Layer v4 — Gating par produit.

    Args:
        d_model        : dimension des représentations (ex. 32)
        context_length : longueur du contexte encodeur (ex. 240)
    """

    def __init__(self, d_model: int, context_length: int):
        super().__init__()

        # M : projection dec_output → distribution sur le contexte passé
        # [B, P, D] → [B, P, C]
        self.evidence_proj = nn.Linear(d_model, context_length)

        # gate : projection h_context → modulation de dec_output
        # [B, P, D] → [B, P, D]
        self.gate_proj = nn.Linear(d_model, d_model)

        # Stabilité numérique (Ba et al. 2016)
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialisation petite pour gate ≈ 1 au départ
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        dec_output: torch.Tensor,   # [B, P, D]
        enc_hidden: torch.Tensor,   # [B, C, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_ev : [B, P, D]  représentation enrichie → entrée de output_projection
            M    : [B, P, C]  carte d'évidence → régularisation + visualisation
        """

        # ── Étape 1 : Carte d'évidence M ──────────────────────────────────
        # Chaque pas futur P distribue ses poids sur C instants passés
        # Justification : softmax = attention normalisée (Vaswani 2017)
        M = torch.softmax(self.evidence_proj(dec_output), dim=-1)
        # M : [B, P, C],  somme sur dim=-1 = 1, valeurs dans [0,1]

        # ── Étape 2 : Agrégation pondérée des états encodeur ──────────────
        # bmm : [B, P, C] × [B, C, D] → [B, P, D]
        # Chaque timestep futur reçoit une combinaison pondérée du contexte
        h_context = torch.bmm(M, enc_hidden)
        # h_context : [B, P, D]

        # ── Étape 3 : Normalisation (Ba et al. 2016) ──────────────────────
        h_context = self.layer_norm(h_context)

        # ── Étape 4 : Gate centré sur 1 ───────────────────────────────────
        # 1 + tanh(x) ∈ (0, 2), centré sur 1 quand x → 0
        # À l'init (gate_proj ≈ 0) : gate ≈ 1 → h_ev ≈ dec_output (identité)
        # Justification : residual gating — Highway Networks (Srivastava 2015)
        gate = 1.0 + torch.tanh(self.gate_proj(h_context))
        # gate : [B, P, D],  ∈ (0, 2)

        # ── Étape 5 : Produit élément-par-élément ─────────────────────────
        # h_ev = dec_output ⊙ gate
        # gate > 1 : amplification (timestep important selon M)
        # gate < 1 : atténuation  (timestep peu pertinent selon M)
        # gate = 1 : identité (cas M uniforme ou init)
        h_ev = dec_output * gate
        # h_ev : [B, P, D]

        return h_ev, M

    def gate_deviation(
        self,
        dec_output: torch.Tensor,
        enc_hidden: torch.Tensor,
    ) -> float:
        """
        Mesure la contribution effective de M :
        déviation moyenne du gate par rapport à l'identité (1.0).
        Si M est uniforme → déviation ≈ 0.
        Si M est concentrée → déviation > 0.
        """
        with torch.no_grad():
            _, M = self.forward(dec_output, enc_hidden)
            h_context = self.layer_norm(torch.bmm(M, enc_hidden))
            gate = 1.0 + torch.tanh(self.gate_proj(h_context))
            return (gate - 1.0).abs().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Régularisation ElasticNet sur M (identique v3)
# L_total = L_forecast + α‖M‖₁ + β‖M‖₂² + γH(M)
# ─────────────────────────────────────────────────────────────────────────────

def evidence_regularization_loss(
    M: torch.Tensor,
    alpha: float = 1e-4,   # poids L1  — sparsité (Zou & Hastie 2005)
    beta:  float = 1e-4,   # poids L2  — stabilité (Zou & Hastie 2005)
    gamma: float = 1e-3,   # poids H(M) — prévention collapse (entropie)
) -> torch.Tensor:
    """Pénalise M pour la rendre sparse, stable et non-collapsée."""
    l1 = M.abs().mean()
    l2 = M.pow(2).mean()
    H  = -(M * torch.log(M + 1e-8)).sum(dim=-1).mean()  # entropie de Shannon
    return alpha * l1 + beta * l2 + gamma * H
