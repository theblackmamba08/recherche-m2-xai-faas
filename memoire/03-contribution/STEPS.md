# Steps — 03-contribution

> Refonte 2026-04-27 : nouvelle priorisation H1 (SoftCAM-Transformer) → H2 (TimeSHAP) → H3 (attention).

## Phase H1 — SoftCAM-Transformer (priorité)

### Préparation conceptuelle
- [ ] Récupérer le code de référence SoftCAM (`https://anonymous.4open.science/r/SoftCAM-E1A3/`) — vérifier accessibilité et licence.
- [ ] Cartographier l'architecture du `TimeSeriesTransformer` HuggingFace : localiser la projection finale du décodeur (cible de la modification).
- [ ] **Décision de design (à arbitrer avec encadreurs)** : forme de l'*evidence map* temporelle.
  - Option A : `[predictionLength × contextLength]` — contribution de chaque pas passé à chaque pas futur.
  - Option B : par cluster DBSCAN — carte « profil de charge ».
  - Option C : par seuil de charge (faible/moyen/fort).

### Implémentation
- [ ] `../../code/src/models/softcam_transformer.py` : variante du `TimeSeriesTransformer` avec couche d'évidence + perte ElasticNet.
- [ ] `../../code/src/training/elasticnet_loss.py` : `L = MSE/NLL + λ₁·||A||₁ + λ₂·||A||₂²`.
- [ ] Recherche `λ₁` / `λ₂` par grille sur validation set (s'inspirer des ordres de grandeur SoftCAM : `[10⁻⁶, 10⁻³]`).

### Évaluation
- [ ] Performance prédictive (sMAPE, RMSE, R², Spearman) — *ne doit pas dégrader vs FAYAM*.
- [ ] Qualité d'explication : faithfulness/comprehensiveness/sufficiency adaptés au temporel + activation precision/sensitivity à redéfinir pour le temporel.
- [ ] Comparer carte d'évidence apprise vs matrice d'attention (sanity check).

### Critère de bascule vers H2
- [ ] Si pas de prototype convergent à fin S6 → basculer sur H2 sans attendre.

## Phase H2 — SHAP-based (repli, S7-S8 si H1 bloque)

- [ ] Étudier TimeSHAP (Bento et al.) — installation, signature `forward()` requise.
- [ ] `../../code/src/xai/timeshap_runner.py` : application au Transformer FAYAM (post-hoc).
- [ ] Comparaison KernelSHAP / TimeSHAP sur ≥ 3 datasets.
- [ ] Évaluation par la même grille faithfulness que H1 (comparabilité).

## Phase H3 — Attention weights (dernier recours OU outil de validation de H1)

- [ ] Exploiter les attention maps déjà extraites en phase 1 (si `output_attentions=True` activé).
- [ ] DBSCAN sur les charges → 3-5 macro-profils.
- [ ] Visualiser attention moyenne par profil.
- [ ] Calculer comprehensiveness + sufficiency par profil.
- [ ] **Si H1 ou H2 abouti** : H3 = sanity check, pas contribution principale.
