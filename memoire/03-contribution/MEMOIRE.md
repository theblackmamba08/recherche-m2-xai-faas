# Mémoire — 03-contribution

## 2026-04-26

- Dossier créé. Structure prête à accueillir expériences XAI.

## 2026-04-27 — Refonte des hypothèses

- Pivot stratégique acté (cf. [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md)).
- Nouvelle priorité : **H1 = SoftCAM-Transformer** (interprétabilité intrinsèque par modification architecturale + ElasticNet sur evidence maps), inspirée de Djoumessi & Berens (2025) et transposée du CNN au TimeSeriesTransformer.
- Replis ordonnés : **H2 = TimeSHAP / KernelSHAP**, **H3 = attention weights + DBSCAN + faithfulness**.
- `STEPS.md` refondu en conséquence.
- Aucune expérience encore lancée — dépend de la reproduction de la baseline FAYAM (phase 1).

## 2026-05-16 — Re-positionnement H1 vs TFT + 4 zooms architecturaux

### Re-positionnement de la motivation H1

**Point critique** : TFT (Temporal Fusion Transformer, Lim et al. 2019, IJF 2021) est un Transformer temporel intrinsèquement interprétable déjà publié et déployé en production. **Ne pas revendiquer** "premier Transformer TS interprétable" — TFT le démentit.

**Argument révisé de H1** : SoftCAM-Transformer est une **alternative à TFT** avec une propriété que TFT n'offre pas — la **fidélité par construction**.

| Critère | TFT (Lim 2019) | H1 SoftCAM-Transformer |
|---------|----------------|------------------------|
| Interprétabilité | Par attention (VSN + Interpretable MHA) | Par combinaison linéaire exacte |
| Fidélité garantie ? | Non — "attention ≠ explanation" (Jain & Wallace 2019) | Oui — M[t,s] est le poids RÉEL du calcul |
| Architecture modifiée ? | Oui (nouvelle archi TFT) | Minimalement (evidence layer + ElasticNet) |
| Backbone réutilisé | Non (nouvelle implémentation) | Oui (TimeSeriesTransformer HuggingFace) |
| Coût d'explication | 0 (intrinsèque) | 0 (intrinsèque) |

**Argument clé pour la soutenance** : TFT attribue l'importance par les poids d'attention — or Jain & Wallace (2019) ont montré qu'on peut permuter les poids d'attention sans changer la prédiction. H1 produit une explication dont chaque poids M[t,s] est algébriquement le coefficient du terme encoder_emb[s] dans le calcul de h[t] — c'est une décomposition linéaire exacte, pas une attribution approximative.

### Figures architecturales créées (Phase 2 — J3-J4)

4 SVG produits dans `figures/` :

- [`encoder-zoom.svg`](figures/encoder-zoom.svg) — zoom encodeur : 4 entrées avec shapes, embedding (ValueEmbed+TimeEmbed+StaticEmbed → (B,240,32)), couche Self-Attention détaillée (scores (B,2,240,240) → encoder_attentions), FFN, output (B,240,32)
- [`decoder-zoom.svg`](figures/decoder-zoom.svg) — zoom décodeur : Masked Self-Attn (causal) + Cross-Attention (Q=déc (B,2,120,16), K=V=enc (B,2,240,16), scores (B,2,120,240) → cross_attentions), output (B,120,32)
- [`evidence-layer-zoom.svg`](figures/evidence-layer-zoom.svg) — tête de sortie : comparaison FAYAM (Linear(32→3) opaque) vs H1 (Linear(32→240)+Softmax → M(B,120,240) → combinaison linéaire + ElasticNet)
- [`architecture-globale-revisee.svg`](figures/architecture-globale-revisee.svg) — vue globale mise à jour : toutes les shapes annotées, H1 en rouge, bypass encoder→evidence, légende comparatif TFT vs H1, hypothèses H1.A–H1.C

### Dimensions clés à maîtriser pour la soutenance

| Tenseur | Shape | Signification |
|---------|-------|---------------|
| past_values | (B, 422) | contexte 240 + max_lag 182 |
| past_time_features | (B, 422, 6) | 6 features cycliques + log-âge |
| encoder_input | (B, 240, 32) | après projection d_model |
| encoder_attentions | (B, 2, 240, 240) | self-attention encodeur (pour H3) |
| cross_attentions | (B, 2, 120, 240) | cross-attention décodeur (pour H3) |
| decoder_output | (B, 120, 32) | 120 représentations latentes |
| evidence_map M | (B, 120, 240) | **L'EXPLICATION H1** |
| forecasts | (B, 100, 120) | 100 trajectoires → médiane |

- Suite → implémenter `evidence_layer` dans `src/models/softcam_transformer.py` (Phase 2, J1-J2 du PLAN-ETUDE-ARCHITECTURE)

## 2026-05-17 — Run A (sanity check pipeline) : **FAIL — bug d'échelle**

**Run** : [`code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/`](../../code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/run.md).

### Résultats

| Métrique | Valeur | Δ vs FAYAM |
|----------|--------|------------|
| Test R² | **-0.1861** | -55.62 pp |
| Test Spearman | **0.9190** | -0.11 pp |

**Verdict : FAIL** — écart R² > 10 pp. Mais profil radicalement différent du NO-GO v1.

### Analyse différentielle v1 → Run A

| Signal | H1 v1 | Run A |
|--------|--------|-------|
| Spearman | -0.87 (anti-corrélation) | **+0.919** (quasi FAYAM) |
| R² | -6.16 | -0.19 |
| Cause probable | Information bottleneck (bypass décodeur) | Bug d'**échelle** (normalisation) |

→ `use_evidence_layer=False` a corrigé le problème d'ordre (Spearman retrouvé). Le résidu est uniquement un problème de magnitude des prédictions.

### Diagnostic (hypothèse principale)

**`inverse_transform` manquant ou mal appliqué** dans la cellule d'évaluation du notebook Run A. Spearman est invariant à l'échelle (corrélation de rang) ; R² est sensible à la variance résiduelle absolue. Si les prédictions sont dans l'espace normalisé et les actuals dans l'espace original (ou vice versa), R² s'effondre pendant que Spearman reste bon.

**Action** : comparer cellule par cellule `softcam-cluster4-v2-runA.ipynb` vs `baseline-cluster4.ipynb` — chercher `scaler.inverse_transform()` ou la normalisation interne HuggingFace `TimeSeriesTransformerForPrediction.forward()`.

### Hypothèses H1.A–H1.E (statut)

| Hypothèse | Statut après Run A |
|-----------|-------------------|
| H1.A (M se concentre sur pic 17h-19h) | Non testable (R² pas encore atteint) |
| H1.B (attention décodeur lags 1440/2880) | Non testable |
| **H1.C (R²≥0.30, Spearman≥0.85)** | **FAIL R²** — mais Spearman OK, problème identifié (échelle) |
| H1.D (cohérence intra-cluster des M) | Non testable |
| H1.E (best/worst function 953 vs 949) | Non testable |

- Suite → corriger le bug d'échelle dans l'évaluation, relancer Run A, vérifier PASS, puis Run B (mix=0.3).

## 2026-05-17 — Premier run H1 sur Colab : **NO-GO** sur GATE H1.C

**Run** : [`code/experiments/runs/2026-05-17_04-52_softcam-cluster4-h1-v1/`](../../code/experiments/runs/2026-05-17_04-52_softcam-cluster4-h1-v1/run.md) (archive locale, gitignored ; HTML + iframes conservés).

### Résultats

| Critère GATE H1.C | Seuil | Mesuré | Statut |
|-------------------|-------|--------|--------|
| Test R² | ≥ 0.30 | **-6.1565** | **FAIL** (-652 pp vs FAYAM=0.37) |
| Test Spearman | ≥ 0.85 | **-0.8731** | **FAIL** (-179 pp vs FAYAM=0.92) |

- best val R² = 0.0837 à epoch 8, puis dégradation → early stop epoch 18.
- Per-series Spearman ≈ -0.86 à -0.90 sur les 5 fonctions du Cluster 4 : **anti-corrélation systématique** (pas un bruit).
- `argmax_mean ≈ 155-160 / 240` sur toutes les fonctions → le modèle regarde **le milieu du passé**, pas le passé immédiat ni un pattern journalier cohérent.

### Diagnostic immédiat

Le **Spearman = -0.87** est le signal critique : les prédictions sont **mathématiquement inversées** par rapport à la réalité. Hypothèses (ordre de plausibilité) :

1. **Bug de sens dans `_evidence_layer()`** — signe inversé quelconque dans la chaîne `M = softmax(evidence_linear(dec_output)) → bmm(M, enc_hidden) → parameter_projection`.
2. **`encoder_last_hidden_state` mal capturé par le forward_hook** lors des appels multiples de `output_params` durant `model.generate()` (num_parallel_samples=100).
3. **Dégénérescence de M** : si toutes les lignes de M convergent vers le même vecteur (i.e. softmax constant), la combinaison linéaire revient à une moyenne fixe → constantes.

### Conséquence pour H1

**Pas de pivot précipité** — investigation obligatoire avant d'abandonner :

1. **Sanity check forward-only** : lancer un forward du parent `TimeSeriesTransformerForPrediction` (sans hook, sans evidence layer) sur exactement le même setup. Si lui converge sur Cluster 4 → le bug est dans notre code H1.
2. **Inspection visuelle de M** : si toutes les lignes sont identiques, la combinaison linéaire dégénère.
3. **Test unitaire fin** : vérifier que `model.explain(...)` rend bien le même M qu'un `forward()` standard, et que les valeurs ont le bon signe.

Si après ces 3 checks le bug n'est pas trouvé → **pivot H2 (TimeSHAP)** sans plus d'effort.

### Hypothèses opératoires H1.A-H1.E (statut)

| Hypothèse | Statut après v1 |
|-----------|-----------------|
| H1.A (M se concentre sur pic 17h-19h) | **Non testable** : modèle non convergent |
| H1.B (attention décodeur sur lags 1440/2880) | **Non testable** |
| **H1.C (non-dégradation R²≥0.30, Spearman≥0.85)** | **FAIL** (critère bloquant) |
| H1.D (cohérence intra-cluster des M) | Curieusement **OK** : tous les argmax ≈ 155-160 → cohérent mais inutile, le modèle est dégénéré |
| H1.E (best vs worst function) | Toutes échouent uniformément |

- Suite immédiate → audit du forward + ablation du forward_hook + comparaison enc_hidden cache vs recomputed.

## 2026-05-16 — Implémentation H1 dans `code/src/models/` (session 32)

- **Package créé** : `code/src/models/` avec `softcam_transformer.py`.
- **Approche** : sous-classement de `TimeSeriesTransformerForPrediction` HuggingFace (encodeur + décodeur **inchangés** → comparaison H1 vs FAYAM propre, une seule variable change).
- **Insertion de l'Evidence Layer** via 3 mécanismes :
  1. Module `evidence_linear = Linear(d_model → context_length)` (seule nouveauté).
  2. Hook `forward_hook` sur l'encodeur → cache automatique de `encoder_last_hidden_state`.
  3. Override de la méthode `output_params(dec_output)` → applique softmax + bmm. Fonctionne pour `forward` ET `generate` sans réécrire ce dernier.
- **Régularisation** : `α·mean(|M|) + β·mean(M²) + γ·H(M)`. Caveat L1 documenté : sous softmax, `mean(|M|) = 1/ctx` est constant (gradient nul). Le vrai terme sparsity-inducing est l'entropie de ligne `γ·H(M)`.
- **Notebook Colab** : `softcam-cluster4.ipynb` généré (33 cellules) — clone du repo, monitoring séparé des 3 composantes de loss, GATE H1.C explicite, extraction et heatmap des cartes M par fonction.
- **Tests pytest** : sanity checks de shape, somme=1, gradients finis, generate fonctionnel.
- Suite → push GitHub + lancer le notebook sur Colab T4 → vérifier le GATE H1.C (R²≥0.30, Spearman≥0.85).

## 2026-05-02 — Premier run baseline tracé

- Run tracker créé : [`code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/run.md`](../../code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/run.md)
- Notebook Colab généré : [`code/notebooks/baseline-fayam-transformer.ipynb`](../../code/notebooks/baseline-fayam-transformer.ipynb) (37 cellules — setup, pipeline GluonTS, entraînement avec checkpointing, métriques MASE/sMAPE/RMSE/R²/Spearman, extraction `output_attentions=True` post-training)
- Status du run : **pending** — à lancer sur Google Colab (T4 GPU)
- Diff des fichiers non commités capturé dans `run.md/diff.patch`

## 2026-05-05 — Cible H1 actée : Cluster 4

- Après 4 runs dédiés (C0/C4/C6/C8), **C4 est le seul cluster où le baseline FAYAM converge** (R²=0.37, Spearman=0.92, sMAPE=0.22). Cf. synthèse dans [`../02-baseline/MEMOIRE.md`](../02-baseline/MEMOIRE.md).
- C4 retenu comme **cible primaire de H1 (SoftCAM-Transformer)**. Décision et alternatives écartées documentées dans [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md) (entrée 2026-05-05).
- **Hypothèses opératoires H1 fixées** (à tester quand SoftCAM tournera) :
  - **H1.A** — l'evidence map se concentre sur les heures du profil journalier (pic 17h-19h, creux 2h-6h)
  - **H1.B** — l'attention décodeur se polarise sur les lags 1440 et 2880 (mémoire 24h/48h)
  - **H1.C** — SoftCAM ne dégrade pas la précision baseline (R²≥0.30, Spearman≥0.85 conservés)
  - **H1.D** — cohérence des evidence maps entre les 5 fonctions (cohérent avec Pearson intra-cluster > 0.95)
  - **H1.E** — test best/worst case : function 953 (R²=0.60) vs function 949 (R²=0.15)
- Suite → Phase 2 : J1 du `PLAN-ETUDE-ARCHITECTURE.md` (étude `TimeSeriesTransformer` HF, localisation projection finale décodeur).
