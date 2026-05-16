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
