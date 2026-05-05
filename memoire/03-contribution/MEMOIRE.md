# Mémoire — 03-contribution

## 2026-04-26

- Dossier créé. Structure prête à accueillir expériences XAI.

## 2026-04-27 — Refonte des hypothèses

- Pivot stratégique acté (cf. [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md)).
- Nouvelle priorité : **H1 = SoftCAM-Transformer** (interprétabilité intrinsèque par modification architecturale + ElasticNet sur evidence maps), inspirée de Djoumessi & Berens (2025) et transposée du CNN au TimeSeriesTransformer.
- Replis ordonnés : **H2 = TimeSHAP / KernelSHAP**, **H3 = attention weights + DBSCAN + faithfulness**.
- `STEPS.md` refondu en conséquence.
- Aucune expérience encore lancée — dépend de la reproduction de la baseline FAYAM (phase 1).

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
