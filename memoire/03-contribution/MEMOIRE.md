# Mémoire — 03-contribution

## 2026-04-26

- Dossier créé. Structure prête à accueillir expériences XAI.

## 2026-04-27 — Refonte des hypothèses

- Pivot stratégique acté (cf. [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md)).
- Nouvelle priorité : **H1 = SoftCAM-Transformer** (interprétabilité intrinsèque par modification architecturale + ElasticNet sur evidence maps), inspirée de Djoumessi & Berens (2025) et transposée du CNN au TimeSeriesTransformer.
- Replis ordonnés : **H2 = TimeSHAP / KernelSHAP**, **H3 = attention weights + DBSCAN + faithfulness**.
- `STEPS.md` refondu en conséquence.
- Aucune expérience encore lancée — dépend de la reproduction de la baseline FAYAM (phase 1).
