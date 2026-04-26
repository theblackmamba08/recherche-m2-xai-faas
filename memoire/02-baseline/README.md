# 02-baseline — Reproduction du mémoire FAYAM

Avant d'apporter une contribution, on **reproduit** la baseline pour disposer d'un point de référence. Le code de reproduction vit dans [`../../code/`](../../code/) ; ce dossier contient les **résultats**, **observations**, et **écarts** mesurés.

## Contenu attendu

- `resultats/` — métriques reproduites (CSV / JSON)
- `figures/` — graphiques de comparaison
- `ecarts.md` — différences observées par rapport au mémoire FAYAM (et causes)
- `MEMOIRE.md` — historique
- `STEPS.md` — todo

## Périmètre

- **Inclus** : Modèle 2 de FAYAM (Transformer HuggingFace)
- **Exclus** : Modèle 1 (CNN-LSTM) — abandonné par décision méthodo
