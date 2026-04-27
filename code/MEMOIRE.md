# Mémoire — code

## 2026-04-26 — Setup initial

- Squelette créé : `src/` (avec `__init__.py`), `tests/`, `notebooks/`, `experiments/`.
- `pyproject.toml` : Python ≥3.10, deps numpy/pandas/sklearn/torch/transformers/matplotlib, extras `dev` (pytest, ruff, jupyter) et `xai` (captum).
- Aucun module métier encore importé (FAYAM à intégrer en phase 1).
- Suite → cf. [STEPS.md](STEPS.md) : importer le code FAYAM Transformer.

## 2026-04-27 — Refonte stratégique

- `STEPS.md` refondu après pivot stratégique : nouveau pipeline H1 SoftCAM-Transformer → H2 TimeSHAP → H3 attention.
- À ajouter dans `pyproject.toml` quand on entrera en phase 2/3 : `shap`, `timeshap` (extra `xai`).
- Le code FAYAM reste à importer (phase 1 inchangée dans son objectif, juste enrichie : activer `output_attentions=True` dès la repro).
