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

## 2026-04-28 — Intégration baseline FAYAM + EDA

- Code FAYAM extrait de `transformer_m2-main.zip` → placé dans `code/src/baseline/fayam/` (référence immuable).
- Notebook DBSCAN copié dans `code/notebooks/Clustering_DBSCAN.ipynb`.
- Hyperparamètres Transformer identifiés : `prediction_length=120`, `context_length=240`, `freq="1T"`, `encoder_layers=4`, `decoder_layers=4`, `d_model=32`.
- Dataset disponible sur HuggingFace Hub (`FaalSa/dataME`) — chargement direct possible sans refaire le pipeline Azure.
- Discordance DBSCAN/HDBSCAN détectée (mémoire vs code) — à éclaircir avec l'encadreur.
- `code/notebooks/EDA_clusters.ipynb` créé : 39 cellules, 11 sections — analyse complète des 4 clusters **par fonction ET par cluster** (stats, séries, zéros, distributions, ACF/FFT, ADF, corrélations, comparaison inter-cluster, recommandations prétraitement).
- Suite → exécuter l'EDA, puis lancer `tsf_transf.py` sur les 4 clusters pour reproduire les métriques FAYAM.

## 2026-04-29 — Registre de résultats EDA

- `code/experiments/eda/REGISTER.md` créé — tableau cumulatif mis à jour automatiquement à chaque run du notebook.
- `EDA_clusters.ipynb` étendu : 39 → 47 cellules. Ajouts : cellule détection Colab/Drive (index 3), 4 cellules de capture métriques (overview, zéros, FFT, ADF), section 12 (sauvegarde JSON + REGISTER.md + `files.download()`).
- Suite → exécuter sur Colab, vérifier le JSON produit, puis lancer `tsf_transf.py`.

## 2026-04-29 — Finalisation EDA notebook (cellule rappel HTML)

- `EDA_clusters.ipynb` : 48 cellules — ajout d'une cellule markdown finale rappelant l'export HTML (Fichier → Télécharger → .html) après Run All sur Colab.
- Stratégie d'archivage : JSON métriques (automatique) + HTML figures (manuel) dans `code/experiments/eda/`.
- Suite → exécuter sur Colab, archiver HTML + JSON, puis lancer `tsf_transf.py`.

## 2026-04-29 — Documentation EDA complète

- `code/notebooks/README.md` créé : guide complet du notebook (données attendues, dépendances, exécution locale/Colab, structure du JSON produit, résultats attendus par cluster).
- `code/experiments/eda/REGISTER.md` enrichi : pointeurs vers JSON, HTML et README.
- `code/STEPS.md` mis à jour : 4 étapes Phase 1 cochées (code FAYAM, hyperparamètres, CSV, EDA), prochaine étape = lancer `tsf_transf.py`.
- `src/baseline/fayam/dataToHub.py` : token HuggingFace hardcodé redacté (`<HF_TOKEN_REDACTED>`) avant push GitHub.
