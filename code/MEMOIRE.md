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

## 2026-04-29 — Correction bug accents + exécution Colab

- Bug `KeyError: 'Zeros (%)'` corrigé dans les cellules de capture (accents manquants : `'Zeros (%)'` → `'Zéros (%)'`, `'Zeros consecutifs max (h)'` → `'Zéros consécutifs max (h)'`).
- Fix commité et poussé (commit `a113c93`).
- Notebook en cours d'exécution sur Colab avec datasets depuis Google Drive (`/content/drive/MyDrive/Recherche/Datasets`).
- Bug récurrent sur Colab : cellule capture zéros avait encore les colonnes sans accents (`Zeros (%)`) car l'ancienne version du notebook était uploadée. Correction : re-télécharger `EDA_clusters.ipynb` depuis GitHub avant tout upload Colab.

## 2026-05-02 — Premier run EDA complet

- Run Colab exécuté avec succès (2026-05-02_08-31). Résultats archivés : `eda_results_2026-05-02_08-31.json` + HTML dans `code/experiments/eda/`.
- `REGISTER.md` mis à jour manuellement (la cellule Drive avait écrit sur Drive mais pas en local).
- Résultats clés : 19/19 fonctions stationnaires (ADF p≈0), période dominante 24h universelle, moyennes C0=121 937 / C4=97 / C6=2 / C8=5 inv/min.
- HTML renommé : `EDA_clusters.ipynb - Colab.html` → `EDA_clusters_2026-05-02.html` (dossier `_files` renommé en cohérence).
- `memoire/02-baseline/EDA_RAPPORT.md` rédigé et poussé (commit `f0f9e73`) — 7 sections avec chiffres réels.

## 2026-05-02 — Refactoring DATA_DIR adaptatif (commit 06b7c77)

- Comparaison notebook Colab (49 cellules, avec outputs) vs local (48 cellules, propre) : seule vraie différence = `DATA_DIR` hardcodé Colab dans la cellule d'imports.
- Refactoring : cellule [02] scindée en 3 — imports / détection+DATA_DIR adaptatif / chargement. `DATA_DIR` se fixe automatiquement selon l'environnement (Colab → Drive, local → chemin relatif). Plus aucune modification manuelle nécessaire.

## 2026-05-02 — Audit documentation + guide EDA cellule par cellule

- Tous les fichiers de suivi mis à jour : `ROADMAP.md`, `JOURNAL.md`, `DECISIONS.md`, `QUESTIONS-OUVERTES.md`, `memory/project_phase1_eda.md`.
- `memoire/02-baseline/EDA_RAPPORT.md` réécrit : remplacé la synthèse scientifique par un **guide cellule par cellule** des 49 cellules du notebook (justification de chaque cellule, résultats attendus, fil narratif pour présentation encadreurs).
- Suite → lancer `src/baseline/fayam/tsf_transf.py` sur les 4 clusters.

## 2026-05-02 — Experiment tracker + notebook Colab baseline

- Run tracé : `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/` (run.md + command.sh + diff.patch)
- Notebook généré : `code/notebooks/baseline-fayam-transformer.ipynb` (37 cellules, JSON valide)
  - Pipeline fidèle au `tsf_transf.py` FAYAM — `Accelerator` remplacé par `torch.device` direct (Colab single-GPU)
  - Ajouts vs original : seeding complet, gradient clipping, checkpointing toutes les 10 époques, tqdm, métriques RMSE/R²/Spearman, extraction `output_attentions=True` post-training (cross_attn + enc_attn couche 4, 100 séries → `.npy`)
  - Résultats sauvegardés sur Google Drive : `MyDrive/m2-xai-faas/experiments/baseline-fayam-transformer/`
- Suite → Upload notebook sur Colab (T4 GPU) et lancer Run All.

## 2026-05-04 — Debug notebook + plan d'étude architecture

- Bug `TypeError: 'Axes' object is not iterable` corrigé dans `baseline-fayam-transformer.ipynb` (cellule 31) : `squeeze=False` + `.flatten()` sur `plt.subplots`.
- Plan d'étude architecture 5 jours créé (`memoire/00-meta/PLAN-ETUDE-ARCHITECTURE.md`) : pipeline données, encodeur, décodeur, inférence + visuels PNG/.excalidraw.
- Suite → lancer le notebook sur Colab (T4), démarrer J1 du plan d'étude en parallèle.

## 2026-05-04 — Run Colab baseline exécuté + archivage HTML

- Run `baseline-fayam-transformer.ipynb` exécuté avec succès sur Colab T4.
- HTML + dossier `_files` archivés dans `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/`.
- Métriques extraites du HTML et reportées dans `run.md` : MASE=0.8169, sMAPE=0.2903, RMSE=4.0750, R²=0.5845, Spearman=0.8342. Status run → done.
- Suite → démarrer J1 du plan d'étude architecture (lecture Rasul & Rogge + esquisse encoder-decoder).

## 2026-05-05 — Adaptation notebook : 4 clusters locaux

- Constat : `FaalSa/dataME` ne contient qu'1 série dans le test split → comparaison FAYAM impossible.
- Notebook `baseline-fayam-transformer.ipynb` adapté en 6 cellules : nouveau RUN_NAME, chargement des 4 CSV depuis Drive (19 séries), métriques avec colonnes `cluster`/`function_id`, nouvelle cellule de synthèse par cluster.
- Pipeline GluonTS et architecture inchangés. Cardinality s'auto-ajuste à 19.
- Suite → re-upload sur Colab et Run All. Comparer ensuite par cluster avec FAYAM Table VII.

## 2026-05-05 — Dossier de traçage nouveau run (session 18)

- `NameError: name 'cross_attn_arr' is not defined` diagnostiqué : problème d'ordre d'exécution Colab (cellule 34 lancée avant cellule 33), pas un bug de code.
- Dossier `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/` créé avec `run.md` pré-rempli (config, tableau métriques vide par cluster C0/C4/C6/C8, chemins Drive).
- Suite → re-upload notebook sur Colab, Run All, remplir le `run.md` avec les métriques obtenues.

## 2026-05-05 — Correction cellule extraction attention manquante (session 19)

- Diagnostic : la cellule de code qui définit `cross_attn_arr` était **absente** du notebook — seul le titre markdown "## 11 — Extraction attention" était présent.
- Cellule insérée après `cell-33` : forward pass teacher-forcé avec `output_attentions=True`, extraction `cross_attentions[-1]` + `encoder_attentions[-1]`, empilement en `cross_attn_arr` (n, heads, pred_len, ctx_len) et `enc_attn_arr`, sauvegarde `.npy` sur Drive.
- Suite → re-upload notebook sur Colab + Runtime → Run All.

## 2026-05-05 — Archivage HTML run local-clusters (session 20)

- Run Colab exécuté avec succès (19 séries, 4 clusters locaux). HTML + dossier `_files` téléchargés.
- HTML copié depuis `Downloads/` et renommé dans `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/` : `baseline-fayam-local-clusters - Colab.html` + `_files/`.
- Suite → récupérer `metrics.csv`, `metrics_by_cluster.csv`, `cluster_mapping.csv`, `metrics.json` depuis Drive pour compléter `run.md` et comparer avec FAYAM Table VII.

## 2026-05-05 — Analyse métriques run local-clusters (session 21)

- 4 fichiers résultats (`metrics.csv`, `metrics_by_cluster.csv`, `cluster_mapping.csv`, `metrics.json`) copiés depuis `Downloads/` dans `results/` du dossier de run.
- `run.md` complété : métriques globales (MASE=1.38, sMAPE=1.45, R²=-1.26) + par cluster + analyse des causes (hétérogénéité, zero-inflation C6/C8, dataset trop petit) + comparaison FAYAM Table VII + pistes pour H1.
- Status run → `done`. C0 (Spearman=0.74) et C8 (MASE=0.44) exploitables pour l'analyse d'attention (H1/H3).

## 2026-05-05 — Archivage figures PNG (session 22)

- `scatter_metrics.png`, `loss_curve.png`, `forecast_samples.png` copiés depuis `Downloads/` dans `results/` du dossier de run.
- Dossier de run `2026-05-05_baseline-fayam-local-clusters/` maintenant complet : HTML + CSV/JSON + PNG.
- `cross_attention_heatmap.png` (Drive/attentions/) à récupérer si besoin ultérieur.

## 2026-05-05 — Runs par cluster + archivage cluster 6 (session 23)

- Notebook `baseline-cluster0.ipynb` créé (34 cellules) : modèle dédié à C0 (3 fonctions), 1 plot par fonction (zoom 6 h + vue 24 h), vue comparative 3 fonctions, attention par fonction.
- Run cluster 6 archivé : dossier `code/experiments/runs/2026-05-05_baseline-cluster6/` créé — HTML + résultats extraits du zip Drive (metrics, 5 PNG forecast, loss_curve, run.md).
- Diagnostic C6 confirmé : RMSE≈0, MASE≈0, sMAPE≈2, R²=0, Spearman=NaN → modèle prédit 0 en permanence (zero-inflated) → C6 exclu de H1.
- Notebook `baseline-cluster8.ipynb` créé (34 cellules, 6 fonctions : 964-977, ~5 inv/min, ~25 % zéros).
- Suite → uploader `baseline-cluster8.ipynb` sur Colab → Run All. Puis lancer C0 et C4. Comparer métriques par cluster.

## 2026-05-05 — Archivage run cluster 8 (session 24)

- Run `baseline-cluster8.ipynb` exécuté sur Colab T4. Dossier `code/experiments/runs/2026-05-05_baseline-cluster8/` créé : zip Drive extrait (10 fichiers), HTML + `_files/` copiés, `run.md` rédigé.
- Métriques C8 dédié : MASE=0.44, sMAPE≈2.0, R²=-0.79, Spearman=+0.05 — identiques au modèle mixte (session 21) → isolation n'améliore pas C8.
- Suite → lancer `baseline-cluster0.ipynb` sur Colab (cible prioritaire H1).

## 2026-05-05 — Synthèse 4 baselines + cible H1 = C4 (session 25)

- Comparaison croisée des 4 runs dédiés archivés dans `experiments/runs/2026-05-05_baseline-cluster{0,4,6,8}/`. Verdict :
  - **C4** : R²=0.37, Spearman=0.92, sMAPE=0.22 — seul cluster où le modèle FAYAM apprend.
  - **C0** : R²=−0.006, Spearman=−0.008 — échec, modèle prédit la moyenne (suspect : magnitude ~120 000 vs scaler interne HF).
  - **C6** : RMSE≈0, sMAPE=2 — trivial predictor (déjà acté session 23).
  - **C8** : R²=−0.79, sMAPE=2 — trivial predictor lui aussi.
- Confrontation EDA vs résultats : 3/4 prévisibles (C6 et C8 par burstiness/zéros, C4 par FFT 75–80 %). C0 = surprise non capturée par l'EDA → ajouter signal d'alerte sur magnitude brute dans futures EDA.
- **Cible H1 actée : C4** (cf. [`memoire/00-meta/DECISIONS.md`](../memoire/00-meta/DECISIONS.md), entrée 2026-05-05). Phase 1 close.
- Suite → Phase 2 : étude architecture `TimeSeriesTransformer` HF (J1 de `PLAN-ETUDE-ARCHITECTURE.md`).
