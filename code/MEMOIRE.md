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

## 2026-05-18 — Fix Run A : ENCODER/DECODER_LAYERS 2→4 (session 39)

- Comparaison cellule par cellule `softcam-cluster4-v2-runA.ipynb` vs `baseline-cluster4.ipynb` : bug identifié dans le générateur `_generate_softcam_cluster4_v2_runA.py`.
- **Cause réelle du FAIL** : `ENCODER_LAYERS = 2, DECODER_LAYERS = 2` au lieu de 4+4 (FAYAM) — 68k params vs 94k params. Modèle trop petit pour calibrer les magnitudes, d'où R²=-0.19 malgré Spearman=0.92.
- Correction : `ENCODER_LAYERS = 4, DECODER_LAYERS = 4` dans le générateur, notebook régénéré, commit `72a2d26`.
- **Prochaine étape** : re-upload le notebook corrigé sur Colab T4 → Run All → PASS attendu (R²≈0.37).

## 2026-05-17 — Résultats Run A : FAIL — bug d'échelle (session 38)

- Run `softcam-cluster4-v2-runA` exécuté sur Colab T4. Archive : `code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/` (HTML + run.md).
- **Résultats TEST** : R²=-0.1861 (-55.62 pp vs FAYAM=0.3701) ; Spearman=0.9190 (-0.11 pp vs FAYAM=0.9201) → **FAIL**.
- **Signal clé** : Spearman ≈ 0.92 (quasi identique FAYAM) mais R² très négatif → le modèle prédit le **bon ordre** mais la **mauvaise échelle**. Ce n'est PAS le bug d'anti-corrélation de v1.
- **Diagnostic** : problème de normalisation/dénormalisation à l'évaluation. Hypothèse principale : `inverse_transform` non appelé sur les prédictions (ou appelé différemment vs FAYAM `baseline-cluster4.ipynb`). À comparer cellule par cellule.
- Per-series : 949 (R²=-0.649), 951 (-0.519), 952 (+0.058), 953 (+0.073), 954 (+0.108) — toutes FAIL R², toutes OK Spearman (≈0.88-0.96).
- **Prochaine étape** : inspecter la cellule d'évaluation du notebook Run A vs `baseline-cluster4.ipynb` — chercher où FAYAM appelle `scaler.inverse_transform()` ou la dénormalisation interne HF.

## 2026-05-17 — Notebook Run A (sanity check pipeline) (session 37)

- Nouveau notebook `code/notebooks/softcam-cluster4-v2-runA.ipynb` (28 cellules, via générateur `_generate_softcam_cluster4_v2_runA.py`).
- Configure `SoftCAMTransformerV2ForPrediction(use_evidence_layer=False)` → comportement parent FAYAM strict, et entraîne **51 epochs full sans early stopping** (aligné `baseline-cluster4.ipynb`).
- Verdict automatique : PASS si `|R² - 0.37| ≤ 10 pp` → pipeline saine, bug v1 isolé dans evidence layer. FAIL sinon → bug pipeline à corriger avant tout.
- En attente du run Colab par l'user (~10-15 min sur T4).

## 2026-05-17 — Implémentation H1 v2 (diagnostic-friendly) (session 36)

- Nouveau fichier `code/src/models/softcam_transformer_v2.py` (~330 lignes) — sous-classe `SoftCAMTransformerV2ForPrediction` avec 2 leviers diagnostiques :
  - `use_evidence_layer: bool` — toggle off → comportement parent FAYAM strict
  - `evidence_mix: float ∈ [0,1]` — interpolation `h = (1-mix)·dec_output + mix·bmm(M,enc)`. mix=0 = FAYAM, mix=1 = v1
- v1 conservé intact. `__init__.py` mis à jour : expose v1 + v2 en parallèle.
- Compilation Python OK sur les 2 fichiers.
- Prochaine étape : décider du plan d'A/B test (4 runs : `use=False/51ep`, `use=True,mix=0`, `mix=1.0` reproduit v1, `mix=0.3` hybrid). User en attente d'arbitrage entre "notebook complet" vs "Run A seul d'abord".

## 2026-05-17 — Premier run H1 sur Colab : **NO-GO** (session 35)

- Run `softcam-cluster4-h1-v1` exécuté sur Colab T4 (04:52, 5.5 min, early stop epoch 18/60).
- **GATE H1.C échoué** : Test R²=-6.16 (vs FAYAM 0.37), Spearman=-0.87 (anti-corrélation systématique).
- Archive locale : `code/experiments/runs/2026-05-17_04-52_softcam-cluster4-h1-v1/` (HTML + iframes + run.md complet — gitignored).
- Synthèse + diagnostic 3 hypothèses (bug signe / forward_hook / softmax dégénéré) → `memoire/03-contribution/MEMOIRE.md`. Mémoire persistante `project_h1_v1_nogo.md` créée.
- Prochaine étape : 3 checks (sanity forward parent / inspection M / test unitaire `_evidence_layer`) avant pivot H2.

## 2026-05-17 — Fix git clone Colab v3 (session 34 — fin)

- Troisième et dernier fix cellule clone : `subprocess.run()` abandonné au profit de `get_ipython().system()` (équivalent Python du `!` Colab) — accès complet au shell, zéro problème TTY/credentials. Commit `83a843e`.

## 2026-05-17 — Fix git clone Colab v2 (session 34 — suite)

- Deuxième fix cellule clone : `capture_output=True` supprimé — c'était la vraie cause (`fatal: could not read Username`). Sans capture, git hérite du terminal Colab et clone sans demander de credentials même sur repo public.
- Générateur `_generate_softcam_cluster4.py` mis à jour, notebook régénéré et poussé (commit `c1b96f1`).

## 2026-05-17 — Fix cellule git clone Colab (session 34)

- Cellule "Récupération du repo" du notebook `softcam-cluster4.ipynb` corrigée : `os.system()` remplacé par `subprocess.run()` avec `capture_output=True`, vérification du `returncode`, et `FileNotFoundError` explicite si `code/src/models/` est absent après clone.
- Script générateur `_generate_softcam_cluster4.py` mis à jour en conséquence — notebook régénéré (33 cellules, inchangé).
- Commit `7aeef09` poussé — Colab peut maintenant voir le message d'erreur exact si le clone échoue.
- Prochaine étape : upload le nouveau notebook sur Colab T4 → Runtime → Disconnect and delete runtime → Run All → vérifier GATE H1.C.

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

## 2026-05-14 — HPO Path B + ablation per-function 949 (session 29)

- **Notebook `optimized-cluster4.ipynb` créé** (46 cellules, 56.7 Ko) : pipeline HPO complet sur Cluster 4.
- **Stratégie HPO** : Optuna TPE + MedianPruner, 15 trials × 20 epochs max, 4 hyperparams (`d_model ∈ {32,64,128}`, `context_length ∈ {240,480,1440}`, `encoder_layers ∈ {2,3,4}`, `lr ∈ log[1e-5, 1e-3]`). Fixés : `n_heads=2`, `embedding_dim=[2]`, `dropout=0.1`, `batch_size=256`, `decoder_layers=encoder_layers`.
- **Validation HPO** : 120 minutes held-out avant le test horizon (positions -240 à -120 de target_full). Objectif : maximiser moyenne R² sur les 5 fonctions de C4. SQLite storage sur Drive pour resume en cas de déconnexion Colab.
- **Retrain final** : early stopping sur val R² (patience=10, max 80 epochs) — remplace les 51 epochs fixes de FAYAM qui plateau dès l'epoch 20-25 d'après la courbe de loss.
- **Ablation 949** : à la fin du notebook, modèle dédié à la fonction 949 (la plus problématique en multi-task, R²=0.15) avec hyperparams optimaux. Diagnostic : R²(dédié) > R²(multi) → fonction écrasée par le multi-task ; ≈ → intrinsèquement difficile ; < → transfert positif.
- **Comparaison FAYAM vs Optimisé** : per-fonction (tableau + barplot) sur MASE, sMAPE, RMSE, R², Spearman.
- Script générateur : `_generate_optimized_cluster4.py` (conservé pour reproductibilité).

## 2026-05-14 — Archivage résultats HPO + décision baseline (session 30)

- Run exécuté sur Colab T4 (15h42–16h41). Résultats rapatriés depuis `Downloads/` et archivés dans `code/experiments/runs/2026-05-14_optimized-cluster4/`.
- **HPO** : 15 trials, 4 complétés, 11 pruned (dont 7 OOM — `context_length ≥ 480` OOM pour `d_model ≥ 64` sur T4). Best val R²=0.5347 avec `d_model=128, context=240, encoder_layers=4, lr=6.41e-4`.
- **Test** : R²=**-1.3854** (vs FAYAM=0.3701) — dégradation -1.76. Cause : `context=240` (OOM-contraint) perd la périodicité 24h.
- **Décision : FAYAM original conservé comme baseline.** Seuil +20pp non atteint.
- **Insight** : `d_model=32` de FAYAM justifié empiriquement — seul `d_model` compatible avec `context=1440` sans OOM T4.
- **Ablation 949** : dédié R²=0.215 > multi-opt R²=-1.257 → "écrasée par multi-task" (attention : contamination context=240).
- **Attention weights** : cross_attn `(5,2,120,240)` et enc_attn `(5,2,240,240)` disponibles sur Drive pour H3.
- Artefacts : `run.md`, `hpo/best_params.json`, `results/{final_summary,metrics_optimized,comparison_fayam_vs_optimized,ablation_949}.json`, `optimized_cluster4.ipynb`.
- Suite → 🔴 Lire TFT (arxiv:1912.09363) + créer fiche `2019_Lim_TFT.md`. Puis Phase 2 J1 (`parameter_projection` HuggingFace).

## 2026-05-05 — Synthèse 4 baselines + cible H1 = C4 (session 25)

- Comparaison croisée des 4 runs dédiés archivés dans `experiments/runs/2026-05-05_baseline-cluster{0,4,6,8}/`. Verdict :
  - **C4** : R²=0.37, Spearman=0.92, sMAPE=0.22 — seul cluster où le modèle FAYAM apprend.
  - **C0** : R²=−0.006, Spearman=−0.008 — échec, modèle prédit la moyenne (suspect : magnitude ~120 000 vs scaler interne HF).
  - **C6** : RMSE≈0, sMAPE=2 — trivial predictor (déjà acté session 23).
  - **C8** : R²=−0.79, sMAPE=2 — trivial predictor lui aussi.
- Confrontation EDA vs résultats : 3/4 prévisibles (C6 et C8 par burstiness/zéros, C4 par FFT 75–80 %). C0 = surprise non capturée par l'EDA → ajouter signal d'alerte sur magnitude brute dans futures EDA.
- **Cible H1 actée : C4** (cf. [`memoire/00-meta/DECISIONS.md`](../memoire/00-meta/DECISIONS.md), entrée 2026-05-05). Phase 1 close.
- Suite → Phase 2 : étude architecture `TimeSeriesTransformer` HF (J1 de `PLAN-ETUDE-ARCHITECTURE.md`).

## 2026-05-18 — Run A 3e exécution + diagnostic RNG drift (session 43)

- 3e exécution Run A archivée dans `code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/softcam-cluster4-v2-runA-corrected.html` : R²=0.0529 (vs 0.3701 FAYAM, -31.72 pp), Spearman=0.9052 (-1.49 pp). Forte amélioration vs runs précédents (R²=-0.46 → 0.05) mais toujours FAIL.
- Cause probable identifiée : le baseline FAYAM n'a PAS de val_loader, alors que Run A appelle `.generate()` (sampling stochastique de 100 trajectoires) chaque epoch sur val_loader → consomme du RNG → diverge la dynamique d'entraînement même avec le même seed.
- Suite → retirer val_loader du notebook Run A (strict baseline) avant la 4e exécution.

## 2026-05-18 — Fix train split target[:-240]→target[:-120] (session 42)

- Analyse du HTML v2.1 (4+4, seed=998) : R²=-0.4604, Spearman=0.8698 — encore FAIL. Dernier écart identifié : train split `target[:-2*PREDICTION_LENGTH]` (notre notebook) vs `target[:-PREDICTION_LENGTH]` (FAYAM). On entraînait sur 120 points de moins.
- Fix appliqué dans `_generate_softcam_cluster4_v2_runA.py` (ligne 217), notebook régénéré, commit `977073f`. Le notebook est maintenant 100 % aligné avec FAYAM (4+4, seed=998, train split identique).
- Suite → recharger le notebook depuis GitHub (File → Open → GitHub → main) et relancer Run All → PASS (R²≈0.37) attendu.

## 2026-05-18 — Archivage Run A (ancien notebook — ENCODER_LAYERS=2) (session 41)

- HTML téléchargé par le user archivé dans `code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/` (écrase l'ancien HTML).
- Diagnostic : le HTML montre `ENCODER_LAYERS = 2` → Colab a exécuté l'ancien notebook en cache, pas la version corrigée. Résultats identiques au run précédent (R²=-0.1861, Spearman=0.9190).
- Suite → user doit ouvrir le notebook depuis GitHub (File → Open → GitHub → main) pour charger la version 4+4, seed=998, puis relancer.

## 2026-05-18 — Comparaison FAYAM Table VII + correction seed (session 40)

- Comparaison FAYAM Table VII (datasets minute, R² Transformer : -0.164 → 0.958) vs nos résultats C4 (R²=0.3701, Spearman=0.92) : cohérence confirmée — notre R² est dans la fourchette FAYAM côté bas-moyen, notre Spearman=0.92 est supérieur à 5/6 datasets FAYAM.
- Bug seed corrigé dans `_generate_softcam_cluster4_v2_runA.py` : `SEED = 2026` → `SEED = 998` (valeur FAYAM). Notebook `softcam-cluster4-v2-runA.ipynb` régénéré, commit `459730b`.
- Suite → uploader `softcam-cluster4-v2-runA.ipynb` sur Colab T4, Run All → attendre PASS (R²≈0.37), puis lancer Run B (`use_evidence_layer=True`, `mix=0.3`).

## 2026-05-16 — Implémentation H1 SoftCAM-Transformer (session 32)

- **Nouveau package** `code/src/models/` créé avec `softcam_transformer.py` (~360 lignes) :
  - `SoftCAMTransformerConfig(TimeSeriesTransformerConfig)` — ajoute `alpha_l1`, `beta_l2`, `gamma_entropy`.
  - `SoftCAMTSPredictionOutput(Seq2SeqTSPredictionOutput)` — expose `evidence_map`, `forecast_loss`, `elastic_loss`, `entropy_loss`.
  - `SoftCAMTransformerForPrediction(TimeSeriesTransformerForPrediction)` :
    - sous-classement strict ; encodeur / décodeur **inchangés** ;
    - 1 seul nouveau module : `evidence_linear = Linear(d_model → context_length)` ;
    - hook automatique sur l'encodeur pour cacher `encoder_last_hidden_state` → `output_params()` retrouve `enc_hidden` sans changer la signature HF ;
    - 1 seul point d'insertion (`output_params`) → fonctionne en `forward` ET `generate` sans réécrire ce dernier ;
    - méthode `explain()` pour extraction teacher-forcée de M sur le test set.
- **Caveat L1 documenté** : sous softmax, `mean(|M|) = 1/ctx` est constant → gradient nul. Le terme sparsity-inducing réel est l'entropie de ligne `γ·H(M)` (ajouté en option).
- **Tests** : `code/tests/test_softcam_transformer.py` (~10 tests pytest — shapes, somme=1, gradients finis, generate fonctionnel, explain).
- **Notebook Colab** : `_generate_softcam_cluster4.py` génère `softcam-cluster4.ipynb` (33 cellules) — clone repo + import `src.models` + boucle entraînement avec monitoring des 3 composantes de loss + GATE H1.C (R²≥0.30, Spearman≥0.85) + extraction & visualisation des cartes M par fonction.
- **Hyperparams H1 v1** : reconduits FAYAM (`d_model=32, ctx=240, layers=2, lr=6e-4`) + `α=0.0, β=1e-3, γ=1e-3`.
- Suite → push GitHub, upload notebook sur Colab T4, lancer le run, vérifier le GATE.
