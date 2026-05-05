# Mémoire — 02-baseline

## 2026-04-26

- Dossier créé.
- Pas encore de reproduction lancée.

## 2026-04-28

- Code FAYAM extrait de `transformer_m2-main.zip` → copié dans `code/src/baseline/fayam/`.
- Hyperparamètres Transformer identifiés : `prediction_length=120`, `context_length=240`, `freq="1T"`, `encoder_layers=4`, `decoder_layers=4`, `d_model=32`.
- Dataset disponible sur HuggingFace Hub : `FaalSa/dataME`.
- **Discordance importante** : le mémoire dit DBSCAN mais le code utilise HDBSCAN — paramètres non documentés.
- Notes complètes dans `code/src/baseline/fayam/BASELINE.md`.
- Prochaine étape : lancer `tsf_transf.py` sur nos 4 clusters pour reproduire les métriques FAYAM (voir `STEPS.md`).

## 2026-05-02 — Rapport EDA

- `EDA_RAPPORT.md` rédigé : 7 sections, chiffres réels du run 2026-05-02_08-31.
- Couvre : contexte, description données, stats descriptives, stationnarité (19/19 ADF p≈0), périodicité (24h universelle), analyse des zéros, profil par cluster, décisions prétraitement, ordre d'entraînement recommandé (C0→C4→C8→C6).
- Document autonome utilisable en présentation encadreurs et comme base du chapitre données du mémoire.

## 2026-05-05 — Synthèse 4 baselines dédiés + bilan EDA→prédiction

- 4 runs dédiés archivés (cf. `code/experiments/runs/2026-05-05_baseline-cluster{0,4,6,8}/`).
- **Tableau comparatif des baselines** :
  | Cluster | MASE | sMAPE | R² | Spearman | Verdict |
  |---|---|---|---|---|---|
  | C0 | 2.13 | 0.25 | −0.006 | −0.008 | Échec — prédit la moyenne |
  | C4 | 1.23 | 0.22 | **0.37** | **0.92** | Succès |
  | C6 | ≈0 | 2.00 | 0.00 | NaN | Trivial predictor |
  | C8 | 0.44 | 2.00 | −0.79 | 0.05 | Trivial predictor |
- **Bilan EDA→prédiction** : 3/4 résultats prévisibles depuis l'EDA (C6 explicite cellule 17, C4 par FFT 75–80 %, C8 partiellement par burstiness+zéros). C0 = échec **non prévu** par l'EDA — magnitude brute (~120 000) probablement responsable d'une instabilité du scaler interne du `TimeSeriesTransformer`.
- **Cible H1 actée : C4** (cf. [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md), entrée 2026-05-05).
- Limite EDA à corriger : ajouter un signal d'alerte sur magnitude absolue (pas seulement CV) dans les futures analyses.
