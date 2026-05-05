# Roadmap — < 3 mois jusqu'à soutenance

> Démarrage : 2026-04-26 — Pivot stratégique : 2026-04-27 — Soutenance cible : *(à fixer)*

> ⚠️ **Refonte du 2026-04-27** : après lecture de SoftCAM (Djoumessi & Berens, 2025), les hypothèses sont reformulées. La piste « attention weights » devient un repli, pas le plan principal. Voir [DECISIONS.md](DECISIONS.md) pour la justification.

## Phase 1 — Reproduction baseline (S1-S2)

- [ ] Récupérer le code FAYAM (Transformer uniquement, pas le CNN-LSTM)
- [ ] Identifier les hyperparamètres manquants (`predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension` — non chiffrés p. 76 du mémoire FAYAM)
- [ ] Reproduire les résultats Transformer sur ≥ 3 datasets parmi les 18 (en utilisant les `HashFunction` du tableau IX p. 86)
- [ ] **Activer dès maintenant `output_attentions=True`** pour disposer du matériel H3 sans réentraîner
- [ ] Cartographier précisément l'architecture du `TimeSeriesTransformer` HuggingFace : identifier la **projection finale du décodeur** (cible de H1)
- [ ] Documenter les écarts éventuels dans `02-baseline/MEMOIRE.md`

## Phase 2 — H1 : SoftCAM-Transformer (S3-S6)

> **Objectif** : transposer le principe SoftCAM (couche d'evidence + ElasticNet) du CNN au TimeSeriesTransformer.

- [ ] Récupérer le code de référence SoftCAM (`https://anonymous.4open.science/r/SoftCAM-E1A3/`) — étudier la `class-evidence layer` et la perte ElasticNet
- [ ] Définir l'équivalent temporel d'une *class evidence map* (carte temps-passé × temps-futur ? par cluster DBSCAN ? par seuil de charge ?) → **point de design critique à débattre avec encadreurs**
- [ ] Implémenter `src/models/softcam_transformer.py` : variante du `TimeSeriesTransformer` avec couche d'évidence et perte ElasticNet
- [ ] Entraîner sur ≥ 3 datasets représentatifs (un par profil DBSCAN : peu fréquente / régulière / populaire)
- [ ] Évaluer **performance prédictive** (sMAPE, RMSE, R², Spearman — mêmes métriques que FAYAM) → la performance ne doit pas dégrader
- [ ] Évaluer **qualité d'explication** (faithfulness/comprehensiveness/sufficiency, activation precision/sensitivity adaptés au temporel)
- [ ] Comparer la carte d'évidence apprise à la matrice d'attention décodeur (sanity check, et préparation à H3 comme outil de validation)

### Critère de bascule vers H2
Si à la fin de S6 (≈ 2 semaines de prototypage) l'adaptation SoftCAM→Transformer ne converge pas ou dégrade fortement la précision : **basculer sur H2 sans attendre**.

## Phase 3 — H2 : SHAP-based (repli, S7-S8 si H1 bloque)

- [ ] Étudier **TimeSHAP** (Bento et al.) — application directe au Transformer FAYAM (post-hoc)
- [ ] Comparer KernelSHAP / TimeSHAP sur les 3 datasets
- [ ] Reprendre la grille d'évaluation faithfulness pour comparabilité H1 ↔ H2

## Phase 4 — H3 : étude des attention weights (dernier recours OU outil de validation)

- [ ] Extraire les attention maps (déjà disponibles si phase 1 a activé `output_attentions=True`)
- [ ] Clustering DBSCAN sur les charges → 3-5 macro-profils dérivés des 33 clusters FAYAM
- [ ] Visualiser attention moyenne par cluster
- [ ] Calculer comprehensiveness + sufficiency par profil
- [ ] **Si H1 ou H2 a abouti** : H3 sert à *valider* la cohérence entre carte d'évidence apprise et attention apprise (et n'est plus une contribution principale).

## Phase 5 — Rédaction (S9-S12)

- [ ] Plan détaillé du mémoire
- [ ] Chapitre 1 : intro + problématique
- [ ] Chapitre 2 : état de l'art (FAYAM + SoftCAM + TimeSHAP + littérature attention temps réel)
- [ ] Chapitre 3 : méthode (l'hypothèse retenue parmi H1/H2/H3, avec justification du choix au regard du calendrier)
- [ ] Chapitre 4 : résultats
- [ ] Chapitre 5 : discussion + limites
- [ ] Relecture, audit similarité, soumission

## Jalons encadreurs

- 2026-04-27 : pivot stratégique acté (passage à H1 = SoftCAM-Transformer).
- *(à compléter après chaque entretien)*

---

## État courant

> 📍 **Première chose à lire en début de session.** Mis à jour à chaque fin de session par le hook Stop.

### Dernière session : 2026-05-05 (session 25)

- **Phase actuelle** : **Phase 1 terminée** — 4 runs dédiés archivés, cible H1 actée.
- **Avancée récente** :
  - Synthèse comparative des 4 baselines dédiés (C0/C4/C6/C8). **Seul C4 apprend** : R²=0.37, Spearman=0.92, sMAPE=0.22.
  - C0 = échec inattendu (R²≈0, modèle prédit la moyenne — probable problème scaler interne sur magnitude ~120 000). C6/C8 = trivial predictors (zero-inflated).
  - **Cible primaire H1 = Cluster 4** actée dans [DECISIONS.md](DECISIONS.md). Hypothèses opératoires H1.A–H1.E fixées (carte d'évidence sur heures du profil journalier, attention sur lags 1440/2880, non-dégradation précision, cohérence intra-cluster, test best/worst sur fn 953/949).
  - C0 archivé comme cas-limite pour chapitre Discussion (diagnostic scaler à creuser).
- **Prochain pas** : démarrer **Phase 2 — H1** : étude architecture `TimeSeriesTransformer` HuggingFace (J1 du `PLAN-ETUDE-ARCHITECTURE.md`) pour localiser la projection finale du décodeur (cible SoftCAM).

### Session précédente : 2026-05-05 (session 24)

- **Phase actuelle** : **Phase 1** — run C8 archivé, C0 à lancer.
- **Avancée récente** :
  - Run `baseline-cluster8` exécuté et archivé (MASE=0.44, sMAPE≈2.0, R²=-0.79). Modèle dédié = même résultat que mixte → C8 intrinsèquement difficile (zero-inflated).
  - Conclusion (depuis invalidée — voir session 25) : *"C0 est la seule cible viable pour H1"*.
- **Prochain pas** : uploader `baseline-cluster0.ipynb` sur Colab T4 → Run All.

### Session précédente : 2026-05-05 (session 23)

- **Phase actuelle** : **Phase 1** — runs par cluster engagés (C0 notebook créé, C6 archivé).
- **Avancée récente** :
  - Notebook `baseline-cluster0.ipynb` créé : modèle dédié C0, plots par fonction (zoom 6 h + vue 24 h + comparaison 3 fonctions), attention par fonction.
  - Dossier run `2026-05-05_baseline-cluster6/` archivé (zip Drive + HTML). Diagnostic C6 : zero-inflated → exclu de H1.
  - Notebook `baseline-cluster8.ipynb` créé (6 fonctions : 964, 965, 967, 968, 969, 977 — ~5 inv/min, ~25 % zéros).
- **Prochain pas** : uploader `baseline-cluster8.ipynb` sur Colab + Run All (T4 GPU). Ensuite lancer `baseline-cluster0.ipynb`. Archiver résultats et comparer métriques par cluster avec FAYAM Table VII.

### Session précédente : 2026-05-05 (session 22)

- **Phase actuelle** : **Phase 1 terminée** — dossier de run complet (HTML + CSV/JSON + PNG).
- **Avancée récente** :
  - `scatter_metrics.png`, `loss_curve.png`, `forecast_samples.png` archivés dans `results/`.
  - Run `2026-05-05_baseline-fayam-local-clusters` entièrement archivé localement.
- **Prochain pas** : démarrer **Phase 2 — H1** : lire `PLAN-ETUDE-ARCHITECTURE.md` (J1), puis implémenter SoftCAM-Transformer sur C0/C8.

### Session précédente : 2026-05-05 (session 17)

- **Phase actuelle** : **Phase 1** — notebook adapté pour les 4 clusters locaux (19 séries), run à relancer.
- **Avancée récente** :
  - Constat : `FaalSa/dataME` ne contient qu'1 série dans le test split → comparaison FAYAM impossible avec ce dataset.
  - Notebook `baseline-fayam-transformer.ipynb` adapté en 6 cellules : nouveau `RUN_NAME=2026-05-05_baseline-fayam-local-clusters`, chargement des 4 CSV depuis Drive, métriques avec colonnes `cluster`/`function_id`, nouvelle cellule de synthèse par cluster.
  - Pipeline GluonTS et architecture inchangés.
- **Prochain pas** : re-upload notebook sur Colab + Run All. Comparer ensuite par cluster avec FAYAM Table VII.

### Session précédente : 2026-05-04 (session 16)

- **Phase actuelle** : **Phase 1** — notebook Colab prêt, run à lancer. Plan d'étude architecture engagé.
- **Avancée récente** :
  - Clarification origine dataset `FaalSa/dataME` : pipeline FAYAM (Azure Trace → HDBSCAN → dataToHub.py → HuggingFace).
  - Plan d'étude architecture 5 jours créé → voir [`memoire/00-meta/PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md).
  - Bug `TypeError: 'Axes' not iterable` corrigé dans `baseline-fayam-transformer.ipynb` (cellule 31).
  - Run Colab exécuté avec succès. HTML archivé. Métriques reportées dans `run.md` : MASE=0.8169, sMAPE=0.2903, RMSE=4.0750, R²=0.5845, Spearman=0.8342.
- **Prochain pas** : démarrer J1 du plan d'étude architecture — lecture Rasul & Rogge + esquisse encoder-decoder en Excalidraw.

### Session précédente : 2026-05-02 (session 15 — suite 4)

- **Phase actuelle** : **Phase 1** — notebook Colab prêt, run tracé, à lancer.
- **Avancée récente** :
  - Skill `experiment-tracker` exécuté : `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/` (run.md + command.sh + diff.patch).
  - Notebook `code/notebooks/baseline-fayam-transformer.ipynb` généré (37 cellules) — pipeline FAYAM complet + RMSE/R²/Spearman + extraction attention `output_attentions=True` post-training.
- **Prochain pas** : uploader `baseline-fayam-transformer.ipynb` sur Google Colab (T4 GPU) → Run All → remplir la table métriques dans `run.md`.

### Avant-dernière session : 2026-04-28 (session 13 — complète)

- **Phase actuelle** : **Phase 1 amorcée** — baseline FAYAM intégrée + EDA des 4 clusters terminée.
- **Avancée récente** : `code/notebooks/EDA_clusters.ipynb` créé (39 cellules, 11 sections) — analyse par fonction ET par cluster : statistiques descriptives, séries temporelles (heatmaps 14×1440, profils journaliers), zéros, distributions, ACF/FFT, ADF, cohérence intra/inter-cluster, recommandations prétraitement.
- **Prochain pas** : (1) exécuter l'EDA notebook pour valider les graphiques ; (2) lancer `tsf_transf.py` sur les 4 clusters (dataset HuggingFace `FaalSa/dataME`) et reproduire les métriques FAYAM (sMAPE, RMSE, R², Spearman).

### Avant-dernière session : 2026-04-28 (session 12 — finalisée)

- **Phase actuelle** : présentation panorama XAI prête à l'oral — 47 slides + script complet.
- **Avancée récente** : `SPEECH.md` créé (~30 s/slide, ~20 min, tableau temps cibles par section, conseils de rythme).
- **Prochain pas** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer, métriques.
