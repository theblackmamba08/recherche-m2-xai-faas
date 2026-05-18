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

### Dernière session : 2026-05-18 (session 52 — Run B4 catastrophe, Run B5 prêt — dernier test)

- **Phase actuelle** : Phase 2 — H1. 5 configurations FAIL. Run B5 est le dernier test avant pivot H2.
- **Avancée** :
  - Run B4 (mix=0.10 sans schedules) : R²=**−3.58**, Spearman=**0.44**. PIRE que Run B3 (−1.59). Sans warm-up, le décodeur n'apprend jamais proprement.
  - **Diagnostic confirmé** : le warm-up est l'ingrédient critique. L'analyse linéaire que j'avais faite était incomplète (oubliait que dec_output évolue end-to-end avec h_evidence).
  - **Run B5 généré** : warm-up + anneal γ + LayerNorm + **mix=0.05** (plus petit que tous les runs). C'est la meilleure combinaison testable.
- **Prochain pas** :
  1. 🔴 Push GitHub (Run B5).
  2. 🔴 Colab : `softcam-cluster4-v3-runB5.ipynb` → Run All.
  3. 🟡 PASS H1.C → H1 défendable (mix=0.05 + carte M exacte).
  4. 🟡 FAIL → 5 configurations FAIL = preuve empirique d'incompatibilité architecturale. **Pivot H2 (TsSHAP / SHAPformer)** documenté comme contribution scientifique négative.

### Session précédente : 2026-05-18 (session 51 — Run B3 FAIL, découverte structurelle, Run B4 prêt)

- **Phase actuelle** : Phase 2 — H1 en cours. 4 variantes essayées, problème structurel identifié.
- **Avancée majeure** :
  - Run B3 (v3 LayerNorm, warm-up + anneal) : R²=−1.59, Spearman=0.78. Le LayerNorm a bien corrigé le collapse de M (max_weight 0.97 → 0.06), mais R² ne remonte presque pas.
  - **Analyse structurelle** : `parameter_projection(0.7·dec + 0.3·h_evidence)` est linéaire. h_evidence (single softmax) est structurellement plus pauvre que dec_output (cross-attention multi-tête multi-couches). À mix=0.3, on impose 30% de la prédiction à venir d'une branche faible → R² ne peut pas atteindre 0.30.
  - **Run B4 généré** : mix=0.10 constant, γ=0, modèle v3. Estimation R² ≈ 0.9×0.53 + 0.1×(-1) ≈ 0.38 → PASS attendu.
- **Prochain pas** :
  1. 🔴 Push GitHub (notebook + v3 + run.md) — IMPORTANT.
  2. 🔴 Sur Colab : `softcam-cluster4-v3-runB4.ipynb` → Disconnect → Run All (~15 min).
  3. 🟡 PASS H1.C → H1 défendable avec petit mix : *« mix=0.10 préserve la précision tout en fournissant une carte M algébriquement exacte »*.
  4. 🟡 FAIL → Run B5 (mix=0.05) ou Run B6 (architectural : M remplace la cross-attention du dernier décodeur).

### Session précédente : 2026-05-18 (session 50 — SoftCAM v3 + Run B3 prêt)

- **Phase actuelle** : Phase 2 — H1 en cours. Fix #4 implémenté dans `softcam_transformer_v3.py`, Run B3 prêt.
- **Avancée** : `SoftCAMTransformerV3ForPrediction` créé (hérite v2 + `evidence_norm = LayerNorm(d_model)` sur `h_evidence` avant mix). Notebook Run B3 généré (36 cellules, commit `96b62a0`). Schedules Run B2 conservés.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → `code/notebooks/softcam-cluster4-v3-runB3.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~15-20 min sur T4).
  3. 🟡 PASS H1.C (R²≥0.30, Spearman≥0.85) → analyser cartes M, rédiger contribution.
  4. 🟡 FAIL → pivot H2 (TsSHAP/SHAPformer).

### Session précédente : 2026-05-18 (session 49 — Run B2 archivé, Fix #4 planifié)

- **Phase actuelle** : Phase 2 — H1 en cours. Run B et Run B2 FAIL, Fix #4 architectural identifié.
- **Avancée** :
  - Run B (mix=0.3 constant, γ=1e-3 constant) : R²=−2.83, Spearman=0.33. *Attention collapse* : M quasi-Dirac sur s≈211, max_weight=0.85.
  - Run B2 (warm-up mix + annealing γ) : R²=−1.97, Spearman=0.80 (progrès +85/+47 pp). M encore effondrée (s≈25, max_weight=0.97) mais Spearman proche du gate 0.85.
  - Cause persistante : `h_evidence = bmm(M, enc_hidden)` n'est pas dans le même espace statistique que `dec_output` → la tête `parameter_projection` produit une mauvaise échelle → R² négatif mais Spearman OK.
  - Résultats archivés : HTML + test_metrics.json + run.md dans `code/experiments/runs/2026-05-18_softcam-cluster4-v2-runB2/`.
- **Prochain pas** :
  1. 🔴 **Fix #4** : ajouter `nn.LayerNorm(d_model)` sur `h_evidence` avant le mix dans `softcam_transformer_v2.py` (2 lignes : `__init__` + `output_params`).
  2. 🔴 Générer Run B3 (même schedules Run B2 + LayerNorm) + lancer sur Colab T4.
  3. 🟡 Si PASS H1.C → analyser cartes M.
  4. 🟡 Si FAIL → pivot H2 (TsSHAP/SHAPformer).

### Session précédente : 2026-05-18 (session 47 — audit Run B + fix docs entropie)

- **Phase actuelle** : Phase 2 — Run B audité au peigne fin, prêt à lancer sur Colab.
- **Audit OK** : tous les fixes de Run A présents (re-seed RNG, 4+4 layers, seed=998, split [:-120], pas de val_loader). Shapes `model.explain()` et batch mismatch enc/dec vérifiés.
- **Bugs de doc corrigés** : cellule 9 et plot cellule 10 affichaient `+ elastic − entropy → max entropie → sparsité`. Or le modèle calcule `forecast + elastic + entropy` (somme) — on **minimise** l'entropie pour obtenir des M piqués. Commit `08be6b3` poussé.
- **Caveat noté** : fenêtre de test = 22h-24h (jour 14), pas 17h-19h. H1.A teste où M se concentre pour cette fenêtre, pas le pic journalier en direct.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → ouvrir `code/notebooks/softcam-cluster4-v2-runB.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~15-20 min sur T4).
  3. 🟡 PASS H1.C (R²≥0.30 ET Spearman≥0.85) → analyser cartes M (H1.A/H1.D).
  4. 🟡 FAIL H1.C → essayer mix=0.1 (Run C) avant pivot H2.

### Session précédente : 2026-05-18 (session 46 — Run A PASS + Run B généré)

- Run A fix5 validé : R²=0.5299, Spearman=0.9176.
- Run B notebook généré (36 cellules, commit `62e38ed`).

### Session précédente : 2026-05-18 (session 44 — Run A strict baseline, val_loader retiré)

- **Phase actuelle** : Phase 2 — Run A reconstruit en strict baseline FAYAM, prêt pour 4e exécution.
- **Avancée** :
  - Retrait complet du val_loader (val_rows, val_dataset, val_loader, monitoring val_r2/val_spear, plot val) dans `_generate_softcam_cluster4_v2_runA.py`. Élimine la consommation de RNG par `.generate()` à chaque epoch.
  - Notebook régénéré et poussé sur GitHub (commit `0b49075` sur main).
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → ouvrir `code/notebooks/softcam-cluster4-v2-runA.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~8-10 min sur T4).
  3. 🟡 Si PASS (R²≈0.37, Spearman≈0.92) → pipeline saine, on lance Run B (`use_evidence_layer=True, mix=0.3`).
  4. 🟡 Si encore FAIL → investigation plus poussée (lags, scaler, time features).

### Session précédente : 2026-05-18 (session 43 — Run A 3e exécution + diagnostic RNG drift)

- **Phase actuelle** : Phase 2 — sanity check Run A toujours FAIL, cause RNG identifiée.
- **Résultats Run A (4+4, seed=998, train split corrigé)** :
  - R² = 0.0529 (FAYAM 0.3701, -31.72 pp)
  - Spearman = 0.9052 (FAYAM 0.9201, -1.49 pp) → ordre quasi correct, magnitude sous-estimée
  - Forte amélioration vs runs précédents (R² évolue de -0.19 → -0.46 → 0.05)
- **Cause identifiée** : le baseline FAYAM n'a PAS de val_loader. Run A appelle `.generate()` (sampling stochastique 100 trajectoires) sur val à chaque epoch → consomme RNG → diverge la dynamique d'entraînement. Code v2 avec `use_evidence_layer=False` est mathématiquement identique au baseline (vérifié ligne 269 de `softcam_transformer_v2.py`).
- **Prochain pas** :
  1. 🔴 Retirer val_loader + val_dataset du notebook Run A (strict baseline).
  2. 🔴 Régénérer notebook, push GitHub, recharger sur Colab, Run All.
  3. 🟡 PASS attendu si l'hypothèse RNG est correcte.

### Session précédente : 2026-05-18 (session 42 — fix train split, notebook 100 % FAYAM)

- **Phase actuelle** : Phase 2 — sanity check Run A, 3e correctif appliqué.
- **Avancée** :
  - Run v2.1 (4+4, seed=998) analysé : R²=-0.4604, Spearman=0.8698. Dernier bug identifié : train split `target[:-240]` au lieu de `target[:-120]` (FAYAM). On entraînait sur 120 points de moins.
  - Fix : ligne 217 du générateur corrigée → `target[:-PREDICTION_LENGTH]`. Notebook régénéré, commit `977073f`.
  - Tous les écarts vs FAYAM sont maintenant corrigés : architecture (4+4), seed (998), train split (target[:-120]).
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → branche main** → `code/notebooks/softcam-cluster4-v2-runA.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All**.
  3. 🟡 PASS attendu : R²≈0.37, Spearman≈0.92. Si PASS → Run B (`use_evidence_layer=True, mix=0.3`).

### Session précédente : 2026-05-18 (session 41 — archivage HTML Run A, Colab cache détecté)

- **Phase actuelle** : Phase 2 — sanity check Run A pas encore validée.
- **Avancée** :
  - HTML du run Colab archivé dans `code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/`.
  - Diagnostic : Colab a exécuté l'ANCIEN notebook (cache) — `ENCODER_LAYERS=2` visible dans le HTML. Résultats inchangés : R²=-0.1861, Spearman=0.9190.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → branche main** → ouvrir `code/notebooks/softcam-cluster4-v2-runA.ipynb` (version corrigée 4+4, seed=998).
  2. 🔴 **Runtime → Disconnect and delete runtime** puis **Run All**.
  3. 🟡 Vérifier dans la cellule config que `ENCODER_LAYERS = 4` avant de lancer.

### Session précédente : 2026-05-18 (session 40 — comparaison FAYAM + seed fix)

- **Phase actuelle** : Phase 2 — sanity check Run A prêt, notebook entièrement aligné sur FAYAM.
- **Avancée** :
  - Comparaison FAYAM Table VII finalisée : R²=0.3701 / Spearman=0.92 sur C4 sont cohérents avec la fourchette FAYAM (-0.164 → 0.958). Notre Spearman=0.92 dépasse 5/6 datasets FAYAM.
  - Dernier écart corrigé : `SEED = 2026` → `SEED = 998` dans `_generate_softcam_cluster4_v2_runA.py`. Notebook régénéré, commit `459730b`. Le notebook est maintenant 100 % aligné avec le protocole FAYAM (architecture 4+4, seed 998).
- **Prochain pas** :
  1. 🔴 Upload `softcam-cluster4-v2-runA.ipynb` sur Colab (File → Open → GitHub → branch main).
  2. 🔴 T4 GPU → Run All (~10-15 min) → vérifier PASS : R² ≥ 0.30, Spearman ≥ 0.85.
  3. 🟡 Si PASS → sanity check validée, pipeline SoftCAMV2 saine → lancer Run B (`use_evidence_layer=True`, `mix=0.3`).
  4. 🟡 Si FAIL → dernier verrou = train split (target[:-240] vs target[:-120]) à investiguer.

### Session précédente : 2026-05-17 (session 36 — H1 v2 code)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer **v2** (diagnostic-friendly).
- **Avancée** :
  - Diagnostic du NO-GO v1 : hypothèse principale = **information bottleneck** (v1 remplace `dec_output` par `bmm(M, enc_hidden)`, bypass tout le travail du décodeur ; `parameter_projection` reçoit des stats très différentes de ce pour quoi il était initialisé).
  - Code v2 écrit dans `code/src/models/softcam_transformer_v2.py` (à part, v1 intact). Ajoute `use_evidence_layer: bool` (toggle parent FAYAM strict) et `evidence_mix: float ∈ [0,1]` (interpolation `h = (1-mix)·dec_output + mix·bmm(M,enc)`).
- **Prochain pas** : décider entre 4 runs A/B chaînés dans un notebook ou Run A seul (sanity check parent FAYAM avec proto exact 51 epochs full, pas d'early stop).

### Session précédente : 2026-05-17 (session 35 — premier run H1)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. **GATE H1.C échoué**, investigation requise avant pivot.
- **Résultats** :
  - Test R² = **-6.1565** (FAYAM 0.37 → -652 pp)
  - Test Spearman = **-0.8731** (FAYAM 0.92 → -179 pp)
  - **Anti-corrélation systématique** (per-series -0.85 à -0.90) → bug architectural, pas un problème de convergence
  - best val R² = 0.0837 (epoch 8), early stop epoch 18, training 5.5 min
- **Archive** : `code/experiments/runs/2026-05-17_04-52_softcam-cluster4-h1-v1/` (HTML + run.md, gitignored)
- **Prochain pas** :
  1. 🔴 **Sanity check forward parent FAYAM** sur Cluster 4 sans evidence layer — si lui converge, bug 100% dans notre code H1
  2. 🔴 **Inspection visuelle de M** (heatmaps `.npy` sur Drive) — détecter softmax dégénéré
  3. 🔴 **Test unitaire fin** : `model.explain()` == forward standard ? Signes corrects ? `encoder_last_hidden_state` capturé au bon moment par le hook lors de `model.generate()` ?
  4. 🟡 Si bug trouvé après 1-2 jours → H1 v2. Sinon → **pivot H2 (TimeSHAP)**.

### Session précédente : 2026-05-17 (session 34 — fin)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook définitivement prêt.
- **Avancée** : Fix final cellule clone — `get_ipython().system()` remplace `subprocess.run()`. Colab peut maintenant cloner sans problème de TTY/credentials (commit `83a843e`).
- **Prochain pas** : Coller le code corrigé dans la cellule Colab → Run → vérifier GATE H1.C (R² ≥ 0.30, Spearman ≥ 0.85).

### Session précédente : 2026-05-17 (session 34 — suite)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook définitivement prêt pour Colab.
- **Avancée** : Double fix cellule clone — `os.system()` → `subprocess.run()` (session 34), puis suppression de `capture_output=True` (session 34 suite) qui causait `fatal: could not read Username` même sur repo public.
- **Prochain pas** : relancer le notebook sur Colab T4 (Runtime → Disconnect and delete runtime → re-upload → Run All) → vérifier GATE H1.C.

### Session précédente : 2026-05-17 (session 34)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook prêt à tourner, bug Colab corrigé.
- **Avancée récente** :
  - Fix cellule git clone de `softcam-cluster4.ipynb` : `os.system()` → `subprocess.run()` avec capture et vérification du returncode (commit `7aeef09`).
  - Notebook poussé sur GitHub — prêt pour Colab T4.
- **Prochain pas** :
  1. 🔴 Sur Colab : Runtime → Disconnect and delete runtime → re-upload notebook → T4 GPU → Run All
  2. 🔴 Vérifier le **GATE H1.C** : R² ≥ 0.30 et Spearman ≥ 0.85 sur le test set
  3. 🟡 Si GO → analyser heatmaps M + comparer à `cross_attentions` (gratuit)
  4. 🟡 Si NO-GO → pivot immédiat vers H2 (TimeSHAP)

### Session précédente : 2026-05-16 (session 33)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. **Code implémenté**, prêt à tourner sur Colab T4.
- **Avancée récente** :
  - **Compréhension complète de l'architecture** validée par dialogue pédagogique (encoder + decoder + evidence layer). Architecture désormais maîtrisée pour la soutenance.
  - **Code H1 produit** :
    - `code/src/models/softcam_transformer.py` (~360 l.) — `SoftCAMTransformerConfig`, `SoftCAMTSPredictionOutput`, `SoftCAMTransformerForPrediction` (sous-classement HF, hook encodeur, override `output_params` → forward ET generate fonctionnent).
    - `code/tests/test_softcam_transformer.py` (10 tests pytest).
    - `code/notebooks/softcam-cluster4.ipynb` (33 cellules, généré par `_generate_softcam_cluster4.py`) — clone repo + GATE H1.C explicite + extraction & heatmaps des cartes M.
  - **Caveat L1 documenté** : sous softmax `mean(|M|) = 1/ctx` est constant (gradient nul) ; le vrai sparsity-inducing est l'entropie `γ·H(M)` (hyperparam exposé).
  - **Mémoires persistantes sauvegardées** (session 32 → 33) : argumentation H1, controverse Jain & Wallace, TFT concurrent, plan validation.
- **Prochain pas** :
  1. 🔴 Commit + push GitHub (`code/src/models/`, tests, notebook + générateur, doc updates)
  2. 🔴 Upload `softcam-cluster4.ipynb` sur Colab T4 → Runtime → Run All
  3. 🔴 Vérifier le **GATE H1.C** : R² ≥ 0.30 et Spearman ≥ 0.85 sur le test set
  4. 🟡 Si GO → analyser heatmaps M + comparer à `cross_attentions` (gratuit)
  5. 🟡 Si NO-GO → pivot immédiat vers H2 (TimeSHAP)

### Session précédente : 2026-05-16 (session 32)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Fiche TFT rédigée. Priorité 🔴 close.
- **Avancée récente** :
  - **Fiche TFT créée** : `memoire/01-litterature/fiches/2019_Lim_TFT.md` — architecture complète (VSN, GRN, IMHA), résultats 9 datasets, critique Jain & Wallace 2019, tableau comparatif TFT vs H1.
  - **Argument H1 consolidé** : M[t,s] = décomposition linéaire exacte (fidèle par construction) vs IMHA TFT = poids d'attention non fidèles.
  - **Phrase-clé soutenance** prête : "...c'est une décomposition linéaire prouvée, pas une attribution approximée."
- **Prochain pas** :
  1. 🟡 **Ajouter TFT au panorama XAI v3** (canevas Dr LACMOU 4 points + plus-value de transition)
  2. ✅ **Implémenter `evidence_layer`** dans `src/models/softcam_transformer.py` (fait session 33)

### Session précédente : 2026-05-14 (session 30 — clôture Phase 1 bis)

- **Phase actuelle** : **Phase 1 bis CLOSE.** HPO Cluster 4 exécutée et archivée. Décision baseline actée. En route vers Phase 2 (H1 SoftCAM-Transformer) + lecture TFT.
- **Avancée récente** :
  - Run `optimized-cluster4.ipynb` exécuté sur Colab T4, résultats rapatriés et archivés dans [`code/experiments/runs/2026-05-14_optimized-cluster4/`](../../code/experiments/runs/2026-05-14_optimized-cluster4/).
  - **HPO** : 15 trials, 4 complétés (73% pruned dont OOM). Best val R²=0.5347 avec `d_model=128, context=240, encoder_layers=4, lr=6.41e-4`.
  - **Test** : R²=**-1.39** (vs FAYAM=0.37) → dégradation -1.76pp. Cause : `context=240` (OOM-contraint) perd la périodicité 24h.
  - **Décision** : **FAYAM original conservé comme baseline** (seuil +20pp non atteint).
  - **Insight majeur** : `d_model=32` de FAYAM justifié empiriquement — seul `d_model` compatible avec `context=1440` sans OOM T4. Réponse à la question encadreur.
  - **Ablation 949** : dédié R²=0.215 > multi-opt R²=-1.257 → "écrasée par multi-task" (contamination context=240 à noter).
- **Prochain pas** (par ordre de priorité) :
  1. 🔴 **Lire le papier TFT** (Lim et al. 2019, arxiv:1912.09363) et créer `memoire/01-litterature/fiches/2019_Lim_TFT.md`
  2. 🟡 Re-positionner la motivation H1 dans `memoire/03-contribution/MEMOIRE.md` (alternative à TFT, fidélité par construction vs attribution attention)
  3. 🟡 Ajouter TFT au panorama XAI v3 avec canevas Dr LACMOU (contexte/problèmes/transposabilité/limites)
  4. 🟢 Démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) — cartographier `parameter_projection` HuggingFace (cible de l'evidence layer SoftCAM)

### Session précédente : 2026-05-09 (session 28)

- **Phase actuelle** : préparation finale de la séance Dr LACMOU (panorama XAI v2 + SPEECH section architecture).
- **Avancée récente** :
  - `SPEECH-architecture.md` créé pour la section 10 (Architecture SoftCAM-Transformer) : 10 slides commentées avec texte à dire, transitions, anticipation de 5 questions probables, notes Cabrel.
- **Prochain pas** : Cabrel relit le speech avant la séance ; en parallèle, démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) (cartographier `parameter_projection` HuggingFace).

### Session précédente : 2026-05-09 (session 27)

- **Phase actuelle** : **Phase 1 terminée**, transition vers Phase 2. Présentation panorama XAI v2 prête.
- **Avancée récente** :
  - **Présentation Panorama XAI v2** construite et compilée : `presentations/2026-05-09-panorama-xai-v2/` (64 pages, 908 Ko). Canevas Dr LACMOU appliqué uniformément (5 environnements TeX dédiés : `ctxbox/probbox/transposbox/limitbox/plusbox`).
  - **Zooms approfondis** : 7 slides SHAPformer (architecture training masking + inférence 2^N + résultats + position semi-intrinsèque + discussion *« peut-on rendre 1-pass ? »*) et 6 slides SoftCAM (architecture intrinsèque + ElasticNet + résultats CNN + perspective ViT/Transformer).
  - **Section Architecture SoftCAM-Transformer (10 slides)** : motivation, schéma TikZ complet, entrées/sorties par bloc, code PyTorch evidence layer, équation centrale, ElasticNet visualisé, 3 variantes A/B/C, exemple carte d'évidence sur C4, hypothèses H1.A--H1.E.
  - SVG architecture créé : `memoire/03-contribution/figures/softcam-transformer-architecture.svg` (réutilisable mémoire + Beamer).
  - Discussion conceptuelle actée : Variante B (carte × embeddings encodeur) retenue pour le design SoftCAM-Transformer ; idée *« SHAPformer self-explainable »* reléguée au chapitre Discussion (dénature SHAPformer si poussée plus loin).
- **Prochain pas** : démarrer **Phase 2** — J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) : cartographier `parameter_projection` HuggingFace pour localiser la cible exacte de l'evidence layer.

### Session précédente : 2026-05-08 (session 26)

- **Phase actuelle** : **Phase 1 terminée**, transition vers Phase 2. Retour panorama XAI traité.
- **Avancée récente** :
  - DEBRIEF de la présentation panorama XAI du 28/04/2026 rédigé. Retour Dr LACMOU : présentation trop technique, étudiant en peine à l'oral.
  - **Cadre prescrit pour toute méthode XAI** : 4 points (contexte / problèmes résolus / transposabilité / limites) + plus-value de transition. Acté dans [DECISIONS.md](DECISIONS.md) (entrée 2026-05-08) et mémoire persistante.
- **Prochain pas** : (1) reprendre les slides du panorama selon le cadre Dr LACMOU avant la prochaine séance ; (2) en parallèle, démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) (architecture `TimeSeriesTransformer` HuggingFace).

### Session précédente : 2026-05-05 (session 25)

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
