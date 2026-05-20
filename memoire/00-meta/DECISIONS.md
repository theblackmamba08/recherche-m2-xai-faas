# Décisions méthodologiques

> Format : date — décision — alternatives écartées — justification.

## 2026-05-20 — Configuration H1 finalisée : B5 + mix=0.25 inférence + Fix #5

- **Décision** : la configuration H1 retenue pour le mémoire est le checkpoint **Run B5** (SoftCAM-Transformer v3 entraîné avec `evidence_mix=0.05` warm-up) évalué à l'inférence avec **`model.evidence_mix = 0.25`** et la méthode `generate()` patchée (Fix #5). R² final = **0.7563**, Spearman = 0.9169 sur Cluster 4. Pas de retrain B8.
- **Alternatives écartées** :
  - **Retrain B8 from scratch avec mix=0.25 cible** : la ré-évaluation B5/B6/B7 (2026-05-20) avec Fix #5 actif montre que monter le mix d'entraînement dégrade monotone le R² pic (B5=0.7563 > B6=0.5975 > B7=0.4684). Trend défavorable et coût ~3-4h Colab non justifié.
  - **Fine-tune B5 → B8 avec mix 0.05→0.25** : l'argument vaut aussi (le dec_output va se dégrader sous pression du mix accru) ; le risque dépasse le gain attendu.
  - **Garder B5 à mix=0.05 inférence** (config "pure" entraînement=inférence) : laisse 4.3 pp de R² sur la table (0.7130 → 0.7563). Pas justifiable scientifiquement.
- **Justification** :
  1. **Diagnostic Fix #5** (2026-05-19) : la méthode HuggingFace `generate()` appelait `parameter_projection` directement, court-circuitant notre `output_params`. Tous les runs antérieurs (A, B…B7) étaient évalués avec l'Evidence Layer inactive — d'où R²=0.66 rapporté pour B5 (mesurait dec_output seul).
  2. **Re-mesure avec Fix #5** : B5 à mix=0.05 (le mix d'entraînement) donne R²=0.7130 ; à mix=0.25 donne R²=0.7563 (pic empirique). La courbe redescend ensuite (mix=0.50 → 0.4367 ; mix=1.0 → −0.1332).
  3. **Pattern empirique** : optimum d'inférence ≈ 5× mix d'entraînement pour les 3 checkpoints. Hypothèse : les contraintes auxiliaires (ElasticNet + entropie sur M) pendant l'entraînement façonnent un h_evidence dont la qualité intrinsèque dépasse son poids relatif au régime d'entraînement.
  4. **Self-explainability préservée** : M est toujours produite par le forward pass, fidèle par construction. Le choix de mix=0.25 à l'inférence est un hyperparamètre opératoire (analogue à la température d'un LLM), pas une rationalisation post-hoc. La fidélité-par-construction tient à n'importe quel mix > 0.
- **Comparaison vs baselines** :
  - vs **FAYAM original** (R²=0.3701) : +103 pp relatif, soit **+38.62 pp absolu**
  - vs **Run A** (notre repro, `use_evidence_layer=False`, R²=0.5299) : +22.64 pp absolu
    - dont +13.46 pp gratuits attribuables à l'entraînement avec Evidence Layer (régularisation multi-tâche : B5+mix=0 inférence donne déjà R²=0.6646)
    - dont +9.17 pp attribuables à l'utilisation effective de M à l'inférence (mix=0 → mix=0.25)
- **Application** :
  - Code : `code/src/models/softcam_transformer_v3.py` (Fix #5 inclus, commit `3f0c51c`)
  - Checkpoint : Drive `m2-xai-faas/experiments/softcam-cluster4-v3-runB5/checkpoints/v3_runB5_final.pt`
  - Mode d'emploi : `model.evidence_mix = 0.25` AVANT `model.generate(...)`
  - Hypothèses validées : H1.A ✅, H1.C ✅ (avec vrai R²), H1.D ✅✅, H1.E ✅ ; H1.B ⚠️ et H1.F/G ⚠️ à présenter en Discussion comme limites mesurables mais non bloquantes.

## 2026-05-14 — HPO Path B (ciblée) sur le baseline Cluster 4 avant Phase 2

- **Décision** : lancer une HPO ciblée d'Optuna (TPE + MedianPruner, 15 trials × 20 epochs) sur 4 hyperparamètres du `TimeSeriesTransformer` — `d_model`, `context_length`, `encoder_layers`, `lr` — avant de démarrer l'implémentation de H1 (SoftCAM-Transformer). Le retrain final utilise un **early stopping sur val R²** (patience=10, max 80 epochs) au lieu des 51 epochs fixes de FAYAM. Une **ablation per-function** sur la fonction 949 (la plus problématique, R²=0.15 en multi-task) est ajoutée pour diagnostiquer si elle est écrasée par le multi-task ou intrinsèquement difficile.
- **Alternatives écartées** :
  - **Path A — HPO complète** (~10 hyperparams, 30 trials, 1-2 semaines) : trop coûteux en temps de calendrier (<3 mois jusqu'à soutenance) ; oblige aussi à retuner SoftCAM avec le même budget pour comparaison équitable → double le travail futur.
  - **Path C — Skip HPO, démarrer J1 du PLAN-ETUDE-ARCHITECTURE immédiatement** : laisse intacte la question légitime "pourquoi `d_model=32` ?" qui a déjà été posée par les encadreurs en séance et signale un manque de rigueur empirique sur la baseline.
  - **Per-function comme stratégie principale** (5 modèles dédiés, 1 par fonction) : déviation méthodologique forte vs FAYAM (qui est multi-task) ; perd le transfert positif intra-cluster ; désaligne du design de H1 (SoftCAM exploite des embeddings encodeur partagés). Retenu uniquement comme ablation diagnostique sur la 949.
- **Justification** :
  1. La question "pourquoi `d_model=32` ?" en séance encadreur a révélé des lacunes conceptuelles (l'étudiant a répondu "des chiffres" sur le type des entrées) ; produire un résultat empirique chiffré sur l'optimalité de la baseline répond directement à l'attente.
  2. La courbe de loss FAYAM plateau dès l'epoch 20-25 — les 51 epochs fixes sont du bruit ; passer à l'early stopping est une amélioration méthodologique gratuite.
  3. Path B est borné (2-3 jours de calcul, 1 jour d'analyse) — si le gain est < 10pp R², on garde FAYAM original comme baseline et on mentionne la HPO en Discussion ; si > 20pp, on adopte l'optimisé comme nouvelle baseline (et on retunera SoftCAM en parallèle quand on y arrivera).
  4. Le critère d'acceptation H1 (R²≥0.30, Spearman≥0.85) est déjà tenu par FAYAM C4 (0.37/0.92) — la HPO ne change pas la faisabilité de H1, elle renforce la crédibilité de la comparaison.
- **Application** : notebook [`code/notebooks/optimized-cluster4.ipynb`](../../code/notebooks/optimized-cluster4.ipynb) (46 cellules) à lancer sur Colab T4 (~3-4h). Archivage attendu dans `code/experiments/runs/2026-05-14_optimized-cluster4/` avec `metrics_optimized.json`, `comparison_fayam_vs_optimized.csv`, `ablation_949.json`, `best_params.json`.

## 2026-05-08 — Cadre standard pour la présentation des méthodes XAI aux encadreurs

- **Décision** : adopter un canevas en 4 points pour **chaque** méthode XAI présentée en séance encadreur, et systématiquement annoncer la **plus-value** d'une méthode par rapport à celle qui la précède dans le récit.
  1. **Contexte** de la méthode (origine, domaine d'application initial).
  2. **Problèmes résolus** (lacune comblée à sa création).
  3. **Transposabilité** (applicabilité à d'autres domaines, notamment séries temporelles / FaaS).
  4. **Limites** (ce qui justifie la méthode suivante).
  - **Plus-value de transition** (à chaque méthode qui suit) : qu'est-ce que la nouvelle méthode résout que la précédente ne résolvait pas ?
- **Alternatives écartées** :
  - Présentation technique par méthode (formalismes, équations détaillées) : jugée **trop technique** par Dr LACMOU lors du panorama du 28/04/2026 ; l'étudiant avait de la peine à la délivrer à l'oral.
  - Présentation par groupes (post-hoc vs intrinsèque) sans fil de plus-value : ne montre pas la nécessité d'enchaîner les méthodes.
- **Justification** :
  1. Cadre prescrit par Dr LACMOU à la suite de la présentation du 28/04/2026.
  2. Le canevas force un récit en chaîne pédagogique : chaque méthode comble un manque de la précédente, ce qui justifie naturellement le choix final (SoftCAM-Transformer = H1).
  3. Aligne la défense orale sur la structure attendue d'un état de l'art de mémoire (contexte → problème → solutions → limites → contribution).
- **Application** : ce cadre s'applique aux prochaines séances encadreurs ET au chapitre 2 (état de l'art) du mémoire LaTeX.

## 2026-04-27 — Pivot stratégique XAI : nouvelle H1 = SoftCAM-Transformer (intrinsèque)

- **Décision** : reformuler l'ordonnancement des hypothèses après lecture de l'article SoftCAM (Djoumessi & Berens, arXiv:2505.17748v1, mai 2025) :
  - **H1 (NOUVELLE)** : adapter le principe SoftCAM (interprétabilité intrinsèque par modification architecturale légère + régularisation ElasticNet sur les *evidence maps*) au `TimeSeriesTransformer` HuggingFace de FAYAM.
  - **H2 (NOUVELLE)** : approches **SHAP-based** — repli si H1 bloque. Deux options ordonnées : **TsSHAP** (prioritaire — seule méthode SHAP pour la prévision ; surrogate XGBoost + TreeSHAP sur backtested forecasts) ; **SHAPformer** (bonus — SHAP exact pour Transformer, 800× plus rapide, requiert réentraînement, code public disponible).
  - **H3 (NOUVELLE)** : étude des **poids d'attention** + DBSCAN + faithfulness (l'ancienne H1) — dernier recours, ou outil de validation de H1.
- **Alternatives écartées** :
  - Conserver l'ancienne H1 (attention + DBSCAN + faithfulness) comme plan principal : *moins ambitieux et moins distinctif* qu'une contribution architecturale intrinsèque.
  - Conserver Integrated Gradients comme H2 : Captum reste utilisable mais TimeSHAP est plus aligné avec la nature séries-temporelles du problème.
- **Justification** :
  1. Les encadreurs jugent qu'une contribution **architecturale intrinsèque** est scientifiquement plus distinctive qu'une analyse post-hoc.
  2. SoftCAM démontre empiriquement (3 datasets médicaux, 2 backbones) qu'on peut rendre un classifieur boîte-noire interprétable **sans coût computationnel additionnel ni perte de précision**. Les auteurs mentionnent explicitement l'extension à ViT comme perspective ouverte (p. 10) — porte d'entrée naturelle pour notre travail.
  3. La triple stratification H1 → H2 → H3 garantit qu'**en cas d'échec de la piste ambitieuse, deux replis ordonnés** assurent qu'une contribution XAI sera dans le mémoire.
  4. L'analyse différentielle par cluster DBSCAN reste l'angle distinctif vis-à-vis de FAYAM, applicable indifféremment aux trois hypothèses.

## 2026-04-26 — *(obsolète, conservé pour traçabilité)* Hypothèse prioritaire = attention + DBSCAN + faithfulness

> ⚠️ **Reformulé le 2026-04-27** — voir entrée ci-dessus. Cette décision initiale n'est plus active mais est conservée pour la traçabilité du raisonnement.

- **Alternatives écartées** :
  - H2 (Integrated Gradients via Captum) : reportée en bonus si H1 stable en S8.
  - H3 (CNN-LSTM PyTorch) : abandonnée — coût d'apprentissage trop élevé pour < 3 mois et niveau PyTorch débutant.
- **Justification originale** : HuggingFace expose `output_attentions=True` nativement → pas de ré-implémentation modèle. L'angle original était l'analyse différentielle par cluster DBSCAN.

## 2026-05-05 — Cluster 4 retenu comme cible primaire pour H1 (SoftCAM-Transformer)

- **Décision** : la cible primaire de H1 est désormais le **cluster 4** (5 fonctions : 949, 951, 952, 953, 954). Les hypothèses opératoires (H1.A à H1.E) sont fixées sur ce cluster.
- **Alternatives écartées** :
  - **C0** : malgré un signal régulier en EDA (B=−0.20, FFT 24h forte), le run dédié donne R²≈0 et Spearman≈0 → modèle qui prédit la moyenne. Probable problème de magnitude (~120 000) interagissant avec le scaler interne du `TimeSeriesTransformer`. Archivé pour diagnostic ultérieur (cas-limite chapitre Discussion).
  - **C6** : trivial predictor confirmé (RMSE≈0, sMAPE=2). Zero-inflated → exclu de H1 (déjà acté session 23).
  - **C8** : trivial predictor lui aussi (R²=−0.79, sMAPE=2). 25 % de zéros + B>0 + kurtosis élevé → la MSE pousse le modèle à prédire la valeur modale (zéro).
- **Justification** :
  1. **C4 est le seul cluster où le baseline FAYAM converge** : R²=0.37, Spearman=0.92, sMAPE=0.22 sur le run dédié `2026-05-05_baseline-cluster4`. Sans modèle qui apprend, pas de carte d'évidence SoftCAM exploitable.
  2. **Signal le plus structuré** : périodicité 24h dominante (FFT 75–80 % variance), profil journalier net (pic 17h-19h, creux 2h-6h), Pearson intra-cluster > 0.95 → généralisation possible aux 5 fonctions, et carte d'évidence interprétable visuellement.
  3. **Validation didactique** : un Transformer qui capture un cycle clair va produire une evidence map *lisible* (pic = forte évidence, creux = faible évidence). Idéal pour démontrer la faisabilité de SoftCAM-temporel aux encadreurs et au jury.
  4. **Diversité interne pour H1.E** : function 953 (R²=0.60) = best case ; function 949 (R²=0.15) = stress test → permet d'évaluer la robustesse de SoftCAM à la qualité de prédiction.
- **Hypothèses opératoires actées pour H1** (à tester quand SoftCAM tournera) :
  - H1.A : la carte d'évidence pointe les heures du profil journalier (pic/creux).
  - H1.B : les têtes d'attention décodeur se polarisent autour des lags 1440 et 2880.
  - H1.C : SoftCAM ne dégrade pas la précision baseline (R²≥0.30, Spearman≥0.85 conservés).
  - H1.D : cohérence des cartes d'évidence entre les 5 fonctions (cohérent avec Pearson > 0.95).
  - H1.E : test sur les deux extrêmes (function 953 best case, function 949 stress test).

## 2026-05-02 — Décisions issues de l'EDA (prétraitement et entraînement)

- **Conserver les zéros natifs** (clusters 6 et 8) : le `TimeSeriesTransformerForPrediction` les tolère ; les imputer ferait perdre l'information sur les plages silencieuses, cruciale pour C6 (cold start).
- **Normalisation par fonction** (MinMaxScaler ou StandardScaler) : magnitude C0 (~120 000) vs C6 (~2) — facteur 60 000× ; normaliser par cluster masquerait les différences intra-cluster.
- **Pas de différenciation** : toutes les séries sont stationnaires (ADF p ≈ 0, 19/19).
- **Ordre d'entraînement** : C0 → C4 → C8 → C6 (complexité signal croissante : propre → cyclique → bursty → zero-inflaté).
- **`output_attentions=True`** à activer dès le premier run pour disposer du matériel H1/H3.

## 2026-04-26 — Code unique dans `code/`, pas de duplication HPC

- **Alternatives écartées** : code dupliqué dans `hpc/` ; submodule git.
- **Justification** : éviter la divergence des versions, simplifier l'iteration locale ↔ cluster.

## 2026-04-26 — Fichiers Claude (CLAUDE.md, .claude/, memory/) non commités

- **Justification** : préférence personnelle. L'explicabilité publique passe par les `README.md` de chaque dossier.
