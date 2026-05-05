# Décisions méthodologiques

> Format : date — décision — alternatives écartées — justification.

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
