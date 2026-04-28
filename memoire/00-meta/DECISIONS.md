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

## 2026-04-26 — Code unique dans `code/`, pas de duplication HPC

- **Alternatives écartées** : code dupliqué dans `hpc/` ; submodule git.
- **Justification** : éviter la divergence des versions, simplifier l'iteration locale ↔ cluster.

## 2026-04-26 — Fichiers Claude (CLAUDE.md, .claude/, memory/) non commités

- **Justification** : préférence personnelle. L'explicabilité publique passe par les `README.md` de chaque dossier.
