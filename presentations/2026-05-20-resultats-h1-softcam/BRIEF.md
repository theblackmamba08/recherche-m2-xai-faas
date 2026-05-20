# BRIEF — Résultats H1 SoftCAM-Transformer

> À remplir AVANT la présentation.

- **Date** : 2026-05-20 *(à ajuster selon disponibilité encadreurs)*
- **Audience** : Encadreurs (Dr LACMOU + co-encadreur)
- **Durée cible** : 20-25 min + questions
- **Format** : Beamer

## Objectif

Présenter la validation complète de l'hypothèse H1 : le SoftCAM-Transformer
améliore la prédiction de charge FaaS sur Cluster 4 tout en fournissant une
explication fidèle par construction via la carte d'évidence M.
Obtenir un retour sur la décision de ne pas réentraîner et sur la formulation
de la contribution pour la rédaction.

## Points clés (suggestions à valider)

1. **Architecture** : Evidence Layer insérée dans le TimeSeriesTransformer HuggingFace —
   M = softmax(Linear(dec_output)), h = (1−mix)·dec_output + mix·LayerNorm(bmm(M, enc_hidden)).
   Fidélité par construction : M est l'unique poids algébrique, contrairement à la
   cross-attention TFT (Jain & Wallace 2019).

2. **Performance** : R²=0.7563, Spearman=0.9169 (B5 + mix=0.25 à l'inférence).
   +38.6 pp vs FAYAM original (0.3701). Gain décomposé : +13.5 pp régularisation
   seule, +9.2 pp utilisation effective de M.

3. **Trouvaille originale — dissociation entraînement/inférence** : l'optimum
   d'inférence (mix=0.25) est 5× le mix d'entraînement (0.05). Interprétation :
   h_evidence apprend des patterns sparses et stables sous contrainte ElasticNet/entropie ;
   à l'inférence, lui donner plus de poids améliore la prédiction.

4. **Validation des 7 hypothèses** :
   - H1.C ✅ R²=0.7563 (seuil : 0.30)
   - H1.D ✅ Pearson=0.992 intra-cluster (cohérence de M)
   - H1.E ✅ ρ=−0.80 entre R² et entropie de M
   - H1.F ✅ +5.37% Δ MAE quand on masque top-k (comprehensiveness)
   - H1.G ✅ 97.13% préservation à k=1 (sufficiency)
   - H1.A ✅ indicatif — argmax M dans pic 17-19h (test partiel — limite à mentionner)
   - H1.B ⚠️ reframé — M ≠ cross-attention (attendu et souhaitable, pas un échec)

5. **Crise Fix #5 résolue** : generate() HuggingFace court-circuitait l'Evidence Layer
   à l'inférence. Patché par override d'une ligne. L'entraînement (forward) n'était
   pas affecté — checkpoints valides.

## Questions anticipées

- **Q : Pourquoi ne pas réentraîner avec mix=0.25 directement ?**
  → R envisagée : B6 (mix=0.10) et B7 (mix=0.15) montrent que monter le mix
  d'entraînement dégrade monotonement le R² pic. B5+mix=0.25 à l'inférence est
  supérieur à tout retrain. Décision économique et empiriquement justifiée.

- **Q : Tuner mix à l'inférence, est-ce encore self-explainable ?**
  → R envisagée : mix est un hyperparamètre global d'inférence (analogue à la
  température d'un LLM), pas une rationalisation par instance. La fidélité par
  construction tient à tout mix > 0. On parle de "calibrated self-explainability".

- **Q : Pourquoi Cluster 4 seulement ? Généralisation ?**
  → R envisagée : C0, C6, C8 ne convergent pas comme baseline FAYAM → impossible
  d'y tester SoftCAM sans biais. C4 est le seul cluster où la baseline est saine.
  Limite à mettre en Discussion.

- **Q : H1.B échoue (ρ=0.16) — est-ce une limite ?**
  → R envisagée : non, c'est cohérent. Comparer M à cross-attention est circulaire
  (cross-attention n'est pas fiable — Jain & Wallace 2019). La divergence montre
  que M est un signal distinct, ce qui est l'objectif.

- **Q : Une seule seed — résultats reproductibles ?**
  → R envisagée : limite assumée (seed=998). Bootstrap non fait. À mentionner
  en Discussion avec la généralisation Cluster 4 comme limite conjointe.

## Demandes à l'audience

1. **Validation de la configuration finale** : B5 + mix=0.25 à l'inférence — pas
   de retrain B8. Accord encadreurs nécessaire avant rédaction.
2. **Formulation de la contribution** : "calibrated self-explainability" vs
   "architectural self-explainability" — quelle formulation préférer dans le mémoire ?
3. **Calendrier rédaction** : quand est la prochaine séance ? Objectif : premier
   draft chapitre H1 pour relecture.
