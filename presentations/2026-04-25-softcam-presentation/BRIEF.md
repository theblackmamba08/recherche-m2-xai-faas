# BRIEF — Soft-CAM : Rendre les Modèles Boîtes Noires Auto-Explicables

> Rempli a posteriori (présentation déjà réalisée).

- **Date** : 2026-04-25 (samedi)
- **Audience** : Dr LACMOU ZEUTOUO Jerry (séance avec un seul encadreur)
- **Durée** : non précisée
- **Format** : Beamer (24 diapositives) — présentation jugée incomplète, non formellement reçue

## Objectif

Présenter la fiche de lecture approfondie de l'article SoftCAM (Djoumessi & Berens, 2025) suite à la demande de Dr LACMOU lors de la séance du 17/04/2026.

## Points clés présentés

1. **Motivation** : CNN surpassent les experts humains en imagerie médicale, mais leur opacité freine l'adoption. Limites des méthodes post-hoc actuelles : infidélité, instabilité, incohérence, surcoût.
2. **Contribution SoftCAM** : modifier l'architecture CNN elle-même pour qu'elle génère ses propres explications en un seul forward pass — sans post-traitement.
3. **Architecture** : remplacer GAP + FCL par une couche convolutive 1×1 → cartes d'évidence A ∈ ℝ^(M×N×C) → AvgPool + Softmax → prédiction. Préserve l'information spatiale.
4. **Régularisation ElasticNet** : L(y, ŷ) = CE + λ₁||A||₁ + λ₂||A||₂². Lasso → parcimonie (lésions focales) ; Ridge → complétude (grandes régions) ; ElasticNet = compromis adaptatif.
5. **Protocole expérimental** : 3 datasets médicaux (Kaggle Fundus DR, Retinal OCT, RSNA CXR) ; architectures ResNet-50 et VGG-16 ; comparaison contre 5 méthodes post-hoc (GradCAM, IG, Guided BP, ScoreCAM, LayerCAM).
6. **Résultats** : performance de classification préservée voire améliorée ; SoftCAM surpasse les méthodes post-hoc en fidélité (OCT, CXR) ; Sparse SoftCAM = meilleure précision de localisation top-k.
7. **Limites** : résolution cartes limitée par backbone, méthodes pixel-par-pixel plus fines sur certains datasets, non évalué sur ViT.
8. **Perspectives ouvertes** : intégration avec Vision Transformers (ViT), évaluation clinique, extension IRM/PET-scan.

## Statut

Présentation **non formellement reçue** — jugée incomplète par Dr LACMOU. À compléter avant la prochaine séance.
