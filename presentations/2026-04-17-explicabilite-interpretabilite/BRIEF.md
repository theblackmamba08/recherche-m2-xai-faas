# BRIEF — Explicabilité et Interprétabilité en IA : Fondations, Méthodes et Enjeux Éthiques

> Rempli a posteriori (présentation déjà réalisée).

- **Date** : 2026-04-17
- **Audience** : Pr KENGNE TCHENDJI Vianney, Dr LACMOU ZEUTOUO Jerry (séance en présentiel — 2 encadreurs)
- **Durée** : non précisée
- **Format** : Beamer (24 diapositives)

## Objectif

Présentation approfondie de l'explicabilité et l'interprétabilité en IA : fondations conceptuelles, taxonomie des méthodes XAI, évaluation, cas pratiques et limites. Première présentation qui traite XAI de façon rigoureuse et structurée, avec références et cas d'usage réels.

## Points clés présentés

1. **Problème de la boîte noire** : paradoxe de la performance — plus un modèle est performant, plus il est opaque. Risques : biais invisibles, manque de redevabilité, erreurs indétectables.
2. **Pourquoi XAI est nécessaire** : 4 piliers — Confiance, Causalité, Éthique & Justice, Conformité réglementaire (RGPD, finance).
3. **Interprétabilité vs Explicabilité** : interprétabilité = mécanique interne (le "comment", pour experts) ; explicabilité = communication des raisons (le "pourquoi", pour utilisateurs finaux). Analogie mécanicien/conducteur.
4. **Transparence intrinsèque vs Post-hoc** : modèles intrinsèquement interprétables (arbres, régression) vs méthodes post-hoc (SHAP, LIME, Grad-CAM). Limite post-hoc : risque d'infidélité.
5. **Classification XAI** : par portée (globale/locale), dépendance au modèle (agnostique/spécifique), résultat produit (importance features, heatmaps, règles…).
6. **LIME** (Ribeiro et al., 2016) : approximation locale par modèle simple. Rapide, intuitif, modèle-agnostique. Limite : instabilité.
7. **SHAP** (Lundberg & Lee, 2017) : valeurs de Shapley, fondement théorique solide, cohérence globale/locale. Limite : coût computationnel.
8. **Comparaison LIME vs SHAP** : tableau complet — fondement, vitesse, stabilité, propriétés mathématiques.
9. **Grad-CAM** (Selvaraju et al., 2017) : carte de chaleur via gradients dernière couche convolutive. Applications : vision, médical, véhicules autonomes.
10. **4 principes NIST** pour une IA explicable : Explication, Signification, Précision, Limites de connaissances.
11. **Évaluation** : humaine (gold standard) vs métriques informatiques (fidélité, robustesse, complétude). Compromis : explication fidèle ≠ toujours compréhensible.
12. **Cas pratique — mélanome** : Grad-CAM vs dermatologues, 93,6% d'accord. Leçon : évaluation humaine indispensable.
13. **Enjeux réglementaires** : BIS (banques), RGPD (droit à l'explication, droit de contester).
14. **Mythes de l'interprétabilité** (Lipton, 2018) : modèles simples ≠ toujours interprétables ; interprétabilité non monolithique ; explications post-hoc ≠ toujours fidèles (fairwashing).
15. **Limites XAI** : trade-off performance/interprétabilité, fairwashing, absence de standardisation, coût post-hoc, LLMs difficiles à expliquer.
16. **Interprétabilité mécaniste** : rétro-ingénierie des réseaux pour comprendre circuits internes — nouvelle frontière (LLMs, détection de biais).

## Demandes à l'audience

Validation du cadre XAI et orientation vers une contribution concrète.
