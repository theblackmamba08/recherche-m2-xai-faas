---
type: fiche-lecture
article: 2017_Lundberg_SHAP.pdf
auteurs: Scott M. Lundberg, Su-In Lee
annee: 2017
journal_conf: NeurIPS 2017, Long Beach — University of Washington
date_lecture: 2026-05-20
pertinence: 3
tags: [shap, shapley, post-hoc, local-explanation, model-agnostic, h2-alternative]
---

# A Unified Approach to Interpreting Model Predictions

> Citation BibTeX-key : `Lundberg2017SHAP` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎲 **Référence fondatrice de SHAP.** Unifie LIME, DeepLIFT, LIME et Shapley values dans un seul cadre théorique. À citer pour positionner H1 par rapport à la famille SHAP (H2 dans notre stratégie).

## Problème traité

Plusieurs méthodes d'explication locale existent (LIME, DeepLIFT, layer-wise relevance propagation) mais elles ne sont pas comparables entre elles et ont des propriétés différentes. L'objectif est de proposer un cadre unifié avec des garanties théoriques.

## Méthode

**SHAP values** = valeurs de Shapley adaptées au ML.

Pour un modèle `f` et une prédiction `f(x)`, la SHAP value φ_i de la feature i est :
```
φ_i = Σ_{S ⊆ F\{i}} |S|!(|F|-|S|-1)!/|F|! × [f(S∪{i}) - f(S)]
```

En pratique : espérance sur toutes les coalitions possibles de features, de la contribution marginale de la feature i.

**Propriétés théoriques (Shapley axioms)** :
- *Efficiency* : Σ φ_i = f(x) - E[f].
- *Symmetry* : features équivalentes → même attribution.
- *Dummy* : feature sans impact → φ_i = 0.
- *Additivity* : linéaire sur les ensembles de features.

**Variantes** :
- **KernelSHAP** : model-agnostic (coûteux).
- **TreeSHAP** : efficace pour arbres (exact, O(TLD²)).
- **DeepSHAP** : pour réseaux de neurones profonds.

## Résultats clés

- SHAP unifie sous un seul cadre : LIME, DeepLIFT, mean centering.
- TreeSHAP permet des attributions exactes en temps polynomial pour les forêts.
- Utilisé massivement en pratique (bibliothèque `shap` Python, intégré Scikit-learn, XGBoost, LightGBM).

## Lien avec H1

**SHAP = H2 dans notre stratégie de repli.** Si H1 avait échoué, on aurait appliqué TimeSHAP (Bento 2021) ou TsSHAP (Raykar 2023) sur le modèle FAYAM baseline.

SHAP est post-hoc par définition — il approxime la contribution des features *après* l'entraînement. Notre M est intrinsèque — elle est calculée dans le forward pass.

**Différence fondamentale** :
| | Evidence Layer (M) | SHAP |
|---|---|---|
| Type | Intrinsèque | Post-hoc |
| Fidélité | Par construction | Approximée (marginalisation sur distribution) |
| Coût | 0 (déjà calculé) | 2^N évaluations (exponentielles) |
| Grain temporel | Pas historique × pas futur | Par feature, pas par step (TimeSHAP adapte) |

À citer en slide 4 (panorama XAI) pour positionner la famille SHAP.

## Citations à réutiliser

> "We present SHAP (SHapley Additive exPlanations), a unified framework for interpreting predictions." (Abstract)

> "The Shapley value is the unique solution satisfying efficiency, symmetry, dummy, and additivity properties." (Theorem 1)

> "SHAP values represent the marginal contribution of each feature to the prediction." (Section 3)

## Idées à creuser

- Si on fait une validation complémentaire, KernelSHAP sur 1-2 fonctions de B5 donnerait une attribution post-hoc comparable à argmax(M). Convergence ou divergence ? Argument pour ou contre M.
- TimeSHAP (Bento 2021, déjà fiché) adapte SHAP aux séries temporelles — c'est l'extension directe à notre contexte.
