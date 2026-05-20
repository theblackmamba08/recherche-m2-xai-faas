---
type: fiche-lecture
article: 2017_Sundararajan_IntegratedGradients.pdf
auteurs: Mukund Sundararajan, Ankur Taly, Qiqi Yan
annee: 2017
journal_conf: ICML 2017, Sydney — Google
date_lecture: 2026-05-20
pertinence: 3
tags: [integrated-gradients, attribution, post-hoc, gradient, axiomatic, h2-alternative]
---

# Axiomatic Attribution for Deep Networks

> Citation BibTeX-key : `Sundararajan2017IG` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🔢 **Référence pour les méthodes d'attribution par gradient.** Integrated Gradients (IG) est l'alternative post-hoc la plus rigoureuse aux saliency maps. À citer pour situer H1 par rapport aux méthodes gradient-based, et comme piste optionnelle de validation complémentaire.

## Problème traité

Les méthodes de saliency basées sur les gradients (gradient simples, guided backprop) ont deux défauts :
1. **Insensibilité** : si un neurone est saturé, le gradient est nul même si la feature est importante.
2. **Implémentation bias** : les gradients mesurent la sensibilité locale, pas la contribution globale.

Integrated Gradients propose une attribution axiomatiquement fondée qui résout ces problèmes.

## Méthode

Attribution d'une feature `i` pour une prédiction `F(x)` par rapport à une baseline `x'` (souvent l'entrée nulle) :
```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
```

En pratique, approximé par une somme de Riemann sur k=50 pas.

**Axiomes satisfaits** :
- *Sensitvity* : si F(x) ≠ F(x') et x_i ≠ x'_i alors IG_i ≠ 0.
- *Implementation invariance* : deux réseaux fonctionnellement équivalents donnent les mêmes attributions.
- *Completeness* : Σ_i IG_i = F(x) - F(x').
- *Dummy* : features sans impact → attribution nulle.
- *Linearity* et *symmetry-preserving*.

## Résultats clés

- Attributions visuellement plus cohérentes que gradient simple sur ImageNet.
- Identifie correctement les pixels discriminants sur des modèles médicaux.
- Très utilisé via Captum (PyTorch) pour l'attribution sur des modèles Transformer.

## Lien avec H1

**Post-hoc par définition** : IG calcule la contribution de chaque feature via un back-pass (ou plusieurs). Notre Evidence Layer est intrinsèque — pas besoin de IG.

Utilité potentielle pour H1 :
- **Validation complémentaire** (optionnelle) : calculer IG sur notre modèle B5 sur 1-2 fonctions et comparer les positions importantes identifiées par IG vs argmax(M). Si convergence → argument de *convergent validity*.
- **Slide 35** (perspectives) : IG comme vérification post-hoc optionnelle recommandée aux encadreurs.

**Différence fondamentale** :
| | Evidence Layer (M) | Integrated Gradients |
|---|---|---|
| Type | Intrinsèque | Post-hoc |
| Coût inférence | 0 (déjà calculé) | k forward-backward passes |
| Fidélité | Par construction | Approximée (baseline dépendante) |
| Axiomes | Linéarité exacte | Satisfaits sous hypothèses |

## Citations à réutiliser

> "Integrated gradients is the unique path method that is both sensitivity-preserving and implementation invariant." (Theorem 1)

> "Attributions sum to the difference between the output of F at the input x and the baseline x'." (Proposition 1 — Completeness)

## Idées à creuser

- Captum (PyTorch) propose une implémentation clé en main d'IG. Si on veut faire une validation complémentaire, 2-3 heures de code suffisent sur notre modèle.
- La baseline `x'` pour des séries temporelles est non triviale : zéros ? Moyenne ? À réfléchir si on lance ce test.
