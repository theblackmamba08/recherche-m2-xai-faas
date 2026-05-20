---
type: fiche-lecture
article: 2005_Zou_ElasticNet.pdf
auteurs: Hui Zou, Trevor Hastie
annee: 2005
journal_conf: Journal of the Royal Statistical Society, Series B, 67(2), pp. 301–320 — Stanford University
date_lecture: 2026-05-20
pertinence: 3
tags: [regularization, elastic-net, lasso, ridge, sparsity, h1-loss]
---

# Regularization and Variable Selection via the Elastic Net

> Citation BibTeX-key : `Zou2005ElasticNet` *(à reporter dans `redaction/biblio/refs.bib`)*

> 📐 **Référence fondatrice de l'ElasticNet.** La régularisation ElasticNet utilisée dans notre loss H1 (termes α||M||₁ + β||M||²) est directement issue de ce papier. À citer pour légitimer le choix de régularisation.

## Problème traité

Deux régularisations classiques existent pour la sélection de variables :
- **Lasso** (ℓ₁) : favorise les solutions sparses (certains coefficients exactement à 0) mais instable quand les features sont corrélées.
- **Ridge** (ℓ₂) : stable face aux corrélations mais ne produit pas de sparsité.

L'ElasticNet combine les deux pour bénéficier de leurs avantages respectifs.

## Méthode

Pénalité ElasticNet pour un vecteur de coefficients β :
```
L(β) = ||y - Xβ||² + λ₁||β||₁ + λ₂||β||₂²
```
- `λ₁||β||₁` : terme Lasso → sparsité (zéros exacts), sélection de variables.
- `λ₂||β||₂²` : terme Ridge → stabilité face aux corrélations entre features.

**Propriété "grouping effect"** : contrairement au Lasso qui sélectionne arbitrairement une variable parmi un groupe de variables corrélées, l'ElasticNet tend à sélectionner le groupe entier ou à les pondérer conjointement.

## Résultats clés

- Sur des simulations avec features corrélées, ElasticNet domine Lasso et Ridge en MSE de prédiction.
- En sélection de variables, ElasticNet est plus cohérent que Lasso quand des features redondantes sont présentes.
- Algorithme LARS-EN pour l'entraînement efficace.

## Lien avec H1

Dans notre loss :
```
L = forecast_loss + α·||M||₁ + β·||M||₂² + γ·H(M)
```

- `α·||M||₁` (Lasso) force M à être sparse : la plupart des poids M[t,s] → 0, seuls quelques pas historiques comptent par pas de prédiction.
- `β·||M||₂²` (Ridge) évite que M ne s'effondre sur une seule position (comme en Run B avec max_weight=0.85).
- L'ElasticNet sur M produit des cartes d'évidence parcimonieuses et stables — directement validé par H1.G (97.13% préservation à k=1).

Résultat H1.D (Pearson=0.992 intra-cluster) : l'ElasticNet pousse M vers des patterns stables cluster-level, pas des solutions idiosyncratiques par fonction.

À citer en slide 8 (la loss régularisée).

## Citations à réutiliser

> "The elastic net encourages a grouping effect, where strongly correlated predictors tend to be in or out of the model together." (Abstract)

> "The elastic net penalty combines the lasso and ridge regression penalties, and enjoys the sparse representation of the lasso and the stability of ridge regression." (Introduction)

## Idées à creuser

- Le terme entropie `γ·H(M)` que nous ajoutons à l'ElasticNet est une contribution propre de Djoumessi & Berens 2025 (pas dans Zou & Hastie). Les deux se complètent : ℓ₁ + ℓ₂ contrôlent la magnitude, H(M) contrôle la concentration.
- Dans le contexte temporel, les corrélations temporelles entre positions s voisines sont fortes → le grouping effect de l'ElasticNet est pertinent pour garantir que M capture des plages temporelles cohérentes (pas des positions isolées).
