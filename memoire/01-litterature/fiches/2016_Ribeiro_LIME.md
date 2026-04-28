---
type: fiche-lecture
article: 2016_Ribeiro_LIME.pdf
auteurs: Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
annee: 2016
journal_conf: KDD 2016, San Francisco, CA, USA. DOI: 10.1145/2939672.2939778. arXiv:1602.04938v3 [cs.LG] 9 août 2016. University of Washington, Seattle.
date_lecture: 2026-04-28
pertinence: 2
tags: [lime, post-hoc, model-agnostic, local-explanation, linear-surrogate, classification, sp-lime, submodular, reference-fondatrice]
---

# LIME : « Pourquoi devrais-je vous faire confiance ? » — Expliquer les prédictions de n'importe quel classifieur

> Citation BibTeX-key : `Ribeiro2016LIME` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Article fondateur de l'XAI post-hoc model-agnostique**. LIME (2016) est la référence incontournable à citer dans l'état de l'art pour comprendre d'où vient SHAP et pourquoi il l'a supplanté pour les séries temporelles. Pertinence directe pour notre tâche FaaS : faible — LIME n'a pas d'extension temporelle mature. Sa valeur dans notre mémoire est **contextuelle et comparative**.

## Problème traité

Les modèles ML état de l'art (random forests, SVM, réseaux de neurones) sont des boîtes noires. L'évaluation sur un jeu de validation peut surestimer la performance réelle, et les utilisateurs ne peuvent pas décider de faire confiance à une prédiction individuelle sans comprendre le raisonnement du modèle. Les auteurs distinguent deux niveaux de confiance : (1) **faire confiance à une prédiction** individuelle, (2) **faire confiance au modèle** dans son ensemble. LIME adresse les deux.

## Méthode

### LIME — Local Interpretable Model-Agnostic Explanations (Section 3)

**Principe** : apprendre un modèle interprétable `g` (modèle linéaire sparse) qui approche localement le comportement du classifieur boîte noire `f` autour de l'instance à expliquer `x`.

**Formulation formelle** (Eq. 1, p. 3) :
```
ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)
```
où :
- `L(f, g, π_x)` = perte de fidélité locale : `Σ_{z,z'∈Z} π_x(z)(f(z) - g(z'))²`
- `π_x(z) = exp(-D(x,z)²/σ²)` = noyau exponentiel de proximité (instances proches de x ont plus de poids)
- `Ω(g)` = complexité du modèle explicatif (ex : nombre de poids non nuls pour un modèle linéaire)

**Représentations interprétables** (Section 3.1) :
- Texte : vecteur binaire présence/absence des mots
- Image : vecteur binaire présence/absence des super-pixels
- Tabulaire : vecteur binaire présence/absence des features

**Algorithme 1** — Sparse Linear Explanations using LIME :
1. Initialiser Z = {}
2. Pour i=1,...,N : échantillonner z'_i autour de x', obtenir f(z_i), ajouter à Z
3. Retourner w ← K-Lasso(Z, K) avec z'_i comme features, f(z) comme cible

En pratique : N = 5000 pour les random forests. L'explication d'une prédiction via Inception (réseau d'images) prend ~10 minutes sur laptop.

### SP-LIME — Submodular Pick pour la confiance globale (Section 4)

**Problème** : expliquer une seule prédiction ne suffit pas pour évaluer le modèle globalement. SP-LIME sélectionne B instances représentatives et non-redondantes par optimisation sous-modulaire greedy.

Fonction de couverture (Eq. 3, p. 5) :
```
c(V, W, I) = Σ_{j=1}^{d'} I[∃i∈V: W_{ij} > 0] · I_j
```
Le problème de sélection (Eq. 4) est NP-hard mais approché à un facteur `1 - 1/e` du optimal par un algorithme greedy.

### Évaluation expérimentale

**Datasets** (Section 5) :
- 2 datasets d'analyse de sentiment : *books* et *DVDs* (2000 instances chacun)
- 20 newsgroups (classification texte : Christianity vs Atheism)
- Images : réseau Inception de Google (pré-entraîné ImageNet)

**Modèles expliqués** : Decision Trees, Logistic Regression (LR), Nearest Neighbors (NN), SVM, Random Forests (RF).

**Paramètres** : N = 15 000 échantillons, K = 10 features par explication.

## Résultats clés

### Fidélité des explications (Figs. 6-7, p. 7)

Recall sur les features « vraiment importantes » (gold set) :
- LIME : **> 90 %** de recall pour tous les classifieurs sur les deux datasets
- Greedy : 64,3 % (books, Sparse LR) à 33,0 % (Decision Tree)
- Parzen : comparable à greedy, parfois moins bon
- LIME surpasse systématiquement toutes les baselines

### Confiance dans les prédictions (Table 1, p. 7)

F1 moyen de trustworthiness sur Books :

| Méthode | LR | NN | RF | SVM |
|---------|----|----|----|----|
| Random | 14,6 | 14,8 | 14,7 | 14,4 |
| Parzen | 84,0 | 87,6 | 94,3 | 92,3 |
| Greedy | 53,7 | 47,4 | 45,0 | 53,3 |
| **LIME** | **96,6** | **94,5** | **96,2** | **96,7** |

Sur DVDs : LIME = 96,6 / 91,8 / 96,1 / 95,6 → résultats stables.

### Expériences avec humains (Section 6)

- **Sélection de modèle** (§6.2, Fig. 9) : avec SP-LIME, les utilisateurs choisissent le meilleur classifieur dans **89 %** des cas vs 74 % avec Greedy (amélioration de +15 points). LIME seul : 80 % vs 68 %.
- **Feature engineering** (§6.3) : les crowd workers améliorent un classifieur défectueux en ~11 minutes avec SP-LIME. Chaque round = 3,6 min en moyenne. 174/200 mots sélectionnés par SP validés par au moins la moitié des utilisateurs.
- **Husky vs Wolf** (§6.4, Table 2) : LIME révèle que le classifieur utilise la **neige en arrière-plan** comme feature discriminante (pas l'animal).
  - Avant explication : 10/27 faisaient confiance au mauvais modèle, 12/27 identifiaient la neige
  - Après explication : seulement **3/27** font encore confiance, **25/27** identifient la neige → LIME détecte le biais

## Limites identifiées par les auteurs

- Seuls les modèles linéaires sparse sont étudiés comme explicatifs G (autres familles possibles mais non explorées).
- La procédure de pick step pour les images n'est pas complètement adressée.
- Coût computationnel : N=5000 random forest (1000 arbres) = 23 s/laptop ; Inception = 10 min/prédiction.
- Exploration des propriétés théoriques (nombre optimal de samples, optimisations GPU) laissée pour travaux futurs.

## Limites identifiées par MOI (lecteur)

1. **Aucune extension pour les séries temporelles**. LIME traite les features comme indépendantes dans les perturbations — or les séries temporelles ont des dépendances temporelles fortes (autocorrélation, saisonnalité). Perturber `y(t-1)` indépendamment de `y(t-2)` produit des instances irréalistes.
2. **Conçu pour la classification uniquement**. La formulation originale explique une probabilité de classe. Pour la prévision (FAYAM prédit des invocations FaaS), l'adaptation est non triviale.
3. **Instabilité des explications**. LIME étant basé sur l'échantillonnage aléatoire, deux runs sur la même instance peuvent produire des explications différentes — problème de stabilité documenté dans la littérature post-LIME.
4. **Fidélité locale ≠ fidélité globale**. Le modèle linéaire local peut être très fidèle dans la région échantillonnée mais complètement faux à l'échelle du dataset. Pour les Transformers (hautement non-linéaires), cette limite est critique.
5. **Surpassé par SHAP sur toutes ses propriétés**. SHAP (Lundberg & Lee, 2017, l'année suivante) satisfait en plus la **cohérence** et la **missingness** — propriétés que LIME ne garantit pas. SHAP est devenu le standard dans la communauté.
6. **Pas de code spécialisé pour les Transformers ou FaaS**. La librairie `lime` Python existe mais n'a pas d'adaptation native pour les séries temporelles multivariées de prévision.

## Lien avec H1 *(notre hypothèse prioritaire — SoftCAM-Transformer)*

LIME est **la référence fondatrice** à situer dans l'état de l'art, mais **pas un compétiteur direct** pour notre tâche :

| Dimension | LIME | SHAP / TsSHAP / SHAPformer | SoftCAM (H1) |
|-----------|------|--------------------------|--------------|
| Tâche | Classification | Classif. + **Prévision** (TsSHAP) | **Prévision** |
| Séries temporelles | ❌ Non adapté | ✅ Extensions dédiées | ✅ Natif |
| Fondement théorique | Heuristique locale | Théorie des jeux (Shapley) | Régularisation ElasticNet |
| Propriétés garanties | Précision locale | Précision + cohérence + missingness | Fidélité par construction |
| Stabilité | Faible (sampling) | Meilleure | Maximale |

**Rôle dans le mémoire** : LIME est à citer dans le chapitre 2 (état de l'art) comme **point de départ historique de l'XAI post-hoc model-agnostique**, avant d'introduire SHAP comme sa généralisation théoriquement rigoureuse, puis les méthodes SHAP spécialisées pour les séries temporelles (TimeSHAP, TsSHAP, SHAPformer) comme contributions récentes directement pertinentes.

**Dans la présentation** : LIME mérite 1 slide dans le panorama « méthodes post-hoc », avec la transition :
> *« LIME (2016) a posé les bases. SHAP (2017) a apporté les garanties théoriques. TimeSHAP, TsSHAP et SHAPformer ont étendu SHAP aux séries temporelles et aux Transformers — c'est ce paysage que nous devons maîtriser pour contextualiser SoftCAM. »*

## Citations à réutiliser

> « We propose LIME, a novel explanation technique that explains the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction. » (Abstract, p. 1)

> « if the users do not trust a model or a prediction, they will not use it. » (Introduction, p. 1)

> « Local fidelity does not imply global fidelity: features that are globally important may not be important in the local context, and vice versa. » (p. 3)

> « LIME consistently provides > 90% recall for both classifiers on both datasets, demonstrating that LIME explanations are faithful to the models. » (§5.2, p. 7)

## Idées à creuser

- **Positionner LIME dans la chronologie XAI** : LIME (2016) → SHAP (2017) → KernelSHAP → TimeSHAP (2021) → TsSHAP (2023) → SHAPformer (2025). Cette progression est le fil narratif naturel du chapitre 2.
- **Différence clé LIME vs SHAP** : LIME optimise un modèle local heuristiquement ; SHAP calcule la solution théoriquement unique (valeur de Shapley) qui satisfait 3 axiomes. C'est l'argument principal pour préférer SHAP dans notre travail.
- **Mention LIME dans la présentation** : 1 slide max, suffisant pour montrer la maîtrise du domaine sans creuser les détails — les encadreurs attendent surtout la justification du choix SHAP.
