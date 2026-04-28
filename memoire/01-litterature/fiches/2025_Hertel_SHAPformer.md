---
type: fiche-lecture
article: 2025_Hertel_SHAPformer.pdf
auteurs: Matthias Hertel, Sebastian Pütz, Ralf Mikut, Veit Hagenmeyer, Benjamin Schäfer
annee: 2025
journal_conf: Preprint arXiv:2512.20514v1 [cs.LG] 23 déc. 2025 (under review). Institute for Automation and Applied Informatics (IAI), Karlsruhe Institute of Technology (KIT), Allemagne. Code : https://github.com/KIT-IAI/SHAPformer
date_lecture: 2026-04-28
pertinence: 5
tags: [shapformer, shap, transformer, sampling-free, attention-manipulation, forecasting, time-series, exact-shap, post-hoc, h2-repli-bonus, multivariate]
---

# SHAPformer : Prévision de séries temporelles explicable avec SHAP sans échantillonnage pour les Transformers

> Citation BibTeX-key : `Hertel2025SHAPformer` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Doublement pertinent pour notre projet** : SHAPformer est à la fois conçu pour les **Transformers** (architecture de FAYAM) ET pour la **prévision** (tâche de FAYAM). Il calcule des valeurs SHAP **exactes** en moins d'une seconde via manipulation d'attention — sans échantillonnage, sans données de fond. Conceptuellement proche de H1 (SoftCAM-Transformer) mais reste post-hoc.

## Problème traité

Les méthodes SHAP standard pour les Transformers de prévision souffrent de deux défauts majeurs (p. 2-3) :
1. **Coût computationnel prohibitif** : le Permutation Explainer nécessite ~4,3M d'évaluations du modèle pour expliquer une seule prédiction sur le dataset TransnetBW (168×7 features passées + 168×6 covariables futures) → 484 secondes par explication.
2. **Contrefactuels irréalistes** : l'échantillonnage Monte Carlo sur fond de données produit des entrées hors-distribution (ex : lundi à minuit suivi de vendredi à midi), conduisant à des explications arbitraires.

SHAPformer résout les deux en **manipulant les poids d'attention** pour exclure les groupes de features absents, éliminant ainsi tout besoin d'échantillonnage.

## Méthode (Section 4.2)

### Principe : attention manipulation + feature grouping

SHAPformer groupe les features d'entrée en **N groupes** et entraîne le Transformer avec des **entrées masquées** (masked inputs). À chaque itération d'entraînement, un sous-ensemble aléatoire de groupes est masqué via manipulation des poids d'attention → le modèle apprend à prédire à partir de n'importe quel sous-ensemble de features.

À l'inférence, on évalue le modèle pour **tous les 2^N sous-ensembles** de groupes de features → calcul exact des valeurs SHAP sans échantillonnage.

```
SHAP(v_i) = Σ_{S ⊆ V\{v_i}} [(n-1-|S|)!·|S|! / n!] · (f(S ∪ {v_i}) - f(S))
```

La contribution marginale `f(S ∪ {v_i}) - f(S)` est calculée **exactement** par deux évaluations du modèle, sans Monte Carlo.

### Setup expérimental

**Groupement des features** (TransnetBW) : 7 groupes passés (une journée chacun : day1–day7) + 6 covariables futures (day of week, hour of day, month, holiday, temperature, precipitation) = **N = 13 groupes** → 2^13 = 8192 évaluations.

**Comparaisons** :
- Permutation Explainer (approx. SHAP par permutation + Monte Carlo)
- Custom Masker (groupement de features + Monte Carlo, sans manipulation d'attention)
- Temporal Fusion Transformer / TFT (importance via feature selection layer — pas SHAP)

### Datasets (Section 2)

| Dataset | Type | Taille | Horizon | Features |
|---------|------|--------|---------|---------|
| Synthétique | Générée avec vérité terrain connue | 100K train, 10K val+test | 168h | Saisonnalités daily/weekly/annual + holidays + multiplicateur + 2 bruits |
| TransnetBW | Charge électrique horaire TSO allemand, 2015-2019 | ~4 ans | 168h (1 semaine) | Charge passée (7j) + météo ERA5 (temp., précipitations) |

## Résultats clés

### Performance prédictive (Table 1, p. 3)

| Modèle | RMSE synthétique | RMSE réel [MW] | Inférence / explication |
|--------|-----------------|----------------|------------------------|
| Transformer | 0,059 | 263,1 | 0,01 s |
| + Permutation Explainer | — | — | **1124,16 s** (synth) / **484,34 s** (réel) |
| + Custom Masker | — | — | 7,84 s / 3,54 s |
| **SHAPformer** | **0,060** | **265,9** | **21,90 s** (synth) / **0,60 s** (réel) |

- SHAPformer est **50× plus rapide** que le Permutation Explainer sur données synthétiques.
- SHAPformer est **800× plus rapide** sur données réelles.
- Précision de prévision préservée : RMSE SHAPformer ≈ RMSE Transformer (écart de ~1 %).

### Validation sur données synthétiques (Fig. 2, p. 6)

- L'importance globale des features SHAPformer est **proche de la vérité terrain** (Fig. 2A — barres vertes).
- Permutation Explainer et Custom Masker dévient davantage, surtout pour `load` et `hour of day`.
- SHAPformer filtre efficacement les 2 features bruit (importance ≈ 0).
- Les courbes de dépendance SHAPformer (Fig. 2B) reproduisent fidèlement la vérité terrain (Fig. 2C).

### Insights sur données réelles TransnetBW (Fig. 3, p. 7)

- Feature la plus importante identifiée : **charge passée** (≈47 % SHAPformer vs ≈10 % Permutation Explainer — ce dernier sous-estime car l'échantillonnage hors distribution casse la corrélation temporelle).
- Ensuite : `day of week` (≈22 %), `hour of day` (≈20 %), `month`, `temperature`, `holiday`.
- `Precipitation` : importance minimale.
- **Pattern décembre** : le modèle apprend un comportement distinct — charge de base réduite, mais jours non-fériés en semaine sont des exceptions (Fig. 4).

### Dépendances apprises (Fig. 3B)

- `Load` : relation linéaire charge passée → SHAP ✓
- `Hour of day` : demi-sinusoïde (2 pics : midi + soirée) ✓
- `Day of week` : dimanches = SHAP plus bas ✓
- `Temperature` : charge augmente en jours froids (<15°C jour ou <0°C nuit) ✓

## Limites identifiées par les auteurs

- **Complexité exponentielle en N** : runtime ∝ 2^N groupes de features. Avec de nombreux groupes (longues séquences), l'approche exhaustive devient impraticable → les auteurs proposent d'évaluer seulement un sous-ensemble de coalitions (runtime linéaire en N) comme adaptation.
- **Temps d'entraînement accru** : 2 à 10× plus long que le Transformer standard (dû au masquage itératif).
- **Nécessite une modification architecturale** : impossible d'appliquer à un Transformer déjà entraîné — il faut réentraîner avec masked inputs.
- Validé sur 2 datasets uniquement (charge électrique).

## Limites identifiées par MOI (lecteur)

1. **Réentraînement requis — mais pas de modification architecturale**. SHAPformer ne change pas la structure du Transformer (pas de nouvelle couche). Il modifie la *procédure d'entraînement* (masquage des poids d'attention par groupes de features à chaque batch) et la *procédure d'inférence SHAP* (évaluation sur tous les sous-ensembles de groupes via masquage d'attention). On ne peut donc pas l'appliquer à un Transformer déjà entraîné — il faut réentraîner. À distinguer de H1 (SoftCAM) qui, lui, ajoute réellement une couche d'évidence et modifie la structure.
2. **2^N évaluations — problème pour FaaS multivarié**. Si FAYAM a de nombreux groupes de features (18 fonctions × lags multiples), N peut être grand → le budget GPU peut exploser. La version linéaire (sous-ensemble de coalitions) approxime alors SHAP, perdant l'avantage de l'exactitude.
3. **Toujours post-hoc conceptuellement**. SHAPformer explique les contributions de features mais ne modifie pas la manière dont le modèle prend ses décisions — contrairement à H1 (SoftCAM) qui intègre l'explication dans la prédiction elle-même.
4. **Preprint** (déc. 2025, under review) : méthode récente, non encore validée par peer review complet — à signaler dans la section état de l'art.
5. **Cas univarié → multivarié testé mais limité** : le dataset TransnetBW est multivarié (charge + météo), mais toutes les séries sont pour une seule région. Pour FAYAM (18 fonctions FaaS différentes), la définition des groupes de features est non triviale.

## Lien avec H1 *(notre hypothèse prioritaire — SoftCAM-Transformer)*

SHAPformer occupe une position **intermédiaire entre H1 et H2** :

| Dimension | SHAPformer | TsSHAP (H2) | SoftCAM-Transformer (H1) |
|-----------|-----------|-------------|--------------------------|
| Paradigme | Mod. architecturale + SHAP | Surrogate post-hoc | Mod. architecturale intrinsèque |
| Réentraînement requis | Oui (masked inputs) | Non (surrogate séparé) | Oui (couche d'évidence) |
| Fidélité SHAP | **Exacte** | Approx. (TreeSHAP sur surrogate) | N/A (pas SHAP) |
| Coût inférence | <1 s / explication | ~TreeSHAP (rapide) | **1 forward pass** |
| Applicable au Transformer | **Oui direct** ✅ | Oui (boîte noire) | Oui (si adaptation) |
| Architecture Transformer | Oui (nat.) ✅ | Non spécifique | Oui (H1) ✅ |
| Tâche prévision | **Oui** ✅ | **Oui** ✅ | **Oui** ✅ |

**Si H1 échoue** : SHAPformer est le meilleur repli technique pour notre architecture — plus fidèle que TsSHAP (SHAP exact, pas approximé par un surrogate XGBoost), directement applicable au TimeSeriesTransformer HuggingFace avec adaptation.

**Outil de validation H1** : si H1 aboutit, comparer la carte d'évidence SoftCAM aux valeurs SHAP exactes de SHAPformer sur les mêmes instances. Une forte corrélation validerait que la carte d'évidence apprend les mêmes dépendances que SHAP détecte.

**Lien conceptuel fort avec H1** : la manipulation d'attention de SHAPformer (exclure des groupes de features en masquant les poids d'attention) est mécanistiquement proche de ce que fait H1 (modifier la couche de projection du décodeur pour produire des cartes d'évidence). Les deux opèrent sur l'architecture interne du Transformer, pas sur ses sorties.

**Code disponible** : `https://github.com/KIT-IAI/SHAPformer` — paquet Python publié, directement utilisable.

## Citations à réutiliser

> « SHAPformer generates explanations in under one second, several orders of magnitude faster than the SHAP Permutation Explainer. » (Abstract, p. 1)

> « SHAPformer achieves both: it is true to the model and true to the data. This is made possible by its sampling-free design and training strategy, which builds robustness to absent features. » (Discussion, p. 8)

> « For Transformer models, it is common to use attention weights as proxies for feature importance, visualizing them to highlight influential inputs. However, this practice remains controversial, as the interpretability of attention mechanisms is debated. » (Introduction, p. 2)

> « Existing SHAP algorithms are either true to the model or true to the data. Conditional sampling tends to produce explanations that are true to the data, while off-the-manifold sampling generates explanations that better reflect the model's internal behavior. We argue that SHAPformer achieves both. » (Discussion, p. 8)

## Idées à creuser

- **Définir les groupes de features FaaS** : pour le TimeSeriesTransformer de FAYAM, définir N groupes cohérents — ex. {lag_1j, lag_7j, lag_30j, features_calendaires, covariables_fonction} → viser N ≤ 10 pour que 2^10 = 1024 évaluations restent raisonnables.
- **Adaptation au TimeSeriesTransformer HuggingFace** : vérifier la compatibilité du masquage d'attention avec l'implémentation HuggingFace (`TimeSeriesTransformerModel`) — les poids d'attention sont accessibles via `output_attentions=True`.
- **Comparaison SHAPformer ↔ SoftCAM-Transformer** : si H1 aboutit, cette comparaison devient la contribution comparative principale du mémoire — deux approches de modification architecturale, l'une post-hoc (SHAP exact), l'autre intrinsèque (carte d'évidence).
- **Récupérer et tester le code** : `pip install shapformer` ou cloner `github.com/KIT-IAI/SHAPformer` — tester sur un mini-dataset FaaS dès la phase 1 pour estimer le N optimal.
- **Preprint à citer avec précaution** : mentionner dans le mémoire que l'article est sous review (déc. 2025) — vérifier si accepté avant soumission finale.
