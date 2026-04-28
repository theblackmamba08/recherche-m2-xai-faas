---
type: fiche-lecture
article: 2023_Raykar_TsSHAP.pdf
auteurs: Vikas C. Raykar, Arindam Jati, Sumanta Mukherjee, Nupur Aggarwal, Kanthi Sarpatwar, Giridhar Ganapavarapu, Roman Vaculin
annee: 2023
journal_conf: ACM Conference on Knowledge Discovery and Data Mining (Conference'22, août 2022, Washington DC). arXiv:2303.12316v1 [cs.LG] 22 Mar 2023. IBM Research Bangalore + Yorktown.
date_lecture: 2026-04-28
pertinence: 5
tags: [tsshap, shap, treeshap, post-hoc, model-agnostic, forecasting, time-series, surrogate, univariate, faas-applicable, h2-repli-prioritaire]
---

# TsSHAP : Explicabilité robuste model-agnostique basée sur les features pour la prévision de séries temporelles univariées

> Citation BibTeX-key : `Raykar2023TsSHAP` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Article le plus directement pertinent pour notre tâche**. TsSHAP est le seul des articles SHAP de cette liste conçu nativement pour la **prévision** (forecasting), pas pour la classification. FAYAM prédit des invocations FaaS (régression multi-horizons) — TsSHAP est la méthode post-hoc naturelle si H1 (SoftCAM-Transformer) échoue.

## Problème traité

La littérature XAI pour les séries temporelles s'est concentrée sur la **classification** (TimeSHAP, WindowSHAP, ShaTS). La prévision (forecasting) reste sous-étudiée en matière d'explicabilité malgré son omniprésence industrielle (retail, finance, santé, IoT). Les méthodes existantes ne gèrent pas : (1) la notion de features *interprétables* définies a priori par l'expert (lags, saisonnalités, tendances), (2) les trois portées d'explication utiles en pratique (locale / semi-locale / globale), (3) la robustesse des explications face aux perturbations de la série.

TsSHAP propose un cadre model-agnostique qui réduit le problème d'explicabilité de la prévision à un problème de **régression supervisée sur un modèle surrogate**, puis applique TreeSHAP pour extraire les importances de features.

## Méthode

### Vue d'ensemble (pipeline en 3 étapes)

```
Série originale y(1)...y(T)
    ↓ Backtesting (expanding window)
Prévisions historiques ŷ(T+h|T) pour h=1,...,H
    ↓ Features interprétables x(t)
Modèle surrogate g: x → ŷ  [XGBoost]
    ↓ TreeSHAP sur g
Explications SHAP par feature
```

### 1. Modèle surrogate (Section 4.1)

TsSHAP est model-agnostique : il ne pénètre pas dans les internals du forecaster. Il entraîne un **modèle surrogate** `g` qui apprend à imiter les prévisions du forecaster original `f̂`. Le surrogate est un **XGBoost** (tree-based regressor) car TreeSHAP est exact et polynomial en temps pour les arbres.

### 2. Données d'entraînement du surrogate — Backtesting (Section 4.2)

Pour générer les paires `(x(t), ŷ(T+h|T))` d'entraînement, les auteurs utilisent le **backtesting par fenêtre expansive** (expanding window, Fig. 2) : on rejoue la prévision sur toutes les partitions train/test historiques, et on concatène toutes les prévisions sur les splits de test.

### 3. Features interprétables (Section 4.4, Tables 1-2)

Sept familles de features, toutes calculables a priori sur la série :

| Famille | Exemples |
|---------|---------|
| `LagFeatures(lags=3)` | `sales(t-1)`, `sales(t-2)`, `sales(t-3)` |
| `SeasonalLagFeatures(lags=2, m=365)` | `sales(t-365)`, `sales(t-2·365)` |
| `RollingWindowFeatures(window=3)` | `sales-max(t-1,t-3)` |
| `ExpandingWindowFeatures()` | `sales-mean(0,t-1)` |
| `TrendFeatures(degree=2)` | `t`, `t²` |
| `DateFeatures()` | month, day-of-year, week-of-year, is-weekend, season... |
| `HolidayFeatures(country)` | jours fériés nationaux |

→ **Pour FaaS** : les lags (invocations passées), saisonnalités journalières/hebdomadaires, et features de date/heure sont directement transposables.

### 4. Multi-horizons : prévision récursive (Section 4.5)

Un seul surrogate est entraîné pour h=1. Pour h=2,...,H, on appelle le surrogate récursivement :
```
g(T+1|T) = G(y(1),...,y(T))
g(T+h|T) = G(y(1),...,y(T), g(T+1|T),...,g(T+h-1|T))
```

### 5. Trois portées d'explication (Section 5.1)

- **Locale** : explique la prévision à un instant t précis. *Exemple : « Pourquoi la prévision du 1er juillet 2019 est-elle élevée ? »*
- **Globale** : explique le comportement du forecaster sur toute la série historique. *Exemple : « Quelles features le modèle utilise-t-il systématiquement ? »*
- **Semi-locale** : agrège les explications locales sur un intervalle de l'horizon. *Exemple : « Pourquoi les 4 prochaines semaines sont-elles prévues à la hausse ? »*

### 6. Métriques d'évaluation des explications (Section 6.4)

- **Fidélité** `μ_F` : corrélation entre le changement de prévision et le changement d'importance feature quand on perturbe la série. Plus élevé = meilleur.
- **Sensibilité** `μ_S` : variation des explications sous perturbation insignifiante de la série. Plus faible = meilleur (explications robustes).
- **Complexité** `μ_C` : entropie de la distribution des importances feature. Plus faible = explication plus concentrée, plus lisible.

Robustesse testée via **block bootstrap** (préserve la structure trend-cycle) :
```
ỹ_residual = Block-Bootstrap(y_residual, L_b)
ỹ = y_trend-cycle + ỹ_residual
```

## Résultats clés

### Fidélité des explications (Table 6, Fig. 4)

- Les explications **semi-locales et locales sont toujours plus fidèles que globales** (Fig. 4a).
- `MovingAverage(k=6)` atteint la fidélité globale moyenne la plus haute : **0,244** (global), **0,654** (local), **0,658** (semi-local).
- `XGBoost` (forecaster complexe) : fidélité 0,00–0,36 (global), sensibilité 0,07–217,22 — plus complexe = explications moins stables.

### Complexité stable

- `μ_C` (complexité) varie peu entre les portées d'explication (Fig. 4c) → le nombre de features importantes ne dépend pas du scope.
- `Naive` obtient la complexité la plus faible (0,074 global, 0,086 local) car ses explications sont naturellement sparses.

### Illustrations qualitatives (Figs. 5-7, Section 7.3)

- **SeasonalNaive** : la feature dominante est `sales(t-52)` (lag saisonnier 52 semaines) → cohérent avec la logique d'un forecaster saisonnier.
- **Prophet** : `discount(t)` (régresseur externe) est la principale feature positive ; `year` et `sales_max(t-1,t-6)` poussent à la baisse.
- **XGBoost** : `sales(t-1)` et `discount(t)` dominent en local et global.

### Précision du surrogate (Table 5, Section 9.2.1)

- La précision du surrogate **diminue avec la complexité du forecaster** (attendu).
- Sur `jeans-sales-daily`, `MovingAverage(k=6)` : MAE=347,26, RMSE=359,37, MAPE=0,10.
- Sur `peyton-manning`, `XGBoost` : MAE=0,96, RMSE=1,16, MAPE=0,12.

## Limites identifiées par les auteurs

- **Univarié uniquement** : la méthode est conçue pour `f(t) : ℤ → ℝ¹`. Extension au multivarié est mentionnée comme travail futur (Section 8).
- **Accès à `fit()` requis** pour certains forecasters classiques (SARIMA, Exponential Smoothing) — pas pour les modèles ML pré-entraînés.
- **Surrogate = XGBoost** : le choix est justifié par TreeSHAP, mais XGBoost peut mal imiter des forecasters non-linéaires complexes (Transformers).

## Limites identifiées par MOI (lecteur)

1. **Univarié uniquement — problème direct pour FaaS**. FAYAM est entraîné sur **18 fonctions simultanément** (multivarié multi-séries). TsSHAP devrait être appliqué indépendamment à chaque série d'invocations — perte de la vue globale inter-fonctions.
2. **Surrogate XGBoost ≠ Transformer**. Le surrogate apprend à imiter le Transformer FAYAM, mais ses explanations reflètent ce que XGBoost a capturé de ce comportement — pas le mécanisme interne du Transformer. C'est une indirection qui peut introduire des biais d'approximation.
3. **Pas de comparaison avec des méthodes intrinsèques** (SoftCAM, architectures self-explainable). La seule baseline est d'autres méthodes post-hoc.
4. **Features interprétables à définir manuellement**. L'expert doit choisir les familles de features pertinentes — pour FaaS, les lags saisonniers (journalier, hebdomadaire) sont évidents, mais d'autres patterns (pics d'événements, cascades de fonctions) sont moins faciles à anticiper.
5. **Pas de code public clair**. L'article ne mentionne pas de dépôt GitHub — à vérifier avant de choisir TsSHAP comme H2.

## Lien avec H1 *(notre hypothèse prioritaire — SoftCAM-Transformer)*

TsSHAP est la **meilleure option post-hoc disponible pour notre tâche de prévision**, et de loin :

| Critère | TsSHAP | TimeSHAP | WindowSHAP | SoftCAM-Transformer (H1) |
|---------|--------|----------|------------|--------------------------|
| Tâche | **Prévision** ✅ | Classification | Classification | Prévision ✅ |
| Modèle cible | Agnostique | RNN/GRU | RNN/GRU | Transformer (modifié) |
| Features interprétables | Définies a priori ✅ | Brutes | Brutes | Carte d'évidence |
| Scopes | Local/Semi-local/Global ✅ | Local/Global | Local/Global | Local/Global |
| Coût inférence | Surrogate + TreeSHAP | Perturbations | Perturbations | 1 forward pass |
| Fidélité garantie | Non (surrogate approx.) | Non | Non | Oui (par construction) |

**Si H1 échoue** : TsSHAP est le premier repli à déployer. Il suffit d'entraîner un surrogate XGBoost sur les prévisions backtestées du Transformer FAYAM, définir les features (lags FaaS journaliers/hebdomadaires, features de date/heure), et appliquer TreeSHAP.

**Outil de validation H1** : les features les plus importantes selon TsSHAP (ex: `invocations(t-7)` = lag hebdomadaire) devraient correspondre aux instants fortement activés dans la carte d'évidence SoftCAM — si oui, H1 est cohérent avec une baseline post-hoc robuste.

**Adaptation multivarié nécessaire** : pour couvrir les 18 fonctions simultanément, envisager d'agréger les explications TsSHAP par fonction et par cluster DBSCAN — ce qui rejoint directement notre plan H3 (profils de charge DBSCAN).

## Citations à réutiliser

> « Although various flavors of explainability have been well-studied in supervised learning paradigms like classification and regression, literature on explainability for time series forecasting is relatively scarce. » (Abstract, p. 1)

> « TsSHAP can explain the forecast of any black-box forecasting model. The method is agnostic of the forecasting model being explained, and can provide explanations for a forecast in terms of interpretable features defined by the user a priori. » (Abstract, p. 1)

> « The novel TsSHAP methodology uniquely reduces the surrogate forecasting task into a regression problem where the surrogate model learns a mapping between the interpretable feature space and the forecast of the black-box model. » (Section 1.1, p. 2)

> « The TsSHAP explanations are more faithful in the local and semi-local scopes. » (Conclusion, p. 8)

## Idées à creuser

- **Définir les features FaaS** : adapter les familles de features au contexte FaaS — lags à 1h, 6h, 24h, 7j, rolling max (pics d'invocations), features calendaires (heure, jour de la semaine, is-weekend, holiday).
- **Entraîner le surrogate** : utiliser le backtesting du Transformer FAYAM déjà entraîné pour générer les cibles — accès uniquement à `predict()`, pas besoin de réentraîner.
- **Portée semi-locale = horizon de prédiction** : FAYAM prédit sur un `predictionLength` fixe — la portée semi-locale naturelle est exactement cet horizon.
- **Comparer fidélité H1 ↔ TsSHAP** : si SoftCAM-Transformer (H1) aboutit, appliquer TsSHAP sur les mêmes instances et comparer les features importantes — validation croisée des deux approches.
- **Vérifier disponibilité du code** : chercher un dépôt GitHub associé à `arXiv:2303.12316` avant de valider TsSHAP comme H2 opérationnel.
