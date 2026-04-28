---
type: fiche-lecture
article: 2023_Nayebi_WindowSHAP.pdf
auteurs: Amin Nayebi, Sindhu Tipirneni, Chandan K. Reddy, Brandon Foreman, Vignesh Subbian
annee: 2023
journal_conf: Journal of Biomedical Informatics 144 (2023) 104438. DOI: 10.1016/j.jbi.2023.104438. Reçu : 10 nov. 2022 ; accepté : 3 juil. 2023.
date_lecture: 2026-04-28
pertinence: 3
tags: [windowshap, shapley, kernelshap, timeshap, post-hoc, model-agnostic, time-series, windowing, computational-efficiency, h2-comparaison]
---

# WindowSHAP : cadre efficace pour expliquer les classifieurs de séries temporelles par valeurs de Shapley

> Citation BibTeX-key : `Nayebi2023WindowSHAP` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Extension de TimeSHAP orientée efficacité computationnelle**. WindowSHAP adresse la même limitation (coût de KernelSHAP sur longues séquences) mais par fenêtrage temporel plutôt que par élagage. Pertinent comme alternative à TimeSHAP pour H2, notamment si les séries FaaS sont longues.

## Problème traité

KernelSHAP n'est pas directement applicable aux séries temporelles pour trois raisons (p. 1-2) : (1) non conçu pour les données à variation temporelle, (2) coût computationnel prohibitif pour les séries longues (croissance exponentielle avec le nombre de features/pas de temps), (3) dépendances temporelles entre pas de temps adjacents rendent les valeurs Shapley individuelles peu fiables. TimeSHAP résout partiellement le problème (2) par élagage, mais suppose que les événements anciens sont moins importants — hypothèse qui peut être fausse (anomalie à un instant lointain). WindowSHAP propose un partitionnement en fenêtres temporelles pour calculer les valeurs Shapley par fenêtre plutôt que par pas de temps, réduisant la complexité tout en restant plus flexible sur la position des pas de temps importants.

## Méthode

### Principe général

WindowSHAP calcule des **valeurs Shapley par fenêtre temporelle** plutôt que par pas de temps individuel. On partitionne la séquence `X ∈ ℝ^{D×L}` (D variables, L pas de temps) en ensembles de fenêtres `Ω^i = {ω^i_1, ω^i_2, ..., ω^i_{w_i}}` pour chaque variable i. La valeur Shapley d'une fenêtre `ω^i_k` est :

```
φ_{ω^i_k} = Σ_{S ⊆ Ω \ {ω^i_k}} [|S|!(|Ω|−|S|−1)! / |Ω|!] · [v_{X*}(S ∪ {ω^i_k}) − v_{X*}(S)]
```

L'importance d'un pas de temps individuel est estimée en distribuant uniformément la valeur Shapley de sa fenêtre.

### Trois algorithmes (Section 3.3)

**1. Stationary WindowSHAP**
- Fenêtres fixes non-chevauchantes de longueur `l`, couvrant toute la séquence.
- Complexité : `O(D · ⌈L/l⌉ · (2^l − 1))`
- Avantage : simple, réduction directe. Limite : les points aux frontières des fenêtres sont mal expliqués.

**2. Sliding WindowSHAP**
- Fenêtres fixes de longueur `l` avec chevauchement (stride `s`). Chaque pas de temps apparaît dans plusieurs fenêtres → sa valeur Shapley est la moyenne des valeurs des fenêtres qui le contiennent.
- Complexité : `O(2D · (L−1)/s · 2^{2l−1})`
- Avantage : atténue l'effet de frontière. Limite : complexité croît avec (L−1)/s.

**3. Dynamic WindowSHAP**
- Fenêtres de longueur variable, déterminées par un algorithme itératif : on part d'une seule fenêtre couvrant toute la séquence, puis on divise les fenêtres dont la valeur Shapley dépasse un seuil `δ`, jusqu'à atteindre `n_w` fenêtres max.
- Complexité : `O(D · n_w · 2^{2n_w−1})`
- Avantage : alloue la granularité là où l'importance est concentrée — pas d'hypothèse sur la position des instants importants.

### Évaluation

- **Datasets** : TRACK-TBI EHR (900 patients, 120 pas de temps, 62 variables), TRACK-TBI physiologique (5816 pts, 360 pas de temps, 8 variables, fréquence 10s), MIMIC-III (22988 pts, 48 pas de temps, 26 variables).
- **Modèles** : 3 RNN (GRU) entraînés sur classification binaire.
- **Baselines** : KernelSHAP, TimeSHAP.
- **Métriques** : perturbation analysis metric (impact d'un pas de temps perturbé sur la loss) + sequence analysis metric (impact d'un segment remplacé par la moyenne).
- **Hyperparamètres** (Table 4, p. 7) :
  - TimeSHAP : `tolerance = 0.05`
  - Stationary WindowSHAP : `l = 20` (TBI EHR), `l = 14` (TBI physio), `l = 5` (MIMIC)
  - Sliding WindowSHAP : `l=15, s=8` (TBI EHR/physio), `l=10, s=6` (MIMIC)
  - Dynamic WindowSHAP : `δ = 0.001·n_w`, `n_w = 14` (TBI EHR), `n_w = 14` (TBI physio), `n_w = 20` (MIMIC)

## Résultats clés

- **Réduction CPU** (Table 5, p. 8) :
  - Sur TBI EHR (L=120, D=62) : KernelSHAP → ~90 GB RAM, ~10⁴ secondes ; Stationary WindowSHAP (l=10 pas adjacents) réduit le CPU de **80 %**.
  - Complexité : `O(D·L²^{L/l})` → `O(D·⌈L/l⌉·2^l)` — la réduction est exponentielle en l.

- **Qualité d'explication** (Fig. 5, p. 8) :
  - WindowSHAP variants **surpassent ou égalent** KernelSHAP et TimeSHAP sur les métriques de perturbation et séquence pour TBI physiologique et MIMIC.
  - Sur TBI EHR : pas de différence significative entre WindowSHAP et TimeSHAP.
  - Dynamic WindowSHAP fourni les meilleures explications globales car il évite l'hypothèse « anciens événements peu importants » de TimeSHAP.

- **Analyse qualitative** (MIMIC-III, Fig. 7-8, p. non numérotée) :
  - Dynamic WindowSHAP concentre les valeurs Shapley sur les derniers pas de temps (logique clinique).
  - KernelSHAP distribue les valeurs sur toute la séquence de façon diffuse.

## Limites identifiées par les auteurs

- Uniquement testé sur la **classification binaire** (outcomes cliniques).
- Validation uniquement sur données **médicales** — généralisation à d'autres domaines (finance, FaaS) non démontrée.
- Évaluation par domain experts difficile à grande échelle — métriques quantitatives adoptées en substitut.

## Limites identifiées par MOI (lecteur)

1. **Classification uniquement, pas de régression**. Comme TimeSHAP, WindowSHAP explique un score de probabilité. Pour FAYAM (prévision de séries temporelles continues), la « quality metric » de référence serait la RMSE ou sMAPE — l'adaptation n'est pas immédiate.
2. **Pas testé sur des architectures Transformer**. Les 3 modèles évalués sont des GRU. Le comportement de WindowSHAP sur un TimeSeriesTransformer est inconnu.
3. **Dynamic WindowSHAP suppose que les instants importants sont groupés**. Pour des séries FaaS avec pics d'invocations éparses, l'hypothèse peut tenir — mais pour des séries avec deux pics distants simultanément importants, l'algorithme peut mal segmenter.
4. **Complexité de Dynamic WindowSHAP croît avec n_w²**. Si n_w est élevé (séquence complexe), le gain sur KernelSHAP peut être limité.
5. **Pas de bibliothèque Python publiée**. Le code est « available online [34] » mais il s'agit d'un lien vers Keras/Python — pas de package installable à la façon de `timeshap` (Feedzai).

## Lien avec H1 *(notre hypothèse prioritaire — SoftCAM-Transformer)*

WindowSHAP est une **alternative à TimeSHAP pour H2**, avec un avantage sur les longues séquences (si les séries FaaS ont des contextes longs — ce qui est probable avec des lags hebdomadaires). La comparaison avec H1 est identique à celle de TimeSHAP :

| Dimension | WindowSHAP | TimeSHAP | SoftCAM-Transformer (H1) |
|-----------|-----------|----------|--------------------------|
| Hypothèse temporelle | Pas d'hypothèse (Dynamic) | Anciens événements peu importants | Aucune (apprise) |
| Coût inférence | Réduit vs KernelSHAP | Réduit par élagage | 1 forward pass |
| Granularité | Fenêtres (configurable) | Événements + features + cellules | Carte continue |
| Applicable Transformer | Oui (boîte noire) | Oui (boîte noire) | Oui (si adaptation) |

**Recommandation d'usage** : si H1 échoue et qu'on opte pour H2, tester **TimeSHAP** en premier (code mieux maintenu, plus cité) puis **Dynamic WindowSHAP** comme alternative si les séquences FaaS sont longues et que l'élagage de TimeSHAP supprime des lags hebdomadaires importants.

## Citations à réutiliser

> « WindowSHAP is based on partitioning a sequence into time windows. Instead of calculating Shapley values for every possible time step and variable combinations, we simply calculate Shapley values for each time window. » (Abstract, p. 1)

> « We show that for time-series data with 120 time steps (hours), merging 10 adjacent time points can reduce the CPU time of WindowSHAP by 80 % compared to KernelSHAP. » (Abstract, p. 1)

> « Dynamic WindowSHAP algorithm focuses more on the most important time steps and provides more understandable explanations. » (Abstract, p. 1)

> « TimeSHAP is developed under the premise that the initial time steps in time-series data are of lesser importance. Consequently, it aggregates the initial time steps and assigns them a single Shapley value. This could be the primary reason behind TimeSHAP's inferior performance. » (Discussion, p. 7)

## Idées à creuser

- **Tester Dynamic WindowSHAP sur les séries FaaS** : si les pics d'invocations sont brefs et non-contigus, Dynamic WindowSHAP devrait mieux les isoler que TimeSHAP.
- **Définir n_w optimal** : calibrer le nombre maximal de fenêtres sur les séries FaaS par validation (grid search sur `δ` et `n_w`).
- **Comparaison H1 ↔ WindowSHAP** : même logique que pour TimeSHAP — corréler la carte d'évidence SoftCAM avec les valeurs Shapley par fenêtre pour valider la cohérence.
- **Background spécifique au cluster** : comme pour TimeSHAP, envisager un background intra-cluster DBSCAN plutôt que la moyenne globale.
