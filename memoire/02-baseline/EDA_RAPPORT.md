# Rapport EDA — Clusters FaaS (Azure Functions Trace 2019)

> **Auteur** : Cabrel Fosso  
> **Date du run** : 2026-05-02 (08h31)  
> **Source des chiffres** : `code/experiments/eda/eda_results_2026-05-02_08-31.json`  
> **Notebook** : `code/notebooks/EDA_clusters.ipynb`  
> **HTML avec figures** : `code/experiments/eda/EDA_clusters_2026-05-02.html`

---

## 1. Contexte et objectif

La baseline de ce mémoire repose sur le travail de FAYAM (2024), qui utilise un
`TimeSeriesTransformerForPrediction` (HuggingFace) pour prédire le nombre d'invocations
concurrentes de fonctions FaaS (colonne `nbrconc`). Les données proviennent de
l'**Azure Functions Trace 2019** (Microsoft), un jeu de données public de 14 jours de
traces d'exécution de fonctions en production à la minute.

FAYAM a identifié **33 clusters** via HDBSCAN sur ces données. Quatre clusters ont été
retenus pour ce travail : **0, 4, 6 et 8**, représentant trois profils de charge distincts
(forte, moyenne, faible/sparse). L'objectif de cette EDA est de :

1. Caractériser précisément chaque cluster avant tout entraînement
2. Identifier les propriétés statistiques qui conditionnent les choix de prétraitement
3. Évaluer l'adéquation des données avec les hypothèses implicites du Transformer
4. Définir un ordre d'entraînement motivé par la complexité croissante des séries

---

## 2. Description des données

### Format

| Caractéristique | Valeur |
|----------------|--------|
| Source | Azure Functions Trace 2019 (Microsoft Research) |
| Variable cible | `nbrconc` — invocations concurrentes par minute |
| Fréquence | 1 minute |
| Durée | 14 jours = 20 160 pas de temps |
| Clusters retenus | 0, 4, 6, 8 (sur 33 au total) |
| Fonctions total | 19 |
| Format fichier | CSV — 1 ligne = 1 fonction, colonnes = timesteps 1..20160 |

### Répartition par cluster

| Cluster | Fonctions | IDs |
|---------|-----------|-----|
| 0 | 3 | 942, 943, 944 |
| 4 | 5 | 949, 951, 952, 953, 954 |
| 6 | 5 | 138, 139, 140, 143, 144 |
| 8 | 6 | 964, 965, 967, 968, 969, 977 |

### Cohérence interne des clusters

Les fonctions d'un même cluster présentent des profils très similaires (corrélations Pearson
et Spearman > 0.95 pour les clusters 0 et 4, plus faibles pour 6 et 8 en raison de la
zero-inflation). Cette cohérence valide la pertinence du clustering HDBSCAN de FAYAM.

---

## 3. Résultats par axe d'analyse

### 3.1 Statistiques descriptives

| Cluster | Fn | Moyenne | Std | CV (%) | Zéros (%) | Burstiness B |
|---------|-----|---------|-----|--------|-----------|--------------|
| 0 | 942 | 117 529 | 78 396 | 66.7 | 0.0 | −0.200 |
| 0 | 943 | 130 751 | 85 654 | 65.5 | 0.0 | −0.208 |
| 0 | 944 | 117 529 | 78 398 | 66.7 | 0.0 | −0.200 |
| 4 | 949 | 97 | 65 | 66.5 | 0.0 | −0.201 |
| 4 | 951 | 95 | 63 | 66.3 | 0.1 | −0.203 |
| 4 | 952 | 106 | 75 | 70.6 | 0.4 | −0.172 |
| 4 | 953 | 90 | 60 | 65.8 | 0.5 | −0.206 |
| 4 | 954 | 98 | 69 | 70.4 | 0.7 | −0.174 |
| 6 | 138 | 2.0 | 13.5 | 657.9 | 90.1 | +0.736 |
| 6 | 139 | 2.0 | 13.5 | 659.5 | 90.1 | +0.737 |
| 6 | 140 | 1.6 | 13.1 | 808.6 | 95.2 | +0.780 |
| 6 | 143 | 1.7 | 13.2 | 783.1 | 94.9 | +0.774 |
| 6 | 144 | 3.1 | 16.5 | 536.3 | 89.4 | +0.686 |
| 8 | 964 | 4.8 | 5.4 | 112.2 | 20.7 | +0.058 |
| 8 | 965 | 5.1 | 5.7 | 113.2 | 24.4 | +0.062 |
| 8 | 967 | 5.0 | 5.7 | 115.1 | 25.6 | +0.070 |
| 8 | 968 | 4.5 | 5.4 | 119.1 | 28.2 | +0.087 |
| 8 | 969 | 4.5 | 5.4 | 119.1 | 28.1 | +0.087 |
| 8 | 977 | 4.6 | 5.4 | 118.0 | 27.4 | +0.083 |

**Lecture du Burstiness B = (σ − μ) / (σ + μ)** :
- B < 0 : série plus régulière qu'un processus de Poisson (clusters 0 et 4)
- B ≈ 0 : comportement proche de Poisson (cluster 8)
- B > 0 : série plus irrégulière / bursty que Poisson (cluster 6 — très marqué)

### 3.2 Stationnarité (test ADF)

**Résultat : 19/19 fonctions stationnaires** (test de Dickey-Fuller augmenté, p ≈ 0.0 pour
toutes les fonctions).

Ce résultat est favorable pour le Transformer : les séries n'ont pas de tendance à long
terme, ce qui signifie que le modèle n'a pas besoin de différenciation préalable. Le
`context_length = 240 min` de FAYAM est suffisant pour capturer les dynamiques locales.

### 3.3 Périodicité (ACF et FFT)

La FFT révèle une **périodicité de 24 heures (1440 minutes) dominante pour les 19
fonctions**, avec des harmoniques à 12h (720 min) et 8h (480 min).

| Cluster | Puissance 24h | Puissance 12h | Signal périodique |
|---------|--------------|--------------|-------------------|
| 0 | 61–63 % | 14–15 % | Fort et clair |
| 4 | 75–80 % | 5–9 % | Très fort |
| 6 | 2–4 % | < 2 % | Quasi inexistant |
| 8 | 36–43 % | — (360 min) | Modéré |

Le cluster 6 est particulier : la densité de zéros (> 89 %) noie le signal périodique.
Les fonctions sont actives par rafales imprévisibles, sans rythme journalier identifiable.

L'ACF confirme ces résultats : pics significatifs à 1440 lags (24h) et 2880 lags (48h)
pour les clusters 0, 4 et 8. Le cluster 6 présente une ACF proche de zéro au-delà de
quelques lags.

### 3.4 Analyse des zéros

| Cluster | Taux de zéros moyen | Plage silencieuse max |
|---------|--------------------|-----------------------|
| 0 | 0.0 % | 0 h |
| 4 | < 1 % | < 0.1 h |
| 6 | **91.9 %** | **16.4 – 16.8 h** |
| 8 | 25.7 % | 0.4 – 0.9 h |

Le cluster 6 est **zero-inflaté** : les fonctions sont inactives pendant ~17h d'affilée en
moyenne. Ces plages silencieuses correspondent à des fonctions peu sollicitées (type *cold
start intensif*). Le modèle devra être capable de prédire la reprise d'activité après une
longue absence.

Le cluster 8 présente un taux de zéros modéré (~25 %) avec des plages silencieuses courtes
(< 1h) — comportement *bursty* mais avec une activité de fond continue.

---

## 4. Profil de chaque cluster

### Cluster 0 — Charges massives et régulières

- **Magnitude** : 117 000 – 131 000 invocations/min (ordres de grandeur supérieurs aux autres)
- **Signal** : propre, continu, aucun zéro, forte cyclicité 24h (61–63 %)
- **Régularité** : B ≈ −0.20 (plus régulier que Poisson)
- **Pour le Transformer** : cas idéal. Signal fort et prévisible. À utiliser en premier
  pour la reproduction baseline — si le modèle échoue ici, il faut revoir l'architecture.
- **Difficulté** : normalisation obligatoire (magnitude 1000× supérieure aux autres clusters)

### Cluster 4 — Charges moyennes et très cycliques

- **Magnitude** : 90 – 106 invocations/min
- **Signal** : fort, quasi pas de zéros (< 1 %), cyclicité 24h dominante à 75–80 %
- **Régularité** : B ≈ −0.20 (identique à C0, profil structurellement similaire)
- **Pour le Transformer** : deuxième cas le plus favorable. Représente les fonctions
  "standard" — bon point de comparaison avec C0 à une échelle différente.
- **Difficulté** : faible. Vérifier que le modèle gère correctement les quelques zéros isolés.

### Cluster 6 — Fonctions rares / zero-inflatées

- **Magnitude** : 1.6 – 3.1 invocations/min
- **Signal** : très dégradé (90–95 % de zéros), CV > 500 %, B > 0.68
- **Cyclicité** : inexistante (puissance 24h < 4 %)
- **Pour le Transformer** : cas le plus difficile. Le modèle doit apprendre à prédire
  des pics rares sur fond de silence prolongé (~17h). Le `context_length = 240 min`
  représente seulement 1/4 d'une plage silencieuse typique — peut être insuffisant.
- **Difficulté** : élevée. Risque de prédiction constante à zéro (*trivial predictor*).
  À traiter en dernier.

### Cluster 8 — Charges faibles et légèrement bursty

- **Magnitude** : 4.5 – 5.1 invocations/min
- **Signal** : modéré, zéros ~25 %, cyclicité 24h à 36–43 %, B ≈ +0.08
- **Pour le Transformer** : difficulté intermédiaire entre C4 et C6. Le signal périodique
  est présent mais bruité par les pics sporadiques.
- **Difficulté** : modérée. La variance est élevée relativement à la moyenne (CV ~115 %).

---

## 5. Décisions de prétraitement

Les observations ci-dessus conduisent aux décisions suivantes pour le pipeline
`tsf_transf.py` :

| Décision | Justification |
|----------|--------------|
| **Normalisation par fonction** (MinMaxScaler ou StandardScaler) | Clusters 0 et 4 incomparables en absolu (facteur ~25 000×). Normaliser par fonction plutôt que par cluster pour préserver les différences intra-cluster. |
| **Conserver les zéros natifs** | Le `TimeSeriesTransformerForPrediction` tolère les zéros. Les supprimer ou imputer ferait perdre l'information sur les plages silencieuses, cruciale pour C6. |
| **Pas de différenciation** | Toutes les séries sont stationnaires (ADF p ≈ 0) — aucune transformation d'ordre 1 nécessaire. |
| **`context_length = 240 min` suffisant pour C0, C4, C8** | 4 cycles de pointe journaliers couverts. Pour C6, 240 min ≪ durée silencieuse typique (~1 000 min) — à surveiller. |
| **`output_attentions = True` dès le premier run** | Nécessaire pour H1 (SoftCAM-Transformer) et H3 (analyse attention). Activer maintenant évite un réentraînement. |
| **Seed fixée** pour la reproductibilité | Résultats stochastiques (sampling Transformer) — fixer `torch.manual_seed` et `numpy.random.seed`. |

---

## 6. Ordre d'entraînement recommandé

L'ordre suit la complexité croissante du signal :

```
Cluster 0  →  Cluster 4  →  Cluster 8  →  Cluster 6
(signal propre)  (cyclique fort)  (bursty modéré)  (zero-inflaté)
```

**Rationnel** :
- **C0 en premier** : reproduction exacte de la baseline FAYAM (signal le plus propre,
  métriques comparables directement au tableau IX du mémoire FAYAM).
- **C4 ensuite** : valide que le modèle généralise à une magnitude différente.
- **C8** : introduit le bruit et les zéros modérés — premier vrai test de robustesse.
- **C6 en dernier** : cas extrême. Si les métriques sont mauvaises, c'est attendu et
  justifiable — ce n'est pas un échec du modèle mais une limite du `context_length`.

---

## 7. Résumé exécutif (pour présentation encadreurs)

> Les 19 fonctions analysées couvrent trois régimes de charge très contrastés.
> Toutes sont stationnaires (pas de tendance) et présentent une cyclicité journalière
> dominante — sauf le cluster 6 dont 90 % des valeurs sont nulles.
> Ces observations valident les choix architecturaux de FAYAM (`prediction_length = 120 min`,
> `context_length = 240 min`) pour les clusters 0, 4 et 8. Le cluster 6 représente
> un défi spécifique (zero-inflation + absence de périodicité) qui constituera un
> point de discussion méthodologique dans le mémoire.
