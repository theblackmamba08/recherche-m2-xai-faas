# Guide de lecture — Notebook EDA Clusters FaaS

> **À lire lors de la présentation aux encadreurs** : ce document suit le notebook
> `code/notebooks/EDA_clusters.ipynb` cellule par cellule. Chaque entrée explique
> ce que fait la cellule, **pourquoi elle est là**, et détaille les concepts statistiques
> en termes accessibles à un informaticien.
>
> **Notebook** : 49 cellules (index 0 à 48) — 12 sections  
> **Run de référence** : 2026-05-02 (08h31) — résultats dans `code/experiments/eda/`

---

## Cellule 0 — Titre et contexte général *(markdown)*

Présente l'objectif du notebook : analyser les 4 clusters HDBSCAN retenus par FAYAM
(clusters 0, 4, 6, 8) avant tout entraînement du Transformer.

**Pourquoi** : un notebook sans introduction est comme une fonction sans docstring — on ne
sait pas ce qu'il fait avant de l'exécuter. La liste des 11 sections donne le plan complet
de l'analyse, ce qui permet à un encadreur ou un relecteur de naviguer directement à la
partie qui l'intéresse.

---

## Section 1 — Setup & chargement

### Cellule 1 — En-tête de section *(markdown)*

Titre `## 1 — Setup & chargement`. Sert de repère visuel dans le HTML exporté.

---

### Cellule 2 — Imports *(code)*

Charge toutes les bibliothèques nécessaires. Voici pourquoi chaque groupe est là :

| Bibliothèque | Rôle dans ce notebook |
|---|---|
| `numpy`, `pandas` | Manipulation matricielle et tableaux de données |
| `matplotlib`, `seaborn` | Génération de toutes les figures |
| `scipy.stats` | Skewness, kurtosis, KDE (densité lissée) |
| `scipy.fft` | Transformée de Fourier (détection des cycles) |
| `statsmodels.tsa.stattools` | ACF (autocorrélation) et test ADF (stationnarité) |
| `pathlib.Path` | Chemins multi-OS (Windows `\` vs Linux `/`) |

Définit aussi deux constantes globales :
- `CLUSTER_IDS = [0, 4, 6, 8]` : les 4 clusters analysés
- `CLUSTER_COLORS` : dictionnaire `{0: rouge, 4: bleu, 6: vert, 8: orange}` utilisé dans
  **tous** les graphiques du notebook — même couleur pour un cluster dans toutes les figures

**Pourquoi centraliser les couleurs** : si on change la couleur de C6 dans cette cellule,
elle change dans les 30+ figures générées en aval. Sans ça, chaque figure aurait ses propres
couleurs et les comparaisons visuelles entre figures deviendraient impossibles.

---

### Cellule 3 — Détection environnement + DATA_DIR adaptatif *(code)*

#### Ce que fait la cellule

Exécute un `try/except` sur `from google.colab import drive` :
- Si l'import **réussit** → on est sur Colab → monte Google Drive → `DATA_DIR` pointe vers
  le Drive → crée le dossier de résultats sur Drive
- Si l'import **échoue** (ImportError) → on est en local → `DATA_DIR` pointe vers
  `memoire/06-datasets/raw/`

Initialise aussi :
- `RUN_DATE = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')` : horodatage du run
- `results = {}` : dictionnaire Python vide qui va **accumuler toutes les métriques** tout
  au long du notebook

#### Pourquoi c'est nécessaire

Sans cette cellule, il faudrait modifier manuellement la ligne `DATA_DIR = ...` à chaque
fois qu'on change d'environnement. C'est exactement le problème qui a causé le bug
`FileNotFoundError` lors du premier run : le notebook avait `DATA_DIR` hardcodé en chemin
local, et en Colab les fichiers étaient sur Drive.

L'horodatage dans `RUN_DATE` sert à nommer les fichiers de résultats de façon unique :
`eda_results_2026-05-02_08-31.json`. Si on relance le notebook demain, on obtient un nouveau
fichier au lieu d'écraser l'ancien.

---

### Cellule 4 — Chargement des données *(code)*

#### Ce que fait la cellule

Lit les 4 CSV et construit **deux représentations** des mêmes données :

```
raw[cid]   → shape (n_fonctions, 20160)   ← format CSV natif : 1 ligne = 1 fonction
long[cid]  → shape (20160, n_fonctions)   ← transposé : 1 ligne = 1 instant (timestep)
```

Le DataFrame `long` reçoit un index `datetime` : `pd.date_range('2021-01-01', periods=20160,
freq='1min')`. Les 14 jours de données sont ainsi indexés de `2021-01-01 00:00` à
`2021-01-14 23:59`.

#### Pourquoi deux représentations

`raw` est utile quand on veut itérer sur les fonctions (boucle sur les lignes). `long` est
indispensable pour tout ce qui est temporel : tracer une courbe `ax.plot(df.index, df[fn])`,
découper une fenêtre `df.loc['2021-01-01':'2021-01-04']`, ou calculer l'ACF avec statsmodels
qui attend une série avec un index temporel.

**Sortie attendue** (à vérifier à l'exécution) :
```
cluster_0 : 3 fonctions [942, 943, 944]
cluster_4 : 5 fonctions [949, 951, 952, 953, 954]
cluster_6 : 5 fonctions [138, 139, 140, 143, 144]
cluster_8 : 6 fonctions [964, 965, 967, 968, 969, 977]
Chargement terminé — 20 160 pas x 14 jours x 4 clusters (19 fonctions total)
```

---

## Section 2 — Vue d'ensemble

### Cellule 5 — En-tête de section *(markdown)*

---

### Cellule 6 — Tableau de synthèse par fonction *(code)*

#### Ce que fait la cellule

Calcule 8 métriques pour chacune des 19 fonctions et affiche un DataFrame de 19 lignes.

#### Explication détaillée de chaque métrique

**Moyenne et médiane**

La moyenne est la valeur "centrale" habituelle. La médiane est la valeur du milieu quand on
trie toutes les observations du plus petit au plus grand.

> *Analogie code* : si une liste de 20 000 valeurs est `[0, 0, 0, ..., 0, 500, 1000]`, la
> **moyenne** sera tirée vers le haut par les quelques grandes valeurs, alors que la
> **médiane** sera 0 (la plupart des valeurs sont nulles). Pour C6, la moyenne est ~2
> inv/min mais la médiane est 0 — ce décrochage signale immédiatement la présence massive
> de zéros.

**Écart-type (std)**

Mesure combien les valeurs s'écartent en moyenne de la moyenne. Un std élevé = la série
monte et descend beaucoup. Mais le std seul est inutile pour comparer C0 (std ≈ 80 000) et
C4 (std ≈ 65) — les magnitudes sont trop différentes.

**CV — Coefficient de Variation = std / moyenne × 100**

Le CV normalise l'écart-type par la moyenne. Il répond à la question : *"relativement à
sa taille, combien cette série oscille-t-elle ?"*

> *Analogie concrète* : une série qui varie entre 100 000 et 140 000 a un std de 20 000,
> mais un CV de seulement 16 % (oscillation faible). Une série qui varie entre 0 et 20 a
> un std de 10, mais un CV de 500 % (oscillation extrême). Le CV permet la comparaison
> honnête entre les deux.

| Valeur CV | Interprétation pratique |
|---|---|
| < 30 % | Série très régulière, facile à prédire |
| 30–70 % | Variabilité modérée — cas typique |
| > 100 % | Forte variabilité — cycles importants ou bruit |
| > 300 % | Série dominée par des pics rares ou des plages nulles |

Pour nos données : C0 (CV ≈ 66 %) et C4 (CV ≈ 67 %) sont comparables malgré leurs magnitudes
radicalement différentes. C6 (CV ≈ 660 %) est dans un régime complètement différent.

**Zéros (%)**

Proportion des 20 160 timesteps où la fonction n'a reçu aucune invocation. Important pour
le Transformer : un timestep à zéro n'est pas "aucune information" — c'est l'information
que la fonction était **inactive** à cet instant.

**Burstiness B = (σ − μ) / (σ + μ)**

C'est la métrique la plus originale. Elle compare l'écart-type à la moyenne via un rapport
normalisé qui vaut toujours entre −1 et +1.

*Pourquoi ce ratio particulier ?* Il vient de la physique des réseaux et de la biologie
(modélisation des trains de neurones). Son intérêt est qu'il a une **référence naturelle** :
un processus de Poisson (le modèle du trafic réseau aléatoire "standard") a σ = μ, donc
B = 0.

| Valeur B | Signification |
|---|---|
| B = −1 | Série parfaitement régulière (intervalle constant entre événements) |
| B ≈ −0.2 | Plus régulier que Poisson — trafic prévisible, patron cyclique |
| B = 0 | Processus de Poisson — aléatoire "standard" |
| B ≈ +0.7 | Très bursty — longs silences entrecoupés de rafales intenses |
| B = +1 | Bursty extrême (limite théorique) |

> *Pourquoi c'est important pour le Transformer* : un Transformer prédit mieux les séries
> régulières (B < 0) parce qu'il peut mémoriser le pattern cyclique dans ses têtes
> d'attention. Pour les séries très bursty (B > 0.5), le modèle n'a pas de pattern à
> mémoriser — chaque pic est une surprise.

**Résultats observés pour nos clusters** :

| Cluster | Burstiness B moyen | Ce que ça veut dire |
|---|---|---|
| C0 | −0.200 | Trafic très régulier, presque métronome |
| C4 | −0.193 | Idem, légèrement moins régulier |
| C8 | +0.073 | Quasi-Poisson, légèrement bursty |
| C6 | +0.742 | Trafic en rafales intenses sur fond de silence |

---

### Cellule 7 — Résumé agrégé par cluster *(code)*

Agrège les 19 lignes de la cellule précédente en 4 lignes (une par cluster) en moyennant
CV, zéros (%) et burstiness par cluster.

**Pourquoi** : lors d'une présentation courte, on présente les 4 lignes, pas les 19.
C'est la version "résumé exécutif" de la vue d'ensemble — la première chose à montrer.

---

### Cellule 8 — Capture métriques overview *(code, silencieuse)*

Ne produit aucune sortie visible. Copie les métriques calculées dans `results['overview']`
et `results['cluster_means']`.

**Pourquoi cette cellule "muette" existe** : le dictionnaire `results` est comme un buffer
d'accumulation. Chaque section du notebook y dépose ses métriques clés. La section 12
videra ce buffer dans un fichier JSON. Sans les cellules de capture, toutes les métriques
calculées depuis le début du notebook resteraient dans des variables locales Python et
disparaîtraient à la fermeture de Colab.

---

## Section 3 — Statistiques descriptives

### Cellule 9 — En-tête de section *(markdown)*

---

### Cellule 10 — `describe()` par cluster *(code)*

#### Ce que fait la cellule

Appelle `pandas.DataFrame.describe()` sur chaque cluster et affiche le résultat transposé
(une ligne par fonction). Renvoie : count, mean, std, min, 25 %, 50 %, 75 %, max.

#### Explication des percentiles

Les percentiles (ou quantiles) donnent une image de la **distribution** de la série sans
avoir à regarder toutes les 20 160 valeurs.

- **p25 (1er quartile)** : 25 % des valeurs sont inférieures à ce nombre
- **p50 (médiane)** : la valeur du milieu
- **p75 (3e quartile)** : 75 % des valeurs sont inférieures à ce nombre
- **IQR = p75 − p25** : l'étendue de la "zone centrale" de la distribution

> *Analogie code* : les percentiles sont comme les statistiques d'un profiler de performance.
> Si le p50 d'un benchmark est 10 ms et le p99 est 500 ms, ça dit que la moitié des requêtes
> sont rapides mais que les pires sont catastrophiques — impossible à voir avec la moyenne seule.

**Ce qu'on observe pour C6** : p50 = 0 (la médiane est zéro), mais la moyenne est ~2 et le
max peut atteindre 200+. Ce décrochage extrême entre médiane et moyenne est le signe
statistique classique d'une distribution **zero-inflatée avec queue lourde**.

---

## Section 4 — Séries temporelles

### Cellule 11 — En-tête de section *(markdown)*

---

### Cellule 12 — Vue complète 14 jours *(code)*

Pour chaque cluster : une figure avec autant de sous-graphes que de fonctions, chacun
montrant la série complète sur 14 jours (20 160 points par courbe).

**Ce qu'on cherche visuellement** :
- C0/C4 : oscille régulièrement entre un minimum nocturne et un maximum diurne — signal
  périodique clair, visualisable à l'œil nu
- C6 : longues plages plates à zéro avec des pics isolés — aucune régularité visible
- C8 : activité de fond faible avec des pics sporadiques

**Pourquoi c'est la première figure à montrer** : c'est la plus intuitive. Avant toute
statistique, on voit immédiatement si la série a une structure ou si c'est du bruit pur.
Un encadreur comprendra C6 en 5 secondes en voyant le graphique, là où un tableau de
métriques demanderait plusieurs minutes d'interprétation.

---

### Cellule 13 — Zoom 3 jours *(code)*

Même structure que la cellule 12 mais limité aux 3 premiers jours.

**Pourquoi le zoom est nécessaire** : sur 14 jours, 20 160 points sont tassés sur l'axe X.
Chaque pixel de l'écran représente plusieurs dizaines de minutes — les variations intra-
journalières deviennent invisibles. Le zoom sur 3 jours (4 320 points) permet de voir :
- L'heure exacte du pic quotidien (ex : 18h-20h pour C0)
- Les creux nocturnes (2h-6h du matin)
- Les micro-variations minute à minute

---

### Cellule 14 — Heatmap 14 jours × 1440 min *(code)*

#### Ce que fait la cellule

Réorganise la série de 20 160 valeurs en **matrice 14 × 1440** :
- 14 lignes = 14 jours
- 1 440 colonnes = 1 440 minutes dans une journée (24h × 60 min)
- La couleur de chaque cellule = la valeur de `nbrconc` à ce jour, à cette minute

#### Pourquoi cette visualisation est puissante

Chaque colonne de la heatmap correspond à une heure fixe du jour. Si la colonne "minute
1020" (17h00) est toujours orange sur les 14 lignes, ça veut dire que **tous les jours à
17h00, cette fonction a un pic**. C'est la periodicité journalière rendue visible.

> *Analogie code* : c'est comme un tableau de profiling 2D — une ligne par run, une colonne
> par fonction de code. Les fonctions lentes apparaissent comme des colonnes rouges dans
> tous les runs.

**Ce qu'on observe** :
- C0/C4 : colonnes cohérentes d'un jour à l'autre → pattern journalier stable
- C6 : matrice presque entièrement jaune (zéros) avec quelques cellules orange dispersées
  aléatoirement → aucun pattern régulier
- C8 : colonnes vaguement cohérentes mais avec beaucoup de bruit

---

### Cellule 15 — Profil journalier moyen *(code)*

#### Ce que fait la cellule

Pour chaque fonction, découpe la série en 14 tranches de 1 440 points (14 jours) et calcule
la **moyenne par minute du jour** :

```
profil_journalier[minute_m] = moyenne(valeur_jour1[m], valeur_jour2[m], ..., valeur_jour14[m])
```

Résultat : une courbe lissée sur 1 440 points représentant "la journée type" de la fonction.

#### Pourquoi moyenner sur les 14 jours

Le bruit aléatoire d'un jour donné disparaît dans la moyenne. Ce qui reste est le **signal
systématique** — le vrai pattern journalier, libéré du bruit. C'est le même principe que
le moyennage d'un signal bruité en électronique pour faire ressortir la fréquence porteuse.

**Ce qu'on observe** : pour C0 et C4, la courbe montre clairement un creux la nuit (~2h-6h)
et un pic en fin d'après-midi (~17h-19h). Pour C6, la courbe est quasi-plate à zéro avec
un léger frémissement — le signal périodique est noyé sous les zéros.

---

## Section 5 — Analyse des zéros

### Cellule 16 — En-tête de section *(markdown)*

---

### Cellule 17 — Statistiques des zéros *(code)*

#### Ce que fait la cellule

Pour chaque fonction :
1. Compte le nombre de timesteps à zéro et calcule le pourcentage
2. Calcule la **plus longue séquence de zéros consécutifs** avec un algorithme O(n) :
   parcourt la série, maintient un compteur `current` qui s'incrémente à chaque zéro et
   se remet à zéro à chaque valeur non-nulle, en mémorisant le max rencontré

#### Pourquoi les zéros consécutifs sont critiques pour le Transformer

Le `TimeSeriesTransformerForPrediction` utilise les `context_length = 240` dernières minutes
comme **contexte** pour prédire les 120 prochaines minutes.

Si la fenêtre de contexte est entièrement constituée de zéros, le modèle n'a aucune
information sur le comportement passé de la fonction. C'est comme demander à quelqu'un de
prédire ce que quelqu'un va faire dans l'heure qui vient, en sachant seulement qu'il a
dormi pendant les 4 dernières heures — on ne sait pas grand chose.

**Ce qu'on observe** :
- C6 : plages silencieuses max de 16–17h consécutives = ~1000 minutes
- `context_length = 240 min` ne représente que **24 %** d'une plage silencieuse typique
- Conséquence : le Transformer verra souvent un contexte entièrement nul pour C6 → risque
  de prédire systématiquement zéro (trivial predictor)

---

### Cellule 18 — Capture métriques zéros *(code, silencieuse)*

Sauvegarde dans `results['zeros']`. Même logique que la cellule 8.

---

### Cellule 19 — Graphique taux de zéros *(code)*

Bar chart avec 19 barres (une par fonction), colorées par cluster, avec les pourcentages
affichés au-dessus de chaque barre.

**Ce qu'on voit immédiatement** : les barres de C6 dépassent 89 %, celles de C8 sont à
~25 %, celles de C0/C4 sont quasi-invisibles. Ce graphique est plus percutant qu'un tableau
pour une présentation — on voit la hiérarchie en une seconde.

---

### Cellule 20 — Distribution des runs de zéros (C6 et C8) *(code)*

#### Ce que fait la cellule

Pour C6 et C8 seulement : collecte toutes les séquences de zéros consécutifs et trace leur
distribution sous forme d'histogramme. Une "séquence" (run) = une série ininterrompue de
timesteps à zéro.

#### Pourquoi cette analyse est distincte pour C6 et C8

Pour C0 et C4 : quasiment pas de zéros → l'histogramme serait vide ou trivial.

Pour C6 et C8, on veut savoir si les zéros sont :
- **Dispersés** (beaucoup de runs de 1–2 min) → bruit ponctuel, moins problématique
- **Structurés** (quelques runs très longs) → plages silencieuses structurelles, critiques
  pour le contexte du Transformer

> *Analogie code* : c'est comme analyser les temps d'inactivité d'un serveur. Un serveur
> qui s'arrête 60 fois pour 1 minute est très différent d'un serveur qui s'arrête 1 fois
> pour 60 minutes — même nombre de minutes d'inactivité totale, mais impact très différent
> sur les utilisateurs.

**Ce qu'on observe pour C6** : distribution bimodale — beaucoup de très courts runs (1–2
min entre deux appels) ET beaucoup de très longs runs (800–1000 min = plages nocturnes). Ce
n'est pas du bruit aléatoire — c'est un comportement structuré.

---

## Section 6 — Distributions

### Cellule 21 — En-tête de section *(markdown)*

---

### Cellule 22 — Histogrammes + KDE *(code)*

#### Ce que fait la cellule

Pour chaque fonction : histogramme des valeurs **non-nulles** avec une courbe KDE
(Kernel Density Estimation) superposée.

#### Qu'est-ce que la KDE ?

L'histogramme découpe les valeurs en intervalles fixes (bins) et compte le nombre de valeurs
dans chaque interval. La KDE est une version lissée : au lieu de bins discrets, elle place
une petite "cloche gaussienne" sur chaque point de données et les additionne pour obtenir
une courbe continue.

> *Analogie* : l'histogramme est comme un diagramme à barres pixelisé, la KDE est sa
> version anti-aliasée. La KDE révèle mieux la forme globale de la distribution (unimodale ?
> bimodale ? asymétrique ?) là où l'histogramme peut créer des artefacts liés au choix
> de la taille des bins.

#### Pourquoi travailler sur les valeurs non-nulles

Pour C6 : 91 % de zéros. Si on les inclut, l'histogramme ressemble à ça :
```
[91 % de la hauteur ici]
|████████████████████|
|                    |  |  |
 0    1    2    3  ...200
```
La partie intéressante (les valeurs > 0) est écrasée et illisible. En filtrant les zéros,
on voit la distribution réelle des pics d'activité.

**Ce qu'on cherche** :
- Distribution **unimodale et symétrique** (en cloche) → plus facile à prédire pour le
  Transformer (ses erreurs seront gaussiennes)
- Distribution **asymétrique à droite** (longue queue vers les grandes valeurs) → le modèle
  sous-estimera les pics

---

### Cellule 23 — Box plots *(code)*

#### Qu'est-ce qu'une boîte à moustaches ?

La boîte montre le IQR (écart entre p25 et p75) — la zone où se trouvent les 50 % centraux
des données. La ligne médiane (p50) est tracée à l'intérieur. Les moustaches s'étendent
jusqu'à p25 − 1.5×IQR et p75 + 1.5×IQR. Les points au-delà des moustaches sont des
**outliers** (valeurs atypiques).

> *Analogie code* : le box plot est comme un résumé statistique compressé en une figure.
> C'est l'équivalent visuel de `df.describe()` — on voit la médiane, l'IQR et les outliers
> d'un coup d'œil.

#### Pourquoi l'échelle logarithmique pour certains clusters

Pour C0 : min ≈ 30 000, max ≈ 450 000, IQR ≈ 60 000–200 000. En échelle linéaire, la boîte
et les moustaches occupent une plage de 170 000 unités. Les outliers extrêmes (> 300 000)
feraient que la boîte paraîtrait minuscule par rapport à l'axe.

En **échelle log**, chaque ordre de grandeur (×10) prend le même espace visuel. La boîte
et les moustaches deviennent lisibles même quand les valeurs s'étendent sur plusieurs ordres
de grandeur.

---

### Cellule 24 — Skewness et kurtosis *(code)*

#### Qu'est-ce que le skewness (asymétrie) ?

Le skewness mesure si la distribution est symétrique ou penchée d'un côté.

```
skewness = 0    → distribution symétrique (gaussienne parfaite)
skewness > 0    → queue longue à droite (quelques valeurs très élevées tirent la moyenne)
skewness < 0    → queue longue à gauche (rare en pratique)
```

> *Analogie* : penser aux temps de réponse d'une API. La plupart des requêtes prennent
> 50 ms (médiane), mais quelques requêtes lentes prennent 5 000 ms (outliers). Distribution
> asymétrique à droite : skewness positif élevé.

Pour C6 : skewness > 10 (extrêmement asymétrique à droite). Les quelques pics de 100–200
invocations/min tirent très loin la queue droite de la distribution.

#### Qu'est-ce que le kurtosis (aplatissement) ?

Le kurtosis mesure la "lourdeur des queues" — combien les valeurs extrêmes sont fréquentes
comparées à une distribution gaussienne.

```
kurtosis = 0    → queues comme une gaussienne (référence)
kurtosis > 0    → queues lourdes (plus d'outliers qu'une gaussienne → distribution leptokurtique)
kurtosis < 0    → queues légères (moins d'outliers qu'une gaussienne)
```

#### Pourquoi ces métriques importent pour le Transformer

Le Transformer de FAYAM est entraîné avec une loss MSE (erreur quadratique moyenne). La MSE
pénalise fort les grandes erreurs — ce qui convient quand la distribution des erreurs est
gaussienne. Mais si la distribution a un skewness élevé et un kurtosis positif (comme C6),
les erreurs du modèle auront aussi cette forme : quelques très grandes erreurs sur les pics
rares feront monter le MSE de façon disproportionnée.

En pratique : le modèle aura tendance à **sous-prédire les pics** pour minimiser le MSE
moyen, sacrifiant les quelques instants bursty pour mieux prédire les nombreux instants
silencieux. C'est une limite importante à mentionner dans le mémoire.

---

## Section 7 — Périodicité (ACF + FFT)

### Cellule 25 — En-tête de section *(markdown)*

---

### Cellule 26 — ACF jusqu'à 48h *(code)*

#### Qu'est-ce que l'autocorrélation (ACF) ?

L'autocorrélation mesure **dans quelle mesure une série est corrélée avec elle-même,
décalée dans le temps**. Un "lag" est l'unité de décalage (ici : 1 lag = 1 minute).

La formule : `ACF(k) = corrélation(série[t], série[t-k])`

> *Analogie code* : c'est comme mesurer la similarité entre une liste et une copie d'elle-
> même décalée de k positions. Si la liste est `[1, 2, 1, 2, 1, 2]`, décalée de 2 positions
> on retrouve exactement la même liste → ACF(2) = 1.0.

**Interprétation des pics dans l'ACF** :
- Un **pic à lag 1440** (24h) signifie que la valeur à l'instant t est très corrélée avec
  la valeur à `t − 1440 min` (il y a 24h). Autrement dit : la série reproduit le même
  comportement chaque jour.
- Un **pic à lag 2880** (48h) confirme que le cycle est à 24h et pas aléatoire : si c'était
  du bruit, le pic à 1440 pourrait être une coïncidence, mais deux pics harmoniques (1440
  et 2880) sont une preuve forte de périodicité.
- Une ACF **qui tombe à zéro après quelques lags** (comme C6) signifie que le passé
  récent n'aide pas à prédire le futur — il n'y a pas de mémoire temporelle.

**Ce qu'on observe** : pour C0 et C4, l'ACF maintient des pics élevés à 1440 et 2880
(corrélations de 0.6–0.8 — très fortes). Pour C6, l'ACF tombe à ~0 dès lag 10.

---

### Cellule 27 — FFT : périodes dominantes (texte) *(code)*

#### Qu'est-ce que la FFT (Transformée de Fourier Rapide) ?

La FFT décompose un signal temporel en somme de sinusoïdes. Chaque sinusoïde est
caractérisée par une **fréquence** (combien de cycles par unité de temps) et une
**amplitude** (puissance du cycle).

> *Analogie* : imaginez une chanson. La FFT décompose la chanson en ses notes constitutives
> — note de do à 261 Hz, note de sol à 392 Hz, etc. Ici, au lieu de notes musicales, on
> décompose la série temporelle en cycles : cycle de 24h, cycle de 12h, cycle de 8h...

**Ce que la cellule calcule** :
1. Soustrait la moyenne (pour éliminer la composante "DC" — la valeur de base constante)
2. Applique `rfft()` (FFT réelle) → donne les amplitudes pour chaque fréquence
3. Calcule la puissance = amplitude² pour chaque fréquence
4. Identifie les 5 fréquences avec la puissance la plus élevée et les convertit en périodes
   (période = 1 / fréquence)
5. Exprime chaque puissance en % de la puissance totale

**Différence ACF vs FFT** : l'ACF répond à "est-ce qu'il y a une périodicité ?". La FFT
répond à "quelle proportion de la variabilité totale est expliquée par chaque cycle ?".

**Ce qu'on observe** : la période 1440 min (24h) explique 61–63 % de la variance de C0 et
75–80 % de C4. Pour C6, aucune période n'explique plus de 4 % — la variance est distribuée
sur toutes les fréquences de façon presque uniforme (bruit blanc).

---

### Cellule 28 — Capture métriques FFT *(code, silencieuse)*

Sauvegarde dans `results['fft_top_periods']`. Même logique que cellules 8 et 18.

---

### Cellule 29 — Périodogramme FFT (graphique) *(code)*

#### Ce que fait la cellule

Trace le spectre de puissance FFT pour chaque fonction :
- Axe X : période en minutes (de 10 min à 20 160 min = 14 jours), en échelle logarithmique
- Axe Y : puissance en échelle logarithmique (semilogy)
- Repères verticaux à 1h (60 min), 8h, 12h, 24h

#### Pourquoi les deux axes sont en log

Les puissances s'étendent sur plusieurs ordres de grandeur (de 10³ à 10¹²). En échelle
linéaire, les petites puissances seraient invisibles comparées aux dominantes. En log-log,
on voit simultanément les composantes fortes ET les harmoniques faibles.

**Ce qu'on observe** :
- C4 : pic très étroit et très haut à 1440 min — signal quasi-sinusoïdal pur
- C8 : pic à 1440 min mais aussi de l'énergie répartie sur d'autres fréquences → structure
  plus complexe
- C6 : pas de pic dominant — le spectre ressemble à une ligne quasi-droite (bruit blanc)

---

## Section 8 — Stationnarité (test ADF)

### Cellule 30 — En-tête de section *(markdown)*

---

### Cellule 31 — Test de Dickey-Fuller augmenté *(code)*

#### Qu'est-ce que la stationnarité ?

Une série est **stationnaire** si ses propriétés statistiques (moyenne, variance, structure
de corrélation) ne changent pas dans le temps.

> *Analogie code* : imaginez un processus qui génère des logs. Stationnaire = le même
> type de logs apparaît tout le long, avec la même fréquence. Non-stationnaire = au départ
> peu de logs d'erreur, puis de plus en plus — le comportement du processus change dans le
> temps.

**Pourquoi c'est crucial pour le Transformer** : un modèle de séries temporelles apprend
des patterns sur les données d'entraînement. Si la série est non-stationnaire (ex : elle
monte régulièrement sur 14 jours), les patterns appris sur la première semaine ne seront
plus valides sur la deuxième — le modèle sera en permanence "en retard" sur la tendance.

#### Qu'est-ce que le test ADF ?

Le test Augmented Dickey-Fuller (ADF) est un test statistique qui vérifie si une série a
une "racine unitaire" — le terme technique pour "tendance non-stationnaire".

**Fonctionnement simplifié** :
- Le test essaie de vérifier si l'équation `valeur[t] = valeur[t-1] + bruit` (marche
  aléatoire pure) décrit bien la série
- Si oui → non-stationnaire (p-value > 0.05)
- Si non → stationnaire (p-value < 0.05)

**La p-value** est la probabilité d'observer les données si la série était non-stationnaire.
Une p-value ≈ 0.0000 signifie : "il est pratiquement impossible que cette série soit non-
stationnaire" → on rejette la non-stationnarité → la série **est stationnaire**.

**Ce qu'on observe** : 19/19 fonctions ont une p-value ≈ 0.000 → toutes stationnaires. Pas
de différenciation nécessaire avant l'entraînement.

---

### Cellule 32 — Capture métriques ADF *(code, silencieuse)*

Sauvegarde dans `results['adf']`, `results['n_stationary']`, `results['n_total_functions']`.

---

### Cellule 33 — ADF après différenciation (cellule de secours) *(code)*

Si certaines séries avaient été non-stationnaires, cette cellule aurait calculé la
différence d'ordre 1 : `diff_serie[t] = serie[t] - serie[t-1]` et relancé le test ADF.

**Qu'est-ce que la différenciation ?** Au lieu d'analyser la valeur absolue à chaque
instant, on analyse la **variation entre deux instants consécutifs**. Si la série monte
régulièrement de 100 par minute, la série différenciée serait constante à 100 — stationnaire.

**Ce qu'on observe** : `"Toutes les séries sont déjà stationnaires."` — la cellule n'a rien
à faire. Elle reste dans le notebook pour que celui-ci soit réutilisable sur d'autres
datasets où certaines séries pourraient être non-stationnaires.

---

## Section 9 — Cohérence intra-cluster

### Cellule 34 — En-tête de section *(markdown)*

---

### Cellule 35 — Corrélations Pearson et Spearman *(code)*

#### Ce qu'on cherche à valider

FAYAM a regroupé les fonctions en clusters via HDBSCAN. L'hypothèse est que les fonctions
d'un même cluster ont des **comportements similaires**. Cette section vérifie cette hypothèse
en mesurant à quel point les fonctions d'un cluster sont corrélées entre elles.

Si deux fonctions Fn942 et Fn943 du même cluster ont une corrélation de 0.98, ça veut dire
que quand l'une monte, l'autre monte aussi — elles partagent le même profil de charge.

#### Pearson vs Spearman : quelle différence ?

**Corrélation de Pearson** : mesure la corrélation **linéaire**. Elle répond à "est-ce
que les deux séries montent et descendent de façon proportionnelle ?"

```
Pearson = 1.0  → si série A monte de 100, série B monte de c×100 (c constant)
Pearson = 0.0  → aucune relation linéaire
Pearson = -1.0 → relation linéaire inverse
```

Problème : Pearson est sensible aux outliers. Si une série a un pic de 500 000 à un seul
instant et l'autre non, Pearson sera fortement abaissé même si le reste de la série est
identique.

**Corrélation de Spearman** : mesure la corrélation des **rangs**. Au lieu de comparer les
valeurs brutes, on compare leur position dans le classement ("la valeur la plus élevée du
jour est-elle la même heure pour les deux fonctions ?").

```
Spearman = 1.0  → quand série A est dans son top 10 %, série B aussi
Spearman = 0.0  → aucune relation de rang
```

Spearman est **robuste aux outliers** et capture les relations non-linéaires monotones.
Utiliser les deux ensemble donne une image complète : si Pearson est bas mais Spearman est
haut, ça signifie que les séries ont la même structure ordinale mais des magnitudes très
différentes.

**Ce qu'on observe** :
- C0 et C4 : Pearson > 0.95 et Spearman > 0.95 → fonctions quasi-identiques dans leur
  cluster (clustering HDBSCAN très cohérent)
- C6 : corrélations plus faibles (0.6–0.8) → les pics de chaque fonction ne sont pas
  synchronisés parfaitement, mais la structure générale (silence long, rafale rare) est
  commune

---

### Cellule 36 — Distance euclidienne normalisée (proxy DTW) *(code)*

#### Ce que fait la cellule

1. Normalise chaque série avec MinMaxScaler (ramène les valeurs entre 0 et 1)
2. Calcule la distance euclidienne entre chaque paire de fonctions du même cluster :
   `dist(A, B) = sqrt(mean((A[t] - B[t])² pour tout t))`

#### Qu'est-ce que le DTW et pourquoi ne pas l'utiliser directement ?

Dynamic Time Warping (DTW) est une mesure de similarité pour les séries temporelles qui
permet un **alignement flexible** dans le temps. Si série A a son pic à 17h05 et série B
à 17h15, DTW reconnaît qu'elles sont similaires malgré le décalage. La corrélation de
Pearson dirait qu'elles sont différentes.

Problème : DTW exact sur 20 160 points est O(n²) en temps et mémoire. Avec 20 000 points,
c'est 400 millions d'opérations par paire — trop lent pour cette EDA.

La distance euclidienne sur séries normalisées est un proxy acceptable : elle donne la même
information qualitative (distance proche de 0 = séries similaires) sans le coût
computationnel.

---

## Section 10 — Comparaison inter-cluster

### Cellule 37 — En-tête de section *(markdown)*

---

### Cellule 38 — CV et burstiness par cluster (box plot) *(code)*

Deux box plots côte à côte : CV (%) et Burstiness B. Chaque boîte = distribution de la
métrique au sein d'un cluster (les points dans la boîte sont les différentes fonctions).

**Pourquoi cette figure est centrale** : elle montre d'un seul graphique que C0 et C4 ont
des profils statistiquement identiques malgré leurs magnitudes radicalement différentes
(CV ≈ 66 %, B ≈ −0.20 pour les deux). Et que C6 est dans un régime statistique différent
(CV > 500 %, B > 0.70).

C'est la **figure de conclusion** de l'EDA : elle justifie que le Transformer va apprendre
facilement sur C0/C4, modérément sur C8, et difficilement sur C6 — et que cet ordre de
difficulté est mesurable, pas juste intuitif.

---

### Cellule 39 — Profils journaliers normalisés superposés *(code)*

Pour chaque cluster : calcule le profil journalier moyen (section 4, cellule 15) puis le
**normalise entre 0 et 1** : `profil_normalisé = (profil - min) / (max - min + ε)`

Trace les 4 courbes normalisées sur le même graphique.

**Pourquoi normaliser avant de superposer** : C0 a une amplitude de ~120 000, C6 de ~2.
Sans normalisation, la courbe de C6 serait écrasée à zéro sur le graphique par les courbes
de C0. La normalisation permet de comparer la **forme** des profils (heure du pic, heure du
creux) indépendamment de l'amplitude.

**Ce qu'on observe** : C0 et C4 ont des courbes qui se superposent presque parfaitement
(même pic vers 17h, même creux vers 3h). C8 a la même forme mais plus bruitée. C6 est
pratiquement plat (aucun cycle).

---

### Cellule 40 — Tableau de synthèse inter-cluster *(code)*

Tableau en 4 lignes : Cluster | Nb fonctions | Moy. globale | CV moy. | Zéros moy. |
Burstiness moy.

**Pourquoi** : résumé numérique complémentaire aux graphiques. C'est le tableau à copier
dans le mémoire ou dans les slides de présentation.

---

## Section 11 — Synthèse & recommandations

### Cellule 41 — En-tête de section *(markdown)*

---

### Cellule 42 — Synthèse ASCII *(code)*

Affiche une boîte ASCII avec le profil de chaque cluster et 6 recommandations numérotées
pour le pipeline d'entraînement.

**Les 6 recommandations et leur justification** :

| # | Recommandation | Justification (d'après l'EDA) |
|---|---|---|
| 1 | Normaliser par fonction (MinMaxScaler ou StandardScaler) | C0 (~120 000) et C6 (~2) : facteur 60 000×. Sans normalisation, le modèle optimise pour C0 et ignore C6 |
| 2 | Conserver les zéros natifs | `TimeSeriesTransformer` tolère les zéros. Les imputer ferait perdre l'information sur les plages silencieuses — cruciale pour comprendre C6 |
| 3 | Pas de différenciation | 19/19 séries stationnaires (ADF p ≈ 0) → aucune transformation nécessaire |
| 4 | Surveiller `context_length = 240 min` pour C6 | Plages silencieuses de C6 = ~1000 min soit 4× le context_length. Le contexte sera souvent entièrement nul |
| 5 | Activer `output_attentions=True` dès le premier run | Indispensable pour H1 (SoftCAM) et H3 (attention). Activer maintenant évite un réentraînement |
| 6 | Ordre d'entraînement : C0 → C4 → C8 → C6 | Complexité croissante : propre → cyclique fort → bursty modéré → zero-inflaté |

---

### Cellule 43 — Tableau de bord EDA (4 métriques) *(code)*

4 bar charts (Moyenne, CV, Zéros, Burstiness) pour les 19 fonctions, colorées par cluster.

**Pourquoi** : vue panoramique qui permet de voir simultanément où chaque fonction se
positionne sur les 4 axes. Utile pour détecter une fonction atypique dans son cluster :
si une barre de couleur bleue (C4) est au milieu des barres rouges (C0), c'est suspect.

---

### Cellule 44 — Scatter CV × Burstiness *(code)*

Nuage de points avec CV en abscisse, Burstiness en ordonnée. Chaque point = une fonction,
colorée et annotée.

**Ce qu'on lit dans ce graphique** :

```
             Burstiness B
    +1 |                         C6 ●●●●●
       |
  +0.5 |
       |
     0 |----------------------------C8●●●●●●--
       |
  -0.5 |   C0●●● C4●●●●●
    -1 |________________________
         0    100  200  300  400  500  600  700
                        CV (%)
```

- C0 et C4 se regroupent en bas à gauche (faible variabilité, réguliers) → cas faciles
- C8 est au centre droit (variabilité modérée, quasi-Poisson)
- C6 est isolé en haut à droite (variabilité extrême, très bursty) → cas difficile

**Pourquoi cette figure est utile pour la présentation** : elle justifie visuellement
l'ordre d'entraînement C0 → C4 → C8 → C6. Les encadreurs voient immédiatement que l'ordre
suit un gradient de difficulté mesurable sur deux axes indépendants.

---

## Section 12 — Sauvegarde des résultats

### Cellule 45 — En-tête de section *(markdown)*

---

### Cellule 46 — Sauvegarde JSON + mise à jour REGISTER.md *(code)*

#### Ce que fait la cellule

1. Ajoute les métadonnées du run dans `results` : date, clusters, nombre de fonctions, etc.
2. Sérialise `results` (le dictionnaire accumulé depuis le début du notebook) en JSON
3. Écrit `eda_results_{RUN_DATE}.json` en local ET sur Drive (si Colab)
4. Lit `REGISTER.md`, y ajoute une ligne avec les métriques clés du run, réécrit le fichier

#### Pourquoi conserver les résultats en JSON plutôt qu'en CSV ou autre

Le dictionnaire `results` contient des structures hétérogènes : listes de records (19 lignes
pour les métriques par fonction), dictionnaires imbriqués (périodes FFT par cluster par
fonction), entiers simples (n_stationary). JSON est le seul format texte standard qui
représente natif ces structures hiérarchiques.

Avantage pratique : `json.loads(open('eda_results_2026-05-02_08-31.json').read())` recharge
instantanément tous les résultats dans un script Python ultérieur — utile pour générer des
tableaux LaTeX ou comparer plusieurs runs.

---

### Cellule 47 — Téléchargement navigateur (Colab) *(code)*

Déclenche `colab_files.download()` pour télécharger le JSON automatiquement dans
`~/Téléchargements` du navigateur.

**Pourquoi** : le JSON sur Drive est accessible de partout, mais nécessite de se connecter
à Drive pour le récupérer. La copie locale immédiate dans les Téléchargements est plus
pratique pour le déposer directement dans le dossier `code/experiments/eda/` sans passer
par l'interface Drive.

---

### Cellule 48 — Instructions archivage HTML *(markdown)*

Rappel en 4 étapes :
1. Fichier → Télécharger → `.html` (contient toutes les figures)
2. Fichier → Télécharger → `.ipynb` (optionnel — contient les outputs)
3. Copier le `.html` dans `code/experiments/eda/`
4. Le JSON a déjà été téléchargé automatiquement par la cellule 47

**Pourquoi le HTML est indispensable** : Colab ne conserve pas les figures générées par
`matplotlib.pyplot.show()` entre les sessions. Une fois la session Colab fermée, toutes les
figures disparaissent. Le HTML exporte le notebook *avec ses outputs* — toutes les figures
sont intégrées en base64 dans le fichier HTML. Il est consultable hors-ligne, dans n'importe
quel navigateur, sans Python ni Jupyter.

---

## Résumé du flux d'exécution et rôle de chaque section

```
[02] imports  →  [03] env + DATA_DIR  →  [04] chargement CSV
       ↓
[06] stats par fonction  →  [07] résumé par cluster  →  [08] ► capture overview
       ↓
[10] describe() — percentiles
       ↓
[12-15] séries brutes (14j + zoom + heatmap + profil journalier)
       ↓
[17] stats zéros + runs consécutifs  →  [18] ► capture zéros  →  [19-20] graphiques
       ↓
[22-24] distributions (histogramme + KDE + boxplot + skewness/kurtosis)
       ↓
[26] ACF 48h  →  [27] FFT texte  →  [28] ► capture FFT  →  [29] périodogramme
       ↓
[31] test ADF  →  [32] ► capture ADF  →  [33] ADF après diff (secours)
       ↓
[35] corrélations Pearson+Spearman  →  [36] distance euclidienne normalisée
       ↓
[38] CV+Burstiness boxplot  →  [39] profils normalisés overlay  →  [40] tableau inter
       ↓
[42] synthèse ASCII + recommandations  →  [43] tableau de bord 4 métriques  →  [44] scatter
       ↓
[46] JSON + REGISTER.md  →  [47] download  →  [48] rappel HTML
```

Les cellules marquées `► capture` (8, 18, 28, 32) ne produisent aucune sortie visible.
Elles alimentent silencieusement le dictionnaire `results` tout au long de l'exécution,
pour la sauvegarde finale en JSON.

---

## Glossaire rapide (pour présentation aux encadreurs)

| Terme | Définition en une ligne |
|---|---|
| **Stationnarité** | La série n'a pas de tendance à la hausse ou à la baisse dans le temps |
| **CV** | Rapport std/moyenne en % — mesure la variabilité relative indépendamment de l'amplitude |
| **Burstiness B** | Indice entre −1 et +1 mesurant si le trafic est régulier (B<0) ou en rafales (B>0) |
| **ACF** | Corrélation de la série avec elle-même décalée de k unités de temps |
| **FFT** | Décomposition du signal en cycles de différentes fréquences |
| **KDE** | Histogramme lissé — version continue de la distribution |
| **Skewness** | Asymétrie de la distribution (0 = symétrique, >0 = queue à droite) |
| **Kurtosis** | Lourdeur des queues (0 = gaussienne, >0 = plus d'outliers que la gaussienne) |
| **Pearson** | Corrélation linéaire (sensible aux outliers) |
| **Spearman** | Corrélation des rangs (robuste aux outliers) |
| **Zero-inflaté** | Distribution dominée par des zéros, avec des pics rares et intenses |
| **Différenciation** | Transformer `serie[t]` en `serie[t] - serie[t-1]` pour supprimer les tendances |
| **IQR** | Écart entre p25 et p75 — la zone où se trouvent les 50 % centraux des données |
