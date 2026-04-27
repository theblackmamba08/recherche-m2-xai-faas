---
type: fiche-lecture
article: 2024_FAYAM_FaaS-prediction.pdf
auteurs: FAYAM MBALA MOUEN Alexandre Savi (encadrant : Pr. KENGNE TCHENDJI Vianney)
annee: 2024
journal_conf: Mémoire de Master 2 — Université de Dschang, URIFIA (juillet 2024)
date_lecture: 2026-04-26
pertinence: 5
tags: [transformer, faas, cold-start, dbscan, azure-functions, openwhisk, baseline-h1, lstm, cnn-lstm]
---

# Prédiction des charges de travail et atténuation du cold start dans les architectures FaaS

> Citation BibTeX-key : `FAYAM2024Prediction` *(à reporter dans `redaction/biblio/refs.bib`)*

> ⚠️ **Mémoire baseline** : c'est l'étude que H1 cherche à étendre par la couche XAI (attention + faithfulness + analyse inter-clusters DBSCAN).

## Problème traité

L'auteur traite **deux verrous FaaS distincts mais couplés** : (1) la **prédiction de la charge de travail** (nombre d'invocations par unité de temps) et (2) l'**atténuation du cold start** (délai et fréquence). Le mémoire propose **deux modèles** : un CNN-LSTM hybride pour le premier verrou, et un Transformer adapté aux séries temporelles pour le second. Le travail s'appuie sur des données réelles (Azure Functions 2019) et déploie la solution Transformer dans OpenWhisk pour valider l'approche en conditions quasi-production.

## Méthode

### Données
- **Dataset** : Azure Functions Trace, juillet 2019 (sous-ensemble « comptes et déclencheurs des invocations »).
- **Volume** : 14 fichiers (un par période de 24 h), 1440 champs par fichier (un par minute).
- **Filtrage** : seules les fonctions au déclencheur HTTP sont retenues (« les plus imprévisibles et populaires »).
- **Clustering** : DBSCAN appliqué sur les motifs d'invocation → **33 clusters distincts** de fonctions HTTP. Les fonctions représentatives sont tirées aléatoirement dans chaque cluster (figure 18).
- **Identifiants** : table IX (p. 86) liste les `HashFunction` des 18 datasets retenus pour reproductibilité.

### Modèles

**Modèle 1 — CNN-LSTM (prédiction de charge, non retenu pour H1)**
- Couches CNN pour extraction de features locaux + LSTM pour dépendances temporelles.
- Comparé à un LSTM seul.

**Modèle 2 — Transformer (atténuation cold start, BASELINE H1)**
- Architecture *encodeur-décodeur* TimeSeriesTransformer (HuggingFace).
- Encodeur reçoit `pastValues + pastTimeFeatures + staticCategoricalFeatures + staticRealFeatures`.
- Décodeur reçoit `futureTimeFeatures + dernière pastValue + représentation encodée`.
- Hyperparamètres explicites : `encoderLayers=4`, `decoderLayers=4`, `dModel=32`. `predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension` mentionnés mais valeurs non chiffrées (non trouvé p. 76).
- Comparé à un LSTM 5 couches × 32 neurones, ReLU, Dropout 0,5, sortie linéaire, MSE comme loss (p. 71).

### Environnement d'exécution
- **Google Colab** (Google Compute Engine), Python 3.
- **GPU** activé. RAM système : 12,7 GB. RAM GPU : 15,0 GB.

### Plateforme de déploiement
- **OpenWhisk** : le Transformer ajuste dynamiquement la *fenêtre de conteneur inactif* (FCI) en remplacement du paramétrage statique par défaut (10 min).

### Métriques
sMAPE, Explained Variance, RMSE, Normalized RMSE, R² Score, Spearman Correlation (formules p. 79-80). Comparaison Transformer vs LSTM sur 12 datasets.

## Résultats clés

### CNN-LSTM (charge de travail, p. 89)
- **RMSE = 4,10** vs LSTM littérature **6,96** → réduction nette.

### Transformer vs LSTM (cold start — délai, tables VI et VII)
- **Transformer surpasse LSTM sur la majorité des métriques et datasets**.
- Exemples (Table VI, fréquence horaire) :
  - Dataset 5 : Explained Variance Transformer **0,837** vs LSTM 0,797 ; R² **0,814** vs 0,796 ; Spearman **0,797** vs 0,762.
  - Dataset 6 : Spearman **0,957** (Transformer) vs 0,865 (LSTM) ; R² **0,835** vs 0,686.
- Exemples (Table VII, fréquence minute) :
  - Dataset 12 : sMAPE **0,043** (Transformer) vs 0,108 (LSTM) ; R² **0,958** vs 0,711.
  - Dataset 7 : R² **0,700** vs -0,009 ; Explained Variance **0,717** vs 0.

### OpenWhisk + Transformer (cold start — fréquence, table VIII)
- **OW par défaut** = FCI fixe à 10 min → 62 à 100 cold starts pour 100 invocations.
- **OW + Transformer** = FCI dynamique adaptée par fonction (intervalles 3-141 min selon dataset).
- **Réductions observées** :
  - Dataset 13 : 100 → 45 cold starts (**-55 %**).
  - Dataset 14 : 66 → 14 (**-79 %**, maximum rapporté).
  - Dataset 15 : 72 → 45 (-37 %).
  - Dataset 16 : 71 → 25 (**-65 %**), avec FCI variant entre 5 et 141 min.
  - Dataset 17 : 62 → 38 (-39 %).
  - Dataset 18 : 66 → 17 (**-74 %**).

## Limites identifiées par les auteurs

L'auteur **ne consacre pas de section explicite aux limites**. Les *Perspectives* (p. 89-90) suggèrent toutefois des manques implicites :

- Architectures Transformers *non explorées* au-delà du modèle vanilla adapté aux séries temporelles.
- Absence de mécanisme d'**adaptation en ligne** (reinforcement learning suggéré comme piste future).
- Gestion des ressources jugée perfectible (« mécanismes plus sophistiqués » à développer).
- Données limitées à *une seule plateforme* (Azure Functions). L'intégration multi-plateformes est listée comme axe futur.

## Limites identifiées par MOI (lecteur)

Critique vis-à-vis de **H1 (XAI : attention + DBSCAN + faithfulness)** :

1. **Aucune analyse XAI du Transformer**. Le modèle est traité comme une boîte noire : pas d'extraction des `attention_weights`, pas de visualisation, pas d'interprétation des prédictions. *C'est exactement le créneau de H1.*
2. **DBSCAN sous-exploité**. Les 33 clusters servent uniquement à échantillonner 12 datasets de manière « représentative » — il n'y a **aucune analyse différentielle inter-clusters** (un même Transformer est appliqué globalement, sans étude des comportements d'attention par profil de charge).
3. **Pas de métrique de fidélité** (comprehensiveness, sufficiency). Le mémoire évalue la qualité de prédiction, pas la qualité d'explication.
4. **Hyperparamètres incomplets** : `predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension` ne sont jamais chiffrés (p. 76). Reproduire la baseline exigera de fixer ces valeurs (à demander à l'encadrant ou à inférer).
5. **Échantillonnage aléatoire dans les clusters** (p. 74) → biais possible si les fonctions choisies ne sont pas représentatives. Pas de moyenne sur plusieurs tirages.
6. **Pas de code public** mentionné. La reproductibilité repose sur les `HashFunction` du tableau IX et sur la reconstruction du pipeline.
7. **Comparaison limitée à LSTM**. Pas de comparaison à des baselines plus simples (ARIMA, Prophet, naïf saisonnier) — utile pour situer le gain réel.
8. **Pas de découpage train/val/test explicité** dans les pages lues. À vérifier dans les pages 65-78 que je n'ai pas relues en détail.

## Lien avec H1 *(notre hypothèse prioritaire)*

H1 = **étendre la baseline FAYAM en activant `output_attentions=True`, en projetant les cartes d'attention par cluster DBSCAN, et en mesurant la fidélité (comprehensiveness/sufficiency) des explications**.

- **Compatibilité forte** : FAYAM utilise déjà le `TimeSeriesTransformer` de HuggingFace, qui expose nativement les attentions encodeur/décodeur. Aucune modification d'architecture nécessaire.
- **Apport scientifique** : H1 comble exactement ce que FAYAM laisse de côté (interprétabilité par profil de charge). On passe de « le Transformer prédit mieux » à « **voici pourquoi**, et **pour quels types de fonctions**, il prédit mieux ».
- **Réutilisation directe** : les 33 clusters DBSCAN, les 18 datasets et leurs `HashFunction` (tableau IX) constituent un terrain prêt à l'emploi.
- **Risque** : si les hyperparamètres exacts ne sont pas récupérables, la reproduction exacte de la baseline (étape 1 de la roadmap) prendra plus de temps que prévu.

## Citations à réutiliser

> « le modèle Transformers dépasse systématiquement le modèle LSTM sur presque toutes les métriques et pour tous les datasets testés » (p. 80)

> « Une réduction allant jusqu'à 79 % des cold starts a été observée pour certaines fonctions, ce qui démontre l'efficacité du modèle Transformers » (p. 84)

> « notre approche utilise un modèle unique pour traiter à la fois les problèmes de délai de cold start et de fréquence de cold start. Cela simplifie grandement le processus de déploiement et de maintenance » (p. 85)

> « Lors du traitement de ces données, seules les instances de fonctions déclenchées par HTTP ont été sélectionnées. Cette décision a été prise car les invocations HTTP sont les plus imprévisibles et les plus populaires dans les architectures FaaS. » (p. 73)

> « Les résultats de la clusterisation ont montré que les fonctions HTTP déclenchées pouvaient être regroupées en 33 clusters distincts. » (p. 74)

## Idées à creuser

- **Récupérer les hyperparamètres manquants** (`predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension`) — soit via le code FAYAM si fourni, soit en les fixant nous-mêmes et en documentant l'écart dans `02-baseline/MEMOIRE.md`.
- **Activer `output_attentions=True`** dès la phase 1 (reproduction) pour ne pas avoir à relancer l'entraînement au moment de H1.
- **Recalculer les 33 clusters DBSCAN** (ou récupérer les labels) pour pouvoir indexer les attention maps par cluster.
- **Définir 3-5 macro-profils** parmi les 33 clusters (peu fréquentes / régulières / populaires — déjà esquissé p. 75 figure 18) pour rendre l'analyse différentielle lisible dans le mémoire.
- **Tester la fidélité** (comprehensiveness / sufficiency) sur quelques fonctions représentatives par profil avant de généraliser.
- **Vérifier le découpage temporel** : sur 14 fichiers de 24 h, comment est constitué l'ensemble de test ? Risque de fuite si le split n'est pas chronologique strict.
- **Comparer l'attention moyenne aux lags effectifs** (`lagsSequence`) : voir si le Transformer s'aligne sur les périodicités attendues (jour, semaine).
