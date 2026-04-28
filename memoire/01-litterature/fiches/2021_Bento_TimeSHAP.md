---
type: fiche-lecture
article: 2021_Bento_TimeSHAP.pdf
auteurs: João Bento, Pedro Saleiro, André F. Cruz, Mário A.T. Figueiredo, Pedro Bizarro
annee: 2021
journal_conf: KDD '21 — Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 14-18 août 2021, Singapour. DOI: 10.1145/3447548.3467166. arXiv:2012.00073v2
date_lecture: 2026-04-28
pertinence: 4
tags: [timeshap, kernelshap, shapley, post-hoc, model-agnostic, rnn, lstm, gru, sequence, temporal, pruning, h2-repli]
---

# TimeSHAP : Expliquer les modèles récurrents par perturbations de séquences

> Citation BibTeX-key : `Bento2021TimeSHAP` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Article fondateur de H2 (repli)**. TimeSHAP est la baseline post-hoc de référence pour les modèles récurrents. Si H1 (SoftCAM-Transformer) ne converge pas, c'est l'approche à déployer directement sur le Transformer FAYAM.

## Problème traité

KernelSHAP et les méthodes SHAP standard n'attribuent de l'importance qu'aux features du pas de temps courant, ignorant les événements passés et l'état caché du modèle (Fig. 1, p. 1). Pour les RNN (LSTM, GRU), la prédiction dépend de toute la séquence historique — ignorer cette dimension donne des explications incomplètes et potentiellement trompeuses. Les auteurs proposent TimeSHAP, une extension model-agnostique et post-hoc de KernelSHAP au domaine séquentiel.

## Méthode

### Architecture générale

TimeSHAP adapte KernelSHAP aux séquences en définissant des **perturbations séquentielles** bidimensionnelles sur la matrice d'entrée `X ∈ ℝ^{d×l}` (d features × l événements). La valeur de fond (background) `B ∈ ℝ^{d×l}` est la matrice des moyennes des features dans le jeu d'entraînement.

### Trois types d'explications (Section 3.1-3.3)

1. **Explicabilité features** (`h_X^f`) : perturbe les *lignes* (features) de X — indique quelles variables ont le plus influencé la prédiction, agrégées sur toute la séquence.
   ```
   h_X^f(z) = D_z X + (I − D_z)B,   D_z = diag(z)
   ```

2. **Explicabilité événements** (`h_X^e`) : perturbe les *colonnes* (pas de temps) de X — indique quels instants passés ont le plus contribué à la prédiction.
   ```
   h_X^e(z) = XD_z + B(I − D_z),   D_z = diag(z)
   ```

3. **Explicabilité cellules** : intersection des features et événements les plus importants — granularité maximale, identifie les cellules `(feature, instant)` les plus critiques. Nécessite un groupement préalable pour rester calculable.

### Élagage temporel (Temporal Coalition Pruning — Algorithme 1)

Problème : le nombre de coalitions d'événements croît en `O(2^l)` avec la longueur de la séquence. Solution : grouper les événements anciens (peu importants) en une seule coalition. L'algorithme parcourt la séquence de la fin vers le début et s'arrête à l'indice `i` où l'importance agrégée `|w_1|` passe sous le seuil `η`.

- Complexité réduite de `O(2^l)` à `O(l · 2^{l−i})`.
- Avec `η = 0.025` (valeur retenue) : la séquence médiane passe de 138,5 événements à **14,0 événements** (Table 1, p. 6).

### Évaluation expérimentale

- **Modèle** : GRU + couche d'embedding + classifieur feed-forward (Feedzai, détection de fraude bancaire).
- **Données** : ~20M d'instances, 3 types d'événements (login ≈70 %, transaction ≈20 %, enrollment ≈10 %). Données géolocalisation + démographie associées.
- **Performance** : Recall 84,3 % à 1 % FPR (validation) ; 79,9 % à 0,89 % FPR (test).
- **Paramètres TimeSHAP** : `n_samples = 32K` coalitions (meilleur compromis coût/variance) ; `η = 0.025` ; `θ = 0.1` (seuil pertinence cellules).

## Résultats clés

- **Élagage** (Table 1, p. 6) :
  - Longueur médiane originale : 138,5 événements → après élagage `η=0.025` : **14,0**
  - RSD (dispersion relative des valeurs Shapley) : 1,71 (brut) → **0,98** (η=0,025) → meilleure stabilité sans sacrifier la granularité
  - 58,3 % des séquences peuvent être réduites à moins de `log₂(32000) ≈ 15` événements

- **Importance globale par événement** (Fig. 3, p. 6) :
  - L'événement `t=0` (le plus récent) contribue en moyenne à **41 %** du score du modèle
  - Les événements précédents (`t < 0`) représentent **59 %** → le modèle conserve bien l'information historique dans son état caché

- **Importance globale par feature** (Fig. 4, p. 6-7) :
  - Top features : Transaction type (0,29), Event type (0,092), Client age (0,090)
  - Features de localisation IP : 0,08 à 0,03

- **Étude de cas — pattern fraud détecté** : séquence enrollment-login-transaction (t=−4 à t=0) identifiée comme signature typique de fraude par takeover → cohérent avec expertise métier.

- **Biais détecté** : forte attribution à l'âge client → confirmé a posteriori par audit de biais (taux de faux positifs plus élevés pour les clients âgés, p. 8).

## Limites identifiées par les auteurs

- Validé uniquement sur un cas de détection de fraude (classification binaire, GRU).
- L'algorithme d'élagage suppose que les événements anciens sont moins importants — pas vrai dans tous les domaines (ex. anomalie détectée à un instant lointain dans une série IoT).
- Calcul exact des valeurs Shapley reste exponentiel pour les séquences longues non élagage.

## Limites identifiées par MOI (lecteur)

1. **Exclusivement pour la classification**. TimeSHAP explique un score de probabilité (fraude/non-fraude). Pour la *prévision de séries temporelles* (FAYAM prédit un nombre d'invocations FaaS), il faudra adapter la notion de « score à expliquer » — quelle sortie ? quelle fenêtre de prédiction ?
2. **Architecture RNN (GRU/LSTM) uniquement**. Le Transformer n'a pas d'état caché récurrent au sens strict : l'information passée est transmise par self-attention, pas par un hidden state. L'approche de perturbation reste applicable (boîte noire), mais le fondement théorique « hidden state » ne s'y transpose pas directement.
3. **Background B = moyenne du dataset**. Ce choix est standard mais discutable : les moyennes peuvent être non représentatives (distributions multimodales, données FaaS avec profils très hétérogènes).
4. **Pas de comparaison avec des méthodes intrinsèques**. Aucun benchmark contre SoftCAM ou des modèles self-explainable. La comparaison reste intra-catégorie post-hoc.
5. **Coût computationnel résiduel élevé**. Même avec élagage, 32K coalitions × longueur résiduelle peut représenter un budget GPU/CPU significatif sur de grands datasets FaaS.
6. **Pas testé sur séries temporelles multivariées continues**. Le dataset de fraude est tabular-séquentiel (events discrets) — les données d'invocations FaaS sont des séries continues avec des lags saisonniers complexes.

## Lien avec H1 *(notre hypothèse prioritaire — SoftCAM-Transformer)*

TimeSHAP est la **référence post-hoc directe** pour H2, mais éclaire aussi H1 de manière indirecte :

| Dimension | TimeSHAP (H2) | SoftCAM-Transformer (H1) |
|-----------|--------------|--------------------------|
| Paradigme | Post-hoc, après entraînement | Intrinsèque, modifie l'architecture |
| Coût à l'inférence | Perturbations multiples (coûteux) | 1 forward pass |
| Type d'explication | Valeurs Shapley par (feature, instant) | Carte d'évidence temporelle continue |
| Fidélité | Approximation par échantillonnage | Exacte par construction |
| Applicable au Transformer | Oui (boîte noire) | Oui (si adaptation réussie) |

**Si H1 échoue** : TimeSHAP s'applique directement au TimeSeriesTransformer de FAYAM en mode boîte noire (accès uniquement aux entrées et à la prédiction via l'API d'inférence).

**Comme outil de validation** : les explications événements de TimeSHAP (`h_X^e`) sont conceptuellement analogues aux cartes d'évidence temporelles de H1 — comparer les deux devrait constituer un *sanity check* : si TimeSHAP et SoftCAM-Transformer s'accordent sur les instants importants, H1 est crédible.

**Métriques communes** : TimeSHAP ne définit pas de comprehensiveness/sufficiency explicitement, mais on peut les calculer a posteriori à partir des valeurs Shapley. Cela permettrait une comparaison directe H1 ↔ H2 sur la même grille d'évaluation.

## Citations à réutiliser

> « Blindly applying state-of-the-art explainers to RNNs disregards the importance of past events and the features throughout the sequence, only attributing importance to features of the current input. » (p. 1)

> « TimeSHAP computes feature-, timestep-, and cell-level attributions. As sequences may be arbitrarily long, we further propose a temporal coalition pruning method that is shown to dramatically decrease both its computational cost and the variance of its attributions. » (Abstract, p. 1)

> « The most recent input event of positive predictions only contributes on average to 41% of the model's score. » (Abstract, p. 1)

> « Our method is suited to explain the predictions of any recurrent model, regardless of architecture, only requiring access to the features of each instance and an inference API, restricting our approach to be perturbation-based, post-hoc, and model-agnostic. » (Conclusion, p. 8)

## Idées à creuser

- **Adapter la cible d'explication** : pour FAYAM, définir le « score à expliquer » — moyenne des prédictions sur l'horizon, ou prédiction à un pas spécifique (pic de charge) ?
- **Background adapté au FaaS** : remplacer la moyenne globale par la moyenne intra-cluster DBSCAN — chaque cluster aurait son propre background, rendant les valeurs Shapley comparables au sein d'un profil de charge.
- **Pruning temporel** : le seuil `η` est à calibrer sur les séries FaaS — les invocations ont des saisonnalités marquées (journalière, hebdomadaire), donc les événements anciens peuvent être importants.
- **Librairie** : TimeSHAP est disponible sur GitHub (`feedzai/timeshap`) — vérifier la compatibilité avec HuggingFace `TimeSeriesTransformer` avant de choisir H2 comme repli.
- **Comparaison H1 ↔ H2** : si H1 aboutit, produire les deux types d'explications sur les mêmes instances et calculer la corrélation entre la carte d'évidence SoftCAM et les valeurs Shapley événements TimeSHAP — une forte corrélation valide H1.
