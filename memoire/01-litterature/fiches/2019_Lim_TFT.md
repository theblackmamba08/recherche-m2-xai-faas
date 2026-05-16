---
type: fiche-lecture
article: 2019_Lim_TFT.pdf
auteurs: Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister
annee: 2019 (arXiv) / 2021 (IJF)
journal_conf: International Journal of Forecasting, 37(4), pp. 1748–1764 — Google Cloud AI
date_lecture: 2026-05-16
pertinence: 5
tags: [transformer-ts, interpretability, attention, variable-selection, baseline-xai, concurrent-h1, tft]
---

# Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting

> Citation BibTeX-key : `Lim2021TFT` *(à reporter dans `redaction/biblio/refs.bib`)*

> ⚠️ **Concurrent direct de H1.** TFT est un Transformer temporel *intrinsèquement interprétable* déployé en production (Google) et publié dans IJF 2021. Il précède H1 et **invalide tout claim de primauté** ("premier Transformer TS interprétable"). Argument révisé de H1 : *alternative à TFT avec fidélité par construction*, là où TFT reste tributaire de la critique "attention ≠ explanation" (Jain & Wallace, 2019).

## Problème traité

Les méthodes de prévision multi-horizons ont deux limites persistantes :
1. **Performance** : les méthodes statistiques (ETS, ARIMA) n'exploitent pas les covariables hétérogènes (statiques + temporelles + connues dans le futur). Les DNN capturent des dépendances complexes mais sont des boîtes noires.
2. **Interprétabilité** : les modèles DL performants (LSTM, DeepAR, N-BEATS) ne permettent pas de savoir *pourquoi* ils prédisent ce qu'ils prédisent. Or les décideurs (gestion des stocks, énergie) ont besoin de comprendre quelles variables comptent et à quel horizon.

TFT propose une architecture unifiée qui optimise les deux simultanément : **meilleures performances ET explications par le mécanisme d'attention**.

## Méthode

### Architecture générale (Fig. 2, p. 4)

Pipeline : `Inputs → Variable Selection Networks → Gated Residual Networks → Static Enrichment → Temporal Self-Attention (IMHA) → Point-wise FFN → Quantile Output`

#### 1. Variable Selection Networks (VSN)

Pour chaque pas de temps, le VSN calcule un vecteur de *softmax weights* (scores d'importance) sur les D variables d'entrée, puis forme la représentation comme **combinaison pondérée** des embeddings de chaque variable :

```
ξ̃_t = Σ_j softmax(ṽ_j) · ξ_j
```

- Fournit une **explication par variable** : l'importance de la feature j au temps t.
- Clé de l'interprétabilité de TFT côté inputs.

#### 2. Gated Residual Networks (GRN)

Couche de traitement non-linéaire avec skip connection et gate sigmoid :

```
GRN(a) = LayerNorm(a + GLU(W₁·ELU(W₂·a + b₂) + b₁))
```

- Adaptatif : si l'information est inutile, le gate peut l'ignorer (skip).
- Pas de couches denses fixes : plus flexible que MLP classique.

#### 3. Static Covariate Encoders

Les features statiques (ex. : type de produit, identifiant d'une série) sont encodées en 4 vecteurs `c_s, c_e, c_h, c_c` qui modulent respectivement la sélection des variables, les enrichissements temporels locaux, l'état initial de l'LSTM, et la couche d'enrichissement statique.

#### 4. Sequence-to-sequence LSTM (locality encoder)

Avant le Transformer, un LSTM bidirectionnel encode le passé et un LSTM forward encode le futur connu. Cette étape capture les dépendances locales à court terme.

#### 5. Interpretable Multi-Head Attention (IMHA)

**Modification clé de l'attention standard pour l'interprétabilité** :

Standard MHA : `Attn(Q_h, K_h, V_h)` avec V_h différent pour chaque tête h.

TFT IMHA : toutes les têtes partagent la **même matrice de valeurs** V :
```
MultiHead(Q, K, V) = [Attn(Q·W_q^h, K·W_k^h, V·W_V)] h=1..H concatenated
Interp. Attn = (1/H) Σ_h softmax(Q·W_q^h · (K·W_k^h)^T / √d_attn) · V·W_V
```

Le résultat est un **unique tenseur d'attention** (moyenne des H têtes) interprétable comme *attribution de l'importance de chaque pas passé pour chaque pas futur*.

### Prévision probabiliste

TFT prédit des quantiles (10e, 50e, 90e percentiles par défaut) via une perte de régression quantile (pinball loss), pas une distribution paramétrique. Cela couvre l'incertitude sans supposer gaussianité.

### Explications fournies par TFT (Section 5, p. 9)

1. **Temporal patterns** : la matrice d'attention moyenne sur l'ensemble de test révèle les lags dominants (ex. : pattern journalier, hebdomadaire).
2. **Variable importance** : les poids VSN agrégés révèlent quelles features comptent le plus, séparément pour les features statiques, les encodeurs passés, et les décodeurs futurs.
3. **Regimes** : le clustering des séries par leur profil d'attention peut révéler des groupes à comportement similaire.

## Résultats clés

Datasets : 9 jeux de données réels couvrant retail (Favorita, M5), énergie (electricity, traffic), finance (volatility), météo (GEFCOM 2014).

- TFT **surpasse** ETS, ARIMA, DeepAR, N-BEATS, TCN, LightGBM sur la majorité des benchmarks (Table 3, p. 8).
- Gagne en particulier sur les datasets avec covariables hétérogènes (statiques + dynamiques connues dans le futur).
- Les VSN montrent des importances cohérentes avec le domaine : pour Favorita, les promotions sont la feature la plus importante ; pour electricity, l'heure de la journée domine.

## Limites identifiées par les auteurs

- La IMHA partage V — une seule représentation de valeurs pour toutes les têtes — ce qui limite l'expressivité de l'attention multi-têtes.
- Complexité O(n²) de l'attention limite la longueur du contexte.
- L'interprétabilité repose sur l'attention : les auteurs *assument* que les poids d'attention reflètent les contributions réelles, sans le prouver formellement.

## Limites identifiées par MOI — point critique pour H1

### La critique "attention ≠ explanation" s'applique à TFT

Jain & Wallace (2019, *ACL*) ont démontré expérimentalement que :
- On peut perturber les poids d'attention (les redistribuer uniformément, les permuter, les inverser) **sans changer la prédiction** de manière significative.
- Conclusion : les poids d'attention ne peuvent pas être interprétés comme mesure causale de l'importance des tokens.

**Implication pour TFT** : les "Temporal patterns" et les "Variable importances" affichés par TFT sont fondés sur des poids d'attention. Ces poids ne sont PAS garantis d'être les vraies contributions algébriques à la prédiction. L'interprétabilité est une *corrélation*, pas une *fidélité prouvée*.

**Argument central de H1 contre TFT** :

| Critère | TFT (Lim 2021) | H1 SoftCAM-Transformer |
|---------|----------------|------------------------|
| Type d'interprétabilité | Attribution par attention (IMHA) | Décomposition linéaire exacte |
| Fidélité garantie ? | **Non** — critique Jain & Wallace 2019 | **Oui** — M[t,s] est le coefficient réel du calcul |
| Backbone modifié | Oui (architecture TFT entière) | Minimalement (evidence layer + ElasticNet) |
| Backbone HuggingFace réutilisé ? | Non | **Oui** (TimeSeriesTransformer HF) |
| Coût d'explication | 0 (intrinsèque) | 0 (intrinsèque) |
| Distributionnel | Quantiles (pinball loss) | Maintenu (NegBin / Student-t selon FAYAM) |

## Lien avec H1 *(notre hypothèse prioritaire)*

### Ce que H1 ne doit PAS revendiquer

- ❌ "Premier Transformer de séries temporelles interprétable" — TFT (2019/2021) l'est déjà.
- ❌ "Meilleure précision que TFT" — pas l'objet de H1 (TFT est entraîné sur des benchmarks différents, comparaison hors-scope).
- ❌ "SoftCAM-Transformer > TFT toutes dimensions confondues" — TFT a des avantages (VSN, GRN, covariables futures connues).

### Ce que H1 DOIT revendiquer

- ✅ **Fidélité par construction** : M[t,s] dans H1 est le coefficient algébrique *exact* de la contribution de encoder_emb[s] à h[t]. Aucune approche post-hoc ni attention approximée.
- ✅ **Réutilisation du backbone HuggingFace** : contrairement à TFT qui est une architecture complète à réimplémenter, H1 *modifie minimalement* un modèle HuggingFace déjà déployé.
- ✅ **Angle FaaS** : ni TFT ni SoftCAM ne traitent la prévision de charge de fonctions serverless. C'est la contribution domaine.
- ✅ **Test différentiel par cluster DBSCAN** : H1 permet de comparer les cartes d'évidence entre clusters (profils de charge distincts) — angle non couvert par TFT dans ses benchmarks.

### Phrase-clé pour la soutenance

> « TFT est déjà interprétable, mais son interprétabilité repose sur des poids d'attention dont Jain & Wallace (2019) ont démontré qu'on peut les permuter sans changer la prédiction. Notre contribution — SoftCAM-Transformer — produit une explication dont chaque poids M[t,s] est algébriquement le coefficient exact du terme encoder_emb[s] dans le calcul de h[t] : c'est une décomposition linéaire prouvée, pas une attribution approximée. »

## Citations à réutiliser

> « We propose a novel attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics at inference time. » (Abstract)

> « By sharing values across heads, the multi-head attention allows us to extract a single set of interpretable attention weights, attributing temporal importance to different time steps. » (Section 4.3, p. 7)

> « [Variable selection networks] provide instance-wise variable selection, allowing TFT to determine the extent to which each input variable and feature contributes to the forecast. » (Section 4.2, p. 6)

## Idées à creuser

- **Comparer les attention patterns TFT vs H1 SoftCAM** sur un même dataset synthétique ou sur C4 (après implémentation H1) : est-ce que M[t,s] corrèle avec IMHA[t,s] ? Si oui → H1 valide TFT ; si non → M[t,s] est plus informatif.
- **Référencer Jain & Wallace (2019)** dans le chapitre "État de l'art" : la critique y est fondamentale pour positionner H1 vs toutes les approches basées attention (TFT, Transformer standard, H3).
- **Ajouter TFT au panorama XAI v3** : canevas Dr LACMOU 4 points — contexte (prévision multi-horizons), problèmes (boîte noire + DL performant), transposabilité (FaaS multi-séries avec covariables statiques), limites (O(n²), attention non fidèle, VSN = soft sélection sans garantie).
