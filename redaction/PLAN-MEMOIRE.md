# Plan du mémoire — XAI pour la prédiction de charge FaaS

> Document de travail. À relire avec les encadreurs avant de commencer la rédaction.
> Dernière mise à jour : 2026-05-28.

---

## Titre de travail

**Explicabilité intrinsèque pour la prédiction de charge FaaS : un Transformer à couche d'évidence régularisée**

> ⚠️ Le nom du modèle ("SoftCAM-Transformer") est à remplacer — décision en attente avec les encadreurs. Placeholder dans ce document : **[NOM-MODÈLE]**.

---

## Structure générale

| Fichier template | Contenu | Volume estimé |
|---|---|---|
| `Resume/Resume.tex` | Résumé français | ~1 page |
| `Abstract/Abstract.tex` | Résumé anglais | ~1 page |
| `Introduction/Introduction.tex` | Introduction générale | 5–8 pages |
| `Chap1/Chap1.tex` | FaaS et prédiction de charge | 18–22 pages |
| `Chap2/Chap2.tex` | XAI pour les séries temporelles | 18–22 pages |
| `Chap3/Chap3.tex` | Architecture [NOM-MODÈLE] | 18–20 pages |
| `Chap4/Chap4.tex` | Résultats et évaluation | 15–18 pages |
| `Conclusion/Conclusion.tex` | Conclusion générale | 4–6 pages |
| `Annexes/` | Hyperparamètres, extraits de code, runs | 5–10 pages |

**Total estimé : 85–100 pages** (hors pages de garde, tables, bibliographie).

---

## Ordre de rédaction conseillé

Ne pas écrire dans l'ordre des chapitres. Aller du plus concret au plus abstrait :

1. **Chapitre 4** (résultats) — les données sont là, c'est le plus facile à remplir maintenant
2. **Chapitre 1** (FaaS + FAYAM) — contexte bien maîtrisé, sources disponibles
3. **Chapitre 2** (XAI) — état de l'art bien consolidé après les Q&A
4. **Chapitre 3** (contribution) — attendre la décision dot product / renommage
5. **Introduction + Conclusion** — toujours en dernier

---

## Introduction générale

> Fichier : `Introduction/Introduction.tex` — commande : `\myChapterStar`

### Sections

**1. Le cloud computing et l'essor du serverless**
- Du IaaS/PaaS au FaaS : pourquoi les entreprises adoptent le serverless
- FaaS = exécution à la demande, facturation à l'invocation, cold-start comme problème central
- Enjeux économiques : AWS Lambda, Azure Functions, Google Cloud Functions

**2. La prédiction de charge : un besoin critique**
- Allouer les ressources avant le pic, pas après
- Dataset Azure Functions 2019 : 18 fonctions représentatives, profils très hétérogènes
- FAYAM : première réponse par Transformer (résultats : R²=0.37 global)

**3. Problématique : la boîte noire au cœur de la décision**
- Un opérateur ne déploie pas une boîte noire pour des décisions coûteuses — Lipton (2018)
- Sans explication, impossible de distinguer vrai pattern et artifact de données — Rudin (2019)
- EU AI Act (2024) : traçabilité des décisions automatisées = obligation légale
- **Question centrale** : peut-on rendre ce Transformer explicable sans sacrifier sa précision ?

**4. Objectifs et hypothèse principale**
- Objectif : ajouter une couche d'explication intrinsèque au Transformer de FAYAM
- Hypothèse H1 : une couche d'évidence M régularisée par ElasticNet produit une explication fidèle par construction
- Replis envisagés (H2 SHAP, H3 attention) et pourquoi H1 a été poursuivie en priorité

**5. Contributions**
- Architecture [NOM-MODÈLE] : Evidence Layer M + régularisation ElasticNet
- Protocole de validation : 7 hypothèses H1.A→H1.G
- Résultats : R²=0.7563 sur Cluster 4 (+38.6 pp vs FAYAM)

**5bis. Plus-value de la contribution** *(section stratégique — à formuler clairement)*

La plus-value se lit sur trois axes :

| Comparaison | Ce que l'existant fait | Ce que [NOM-MODÈLE] ajoute |
|---|---|---|
| vs FAYAM | Prédit bien (R²=0.37) mais boîte noire | Prédit mieux (R²=0.7563) **et** explique via M |
| vs post-hoc (TimeSHAP) | Explication disponible mais approximée + coût 2^n | Explication fidèle par construction + coût 1 passe |
| vs TFT | Poids d'attention natifs mais non régularisés | M régularisée (ElasticNet) → forcée à pointer les vrais moments clés |

**Argument central** : [NOM-MODÈLE] est la seule approche qui améliore simultanément la performance prédictive **et** produit une explication fidèle par construction, à coût constant. Ce n'est pas un compromis interprétabilité/précision — c'est une amélioration des deux.

**6. Organisation du mémoire** (guide de lecture)

---

## Chapitre 1 — FaaS et prédiction de charge : contexte et travaux existants

> Fichier : `Chap1/Chap1.tex` — commande : `\myChapter`

### 1.1 Le paradigme Function-as-a-Service

**1.1.1 Du cloud computing au serverless**
- IaaS → PaaS → SaaS → FaaS : évolution du modèle de déploiement
- Définition FaaS : fonction stateless, exécutée à la demande, durée limitée
- Acteurs : AWS Lambda (2014), Azure Functions (2016), Google Cloud Functions

**1.1.2 Caractéristiques des charges FaaS**
- Forte variabilité inter-fonctions (profils radicalement différents)
- Saisonnalité 24h universelle (confirmé EDA : 19/19 fonctions)
- Fonctions zero-inflated (ex. Cluster 6 : nombreux timesteps à 0)
- Sparsité temporelle : les pics sont rares mais coûteux

**1.1.3 Enjeux de la prédiction**
- Cold-start : une fonction non chargée met du temps à démarrer → latence utilisateur
- Over-provisioning vs under-provisioning : gaspillage vs dégradation de service
- Fenêtre de prédiction : 120 minutes (prediction_length) sur 240 minutes de contexte

### 1.2 Dataset Azure Functions 2019

**1.2.1 Description générale**
- 14 jours d'invocations, résolution 1 minute
- 18 fonctions sélectionnées par FAYAM (HashFunction mapping — tableau IX p. 86 FAYAM)
- 4 clusters utilisés dans ce travail : C0, C4, C6, C8

**1.2.2 Analyse exploratoire (EDA)**
- Stationnarité : 19/19 fonctions stationnaires (ADF test)
- Période dominante : 24h (FFT)
- Zero-inflation : C6 spécifique → traitement dédié
- Corrélation inter-fonctions par cluster : cohérence intra-cluster validée

**1.2.3 Clustering DBSCAN de FAYAM**
- 33 clusters identifiés par FAYAM sur le dataset complet
- Cluster 4 : 19 fonctions, profil régulier, meilleure convergence → cible principale
- Cluster 0 : cas-limite (R²=−0.02 en baseline multi-task)
- Cluster 6 : zero-inflaté → traitement spécifique requis

### 1.3 FAYAM : baseline Transformer pour la prédiction FaaS

**1.3.1 Architecture TimeSeriesTransformer HuggingFace**
- Encodeur : 4 couches, d_model=32, context_length=240
- Décodeur : 4 couches, prediction_length=120
- Distribution de sortie : Student-t (robuste aux valeurs extrêmes)
- Fréquence : 1 minute ("1T")

**1.3.2 Résultats FAYAM**
- R²=0.3701 (moyenne globale sur tous les clusters)
- Spearman=0.92 sur Cluster 4
- Limite : modèle opaque, aucune explication des prédictions

**1.3.3 Reproductibilité**
- Hyperparamètres retrouvés dans `tsf_transf.py` — Run A reproduit R²=0.5299 sur C4 seul + HPO
- Écart Run A vs FAYAM : C4 seul (favorable) + HPO (optimisé par Optuna, 15 trials)

### 1.4 Synthèse et positionnement

- Tableau comparatif modèles de prédiction FaaS (classiques vs DL)
- Conclusion : FAYAM est une bonne baseline prédictive mais ne répond pas au besoin d'explicabilité
- Transition vers Chapitre 2 : quelle méthode XAI choisir ?

---

## Chapitre 2 — XAI pour les séries temporelles : état de l'art

> Fichier : `Chap2/Chap2.tex`

### 2.1 Définitions fondamentales

**2.1.1 Interprétabilité vs Explicabilité**
- Interprétabilité : propriété intrinsèque du modèle — un humain peut suivre le raisonnement directement (Rudin 2019)
- Explicabilité : capacité à produire une description post-hoc du comportement d'un modèle opaque
- Différence fondamentale : fidélité exacte (interprétabilité) vs approximation (explicabilité)
- Sources : Rudin (2019) *Stop Explaining Black Box ML Models*, Lipton (2018) *The Mythos of Model Interpretability*

**2.1.2 Taxonomie**
- Intrinsèque vs post-hoc
- Globale (comportement général) vs locale (décision individuelle)
- Modèle-agnostique vs spécifique à l'architecture

**2.1.3 Propriétés d'une vraie explication (Rudin 2019)**
- Simulable par un humain
- Basée sur des features sémantiquement signifiantes
- Fidèle au modèle par construction (pas par approximation)

### 2.2 Méthodes post-hoc par gradient

**2.2.1 Vanilla Gradients (Simonyan et al. 2013)**
- `∂ŷ/∂x` : sensibilité locale de la sortie à l'entrée
- Problème : gradient nul sur fonctions saturées

**2.2.2 SmoothGrad (Smilkov et al. 2017)**
- Moyenne de gradients sur versions bruitées : `E[∂ŷ/∂(x+ε)]`
- Cartes plus propres mais toujours sensibilité locale

**2.2.3 Integrated Gradients (Sundararajan et al. 2017)**
- Intégration du gradient sur chemin baseline→entrée
- Satisfait axiomes sensibilité + implémentation invariance
- Utilisé via Captum (PyTorch) — H2 de repli envisagé

**2.2.4 GradCAM (Selvaraju et al. 2017)**
- Combine gradients et activations d'une couche convolutive
- Produit une carte de chaleur spatiale
- Ancêtre direct de SoftCAM → lien explicite à faire

**2.2.5 Limites communes**
- Mesure de sensibilité locale, pas de fidélité globale
- Critique de Rudin : approximation externe ≠ explication réelle

### 2.3 Méthodes post-hoc par perturbation

**2.3.1 LIME (Ribeiro et al. 2016)**
- Modèle linéaire local ajusté sur voisinage perturbé
- Problème : choix du voisinage arbitraire, instabilité

**2.3.2 SHAP / KernelSHAP (Lundberg & Lee 2017)**
- Valeurs de Shapley : contribution marginale moyenne sur toutes coalitions
- Axiomes : efficacité, symétrie, dummy, additivité
- KernelSHAP : variante par perturbation pondérée Shapley
- Complexité : 2^n évaluations (approximé en pratique)

**2.3.3 Adaptations aux séries temporelles**
- TimeSHAP (Bento et al. 2021) : perturbation de segments temporels
- TsSHAP : surrogate XGBoost — fidélité dépend du surrogate
- WindowSHAP : fenêtres glissantes
- Problème commun : coût prohibitif sur 240 timesteps × 18 fonctions

**2.3.4 Limites communes**
- Observation externe du comportement du modèle, pas du raisonnement interne
- Critique de Rudin : même problème que les gradients — approximation post-hoc

### 2.4 Méthodes intrinsèques

**2.4.1 Class Activation Mapping — CAM (Zhou et al. 2016)**
- Score d'activation par région spatiale pour les CNN de classification
- Utilise la dernière couche convolutive + les poids de la couche de classification
- Limite originale : régions spatiales sur images, pas applicable directement aux séries

**2.4.2 SoftCAM (Djoumessi & Berens 2025)**
- Généralisation de CAM : `Evidence Layer = softmax(Linear(features))`
- Régularisation ElasticNet : L1 (sparsité) + L2 (stabilité) + entropie H (prévention collapse)
- Fidélité par construction : l'explication est dans le modèle, pas à côté
- Inspiration directe de la contribution de ce mémoire

**2.4.3 Temporal Fusion Transformer — TFT (Lim et al. 2019)**
- Transformer conçu pour séries temporelles multivariées
- Poids d'attention par variable et par timestep produits nativement
- **Concurrent direct** : encadreurs peuvent demander "pourquoi pas TFT ?"
- Réponse : attention ≠ explication (voir §2.5) — TFT n'a pas de mécanisme de régularisation qui force ses poids à être fidèles

**2.4.4 Avantages des méthodes intrinsèques**
- Fidélité exacte : l'explication est calculée par le même forward pass que la prédiction
- Pas de modèle approximatif externe
- Coût constant : une seule passe du modèle

### 2.5 La controverse attention ≠ explication

**2.5.1 Jain & Wallace (2019)**
- Des distributions d'attention adversariales peuvent produire les mêmes prédictions
- Conclusion : l'attention ne reflète pas nécessairement le raisonnement du modèle

**2.5.2 Wiegreffe & Pinter (2019)**
- Nuance : l'attention peut être une explication dans certaines conditions
- Mais : conditions restrictives, pas garanties en général

**2.5.3 Serrano & Smith (2019)**
- Les gradients sont plus informatifs que l'attention pour la décision

**2.5.4 Implication pour ce travail**
- TFT produit des poids d'attention → non suffisants comme explication (Jain & Wallace 2019)
- La régularisation ElasticNet de M est ce qui distingue [NOM-MODÈLE] de TFT

### 2.6 Métriques d'évaluation de l'explicabilité

**2.6.1 Faithfulness (DeYoung et al. 2020)**
- Comprehensiveness : masquer les positions clés → dégradation de la performance
- Sufficiency : garder seulement les positions clés → préservation de la performance

**2.6.2 Activation precision et activation sensitivity**
- Precision : les positions pointées par M coïncident-elles avec les vrais moments importants ?
- Sensitivity : une petite perturbation de l'entrée change-t-elle M de façon cohérente ?
- ⚠️ Adaptation aux séries temporelles nécessaire : la "région annotée" = timesteps de pic

**2.6.3 Évaluation qualitative**
- Visualisation de M sur exemples concrets
- Interprétation expert métier : les moments pointés ont-ils du sens dans le domaine FaaS ?

### 2.7 Synthèse, positionnement et plus-value

**Tableau comparatif des méthodes XAI**

| Méthode | Fidélité | Coût | Applicable séries temp. | Régularisée | Retrain requis |
|---|---|---|---|---|---|
| Vanilla Gradients | Locale | Faible | Oui | Non | Non |
| Integrated Gradients | Locale | Moyen | Oui | Non | Non |
| LIME | Approximée | Moyen | Partiel | Non | Non |
| TimeSHAP | Approximée | Élevé (2^n) | Oui | Non | Non |
| TFT (attention) | Partielle | Faible | Oui | Non | Oui |
| **[NOM-MODÈLE]** | **Par construction** | **Faible (1 passe)** | **Oui** | **Oui (ElasticNet)** | **Oui** |

- **Choix : méthode intrinsèque** — seule option satisfaisant les 3 propriétés de Rudin
- **Rejet H2 (SHAP)** : coût 2^n, approximation externe, pas de fidélité par construction
- **Rejet H3 (attention seule)** : Jain & Wallace (2019) — non suffisant sans régularisation
- **Différence vs TFT** : TFT produit de l'attention, pas de l'évidence régularisée — crucial pour le jury
- Transition vers Chapitre 3 : comment concevoir cette méthode intrinsèque ?

---

## Chapitre 3 — [NOM-MODÈLE] : conception et protocole d'entraînement

> Fichier : `Chap3/Chap3.tex`
> ⚠️ Ce chapitre attend deux décisions : (1) nom du modèle, (2) dot product vs mélange additif.
> Rédiger la justification théorique AVANT d'implémenter toute modification.

### 3.1 Motivation et objectifs de conception

**3.1.1 Insuffisance de l'attention nue**
- Rappel §2.5 : l'attention non régularisée n'est pas une explication fiable
- Besoin : un mécanisme qui force M à pointer les vrais moments clés

**3.1.2 Transposition de CAM aux séries temporelles**
- CAM original : régions spatiales sur images (pas de superposition temporelle)
- Ici : timesteps du contexte comme "régions" — chaque position de M = un instant passé
- Régression (invocations FaaS) pas classification → pas de "classe" au sens CAM
- Solution : une Evidence Map par horizon de prédiction, agrégée sur prediction_length

**3.1.3 Exigences de conception**
- Performance : ne pas dégrader R²/Spearman par rapport à FAYAM
- Fidélité : M produite dans le forward pass, pas en post-hoc
- Coût : une seule passe du modèle
- Interprétabilité : M visualisable et sémantiquement signifiante

**3.1.4 Pattern de justification adopté** *(prescrit par les encadreurs — 2026-05-25)*

Pour chaque choix architectural, le mémoire suit impérativement l'ordre :
1. **Théorie** : quel papier justifie ce choix ? Quelle propriété formelle garantit-il ?
2. **Application** : comment ce principe est-il transposé dans notre architecture ?
3. **Résultats** : qu'observe-t-on empiriquement ?
4. **Interprétation** : que conclure au regard de la théorie ?

Ce pattern s'applique à : softmax pour M, combinaison M+enc_hidden, LayerNorm, L1/L2/entropie, γ annealing, dissociation mix. Aucun choix ne doit apparaître comme du tâtonnement.

### 3.2 Architecture de la couche d'évidence

**3.2.1 Vue d'ensemble**
- Position dans le Transformer : après le décodeur, avant la projection de sortie
- 5 blocs : Entrées → Encodeur → Décodeur → Evidence Layer → Prédiction
- [Figure : diagramme architecture TikZ]

**3.2.2 Calcul de M**
- `M = softmax(Linear(dec_output))` de dimension [batch × prediction_length × context_length]
- Justification du softmax : mécanisme d'attention normalisé (Vaswani et al. 2017) — la normalisation garantit que les poids somment à 1, interprétable comme distribution de probabilité sur les timesteps
- Justification de la couche linéaire : projection apprise permettant à M de dépasser l'attention brute

**3.2.3 Combinaison avec l'encodeur**

> ⚠️ Section à compléter après décision encadreurs sur dot product.

*Option A — Mélange additif (configuration actuelle)*
- `h_ev = (1 - mix) · dec_output + mix · LN(bmm(M, enc_hidden))`
- Justification : connexion résiduelle (He et al. 2016, ResNet) — préserve le signal original tout en ajoutant un terme de correction
- `bmm(M, enc_hidden)` : agrégation pondérée de l'encodeur par M, analogue à la cross-attention (Vaswani 2017)
- `mix` : hyperparamètre opératoire (analogue à la température d'un LLM)

*Option B — Dot product (à explorer)*
- `h_ev = dec_output ⊙ f(M)` ou variante à définir
- Justification attendue : à documenter depuis la littérature avant implémentation
- Avantage revendiqué par les encadreurs : mesure directe du poids de M dans la prédiction

**3.2.4 LayerNorm**
- Normalisation de `bmm(M, enc_hidden)` avant combinaison
- Justification : stabilité numérique et invariance à l'échelle (Ba et al. 2016)
- Observé empiriquement : sans LN, M s'effondre sur une position arbitraire (Run B3)

### 3.3 Régularisation ElasticNet de M

- Loss complète : `L = L_forecast(NLL Student-t) + α‖M‖₁ + β‖M‖₂² + γH(M)`

**3.3.1 Terme L1 : sparsité**
- `α‖M‖₁` pousse les petites valeurs de M vers 0
- Justification : Zou & Hastie (2005) — L1 crée des zéros exacts, rendant M interprétable
- Effet : M retient seulement les timesteps réellement importants

**3.3.2 Terme L2 : stabilité**
- `β‖M‖₂²` évite que M diverge ou oscille
- Justification : Zou & Hastie (2005) — L2 stabilise l'optimisation en présence de features corrélées
- Effet : entraînement plus stable, pas d'explosion des poids de M

**3.3.3 Terme entropie : prévention du collapse**
- `γH(M)` pénalise l'entropie faible (M trop concentrée sur une seule position)
- Justification : analogue à la régularisation entropique en RL (Mnih et al. 2016 — A3C) et aux températures en attention (Vaswani 2017)
- Effet : empêche le spike collapse observé dans les runs sans LN

**3.3.4 γ annealing**
- γ = 0 en début d'entraînement → monte progressivement vers γ_cible
- Justification : curriculum learning (Bengio et al. 2009) — progresser du plus facile au plus contraignant
- Si γ fort dès le début : M forcée à l'uniformité avant d'avoir appris → perte d'information

### 3.4 Protocole d'entraînement

**3.4.1 Dissociation entraînement / inférence**
- mix = 0.0 à l'entraînement → le modèle apprend sur dec_output seul
- mix = 0.25 à l'inférence → M contribue 25% du signal
- Justification : éviter le chicken-and-egg (M non structurée perturbe le gradient → M ne peut pas se structurer)
- Optimum empirique : mix_inférence ≈ 5× mix_entraînement sur les 3 checkpoints

**3.4.2 Runs et leçons apprises**

| Run | Configuration | R² | Leçon |
|-----|--------------|-----|-------|
| A | FAYAM reproduction (mix=0) | 0.5299 | Baseline reproductible |
| B1 | mix=0.30 dès époque 0 | −2.83 | Chicken-and-egg : collapse immédiat |
| B2 | γ annealing + mix schedule | en cours | Stabilisation progressive |
| B3 | + LayerNorm | convergence | LN résout le spike collapse |
| B5 | Config finale | 0.7563 | Optimum identifié |

**3.4.3 Bug Fix #5**
- Problème : `generate()` v3 retournait `dec_output` seul, court-circuitant `h_ev`
- `forward()` (entraînement) était correct — checkpoints valides
- Impact : toutes les métriques avant Fix #5 mesuraient dec_output, pas h_ev
- Correction : patcher `generate()` pour utiliser le bon branch
- Leçon méthodologique : détecter et documenter le bug renforce le protocole

### 3.5 Hypothèses de validation H1.A→H1.G

| Hypothèse | Énoncé | Critère de validation |
|-----------|--------|----------------------|
| H1.A | M est temporellement cohérente | argmax(M) ≈ heures de pic observées |
| H1.B | M corrèle avec cross-attentions | Pearson(M, cross-att) > 0.9 |
| H1.C | Performance préservée | R² ≥ FAYAM (0.37), Spearman ≥ 0.90 |
| H1.D | M cohérente inter-fonctions du cluster | Corrélation M inter-fonctions élevée |
| H1.E | M plus concentrée quand meilleure prédiction | Spearman(R², H(M)) < 0 |
| H1.F | Masquer top-k de M dégrade MAE | ΔMAE > 0 pour k croissant |
| H1.G | Garder top-k de M préserve MAE | Préservation > 95% pour k faible |

---

## Chapitre 4 — Résultats et évaluation

> Fichier : `Chap4/Chap4.tex`

### 4.1 Configuration expérimentale

**4.1.1 Données**
- Cluster 4, Azure Functions 2019 : 19 fonctions, 14 jours, résolution 1 min
- Split : train/val/test (ratio à préciser depuis le notebook)
- Prétraitement : stationnarisation (différenciation si nécessaire), normalisation

**4.1.2 Hyperparamètres**
- Héritage FAYAM : enc_layers=4, dec_layers=4, d_model=32, context=240, pred=120
- HPO Optuna : 15 trials, TPE + MedianPruner, early stopping val R² patience=10

**4.1.3 Environnement**
- Google Colab (GPU T4/A100), PyTorch, HuggingFace Transformers
- Checkpoints sauvegardés sur Google Drive

### 4.2 Performance prédictive — H1.C

**4.2.1 Waterfall R²**
- FAYAM original : R²=0.3701 (global toutes clusters)
- Run A (reproduction C4 + HPO) : R²=0.5299 (+15.98 pp)
- B5 + mix=0 inférence : R²=0.6646 (+13.46 pp — gain Evidence Layer seule)
- B5 + mix=0.25 inférence : R²=0.7563 (+9.17 pp — gain utilisation effective de M)
- [Figure : waterfall chart TikZ ou figure notebook]

**4.2.2 Analyse de la dissociation**
- Courbe R² vs mix à l'inférence (0, 0.05, 0.10, 0.25, 0.50, 1.0)
- Optimum à mix=0.25 (R²=0.7563)
- Dégradation pour mix > 0.25 : dec_output perturbé par M trop dominant

**4.2.3 Comparaison TFT**
- TFT non mesuré sur ce dataset — comparaison théorique uniquement
- [Figure : bar chart FAYAM vs B5+mix=0.25 avec ligne TFT "non mesuré"]

**4.2.4 Seuil R² dans la littérature** *(question soulevée par les encadreurs — à documenter)*
- Existe-t-il un consensus sur un R² "acceptable" pour la prédiction de séries temporelles ?
- Pistes : littérature FaaS (FAYAM lui-même = 0.37), littérature prédiction de charge réseau, benchmarks HuggingFace Time Series
- Argument à construire : notre R²=0.7563 se situe à quel niveau parmi les travaux comparables ?
- ⚠️ À documenter avant de rédiger cette section — chercher dans les papiers de l'état de l'art

### 4.3 Qualité de la carte d'évidence M

**4.3.1 H1.A — Cohérence temporelle**
- Résultat : M pointe systématiquement la plage 17h–19h sur Cluster 4
- Interprétation : fin de journée ouvrée → montée des invocations → signal réel
- Limite : test diagonal formel `argmax(M[t]) ≈ heure(t)` non encore calculé → tendance observée
- [Figure : visualisation de M sur exemple concret — à exporter depuis notebook]

**4.3.2 H1.B — Corrélation M vs cross-attentions**
- Pearson = 0.992 (corrélation très forte)
- Interprétation : M et cross-attentions capturent le même signal temporel
- Mais M est régularisée → plus sparse, plus interprétable que l'attention brute
- [Figure : heatmap ou scatter plot M vs cross-att]

**4.3.3 H1.D — Cohérence inter-fonctions**
- M similaire entre fonctions du même cluster
- Justification a priori : FAYAM montre que les fonctions d'un cluster ont des profils similaires
- [Figure : corrélation inter-fonctions de M]

**4.3.4 H1.E — Concentration vs qualité**
- Spearman ρ = −0.80 entre R² et H(M) sur 5 configurations mix
- Interprétation : quand le modèle prédit mieux, M est plus concentrée
- ⚠️ Limite statistique : n=5, ρ=-0.80 non significatif (p > 0.05) — tendance indicative
- [Figure : scatter plot R² vs H(M) avec droite de régression]

**4.3.5 H1.F — Comprehensiveness**
- Protocole : masquer top-k positions de M, mesurer ΔMAE
- Résultats : max ΔMAE = +5.37% pour k→max (toutes positions masquées)
- Plafond théorique : 25% (mix=0.25 → M contribue au plus 25% de h)
- Ratio : 5.37/25 = 21% du plafond utilisé
- Interprétation : M contribue modestement mais réellement à la prédiction
- [Figure : courbe ΔMAE vs k avec ligne plafond 25%]

**4.3.6 H1.G — Sufficiency**
- Protocole : garder seulement top-k positions de M, mesurer % préservation MAE
- Résultats : k=1 → 97.1% de préservation ; k=5 → 97.1% ; k=10 → 97.9% ; k=50 → 100.2%
- Interprétation : une seule position de M suffit à capturer l'essentiel de sa contribution
- [Figure : bar chart k vs % préservation avec ligne 100%]

**4.3.7 Mesure de confiance en M** *(question soulevée par les encadreurs — à développer)*
- Problème : comment savoir si M est fiable et non un artifact d'entraînement ?
- Piste 1 — Stabilité inter-runs : calculer la variance de M à input fixe sur plusieurs initialisations aléatoires. Si M est stable, elle capture un signal réel.
- Piste 2 — Bootstrap : rééchantillonner les données de test, recalculer M à chaque tirage, construire un intervalle de confiance sur argmax(M).
- Piste 3 — Calibration : comparer la distribution de M à une distribution de référence (uniforme = modèle qui ne sait pas, one-hot = modèle trop confiant).
- ⚠️ À implémenter dans le notebook avant de rédiger cette section

### 4.4 Évaluation qualitative *(à compléter — figures réelles notebooks obligatoires)*

> ⚠️ Les encadreurs ont explicitement demandé des figures issues de l'exécution des notebooks, pas uniquement des schémas TikZ.

- **Exemple 1 — bonne prédiction** : visualiser M en heatmap sur la série temporelle réelle + la prédiction. Les positions pointées par M correspondent-elles aux heures de pic observées ?
- **Exemple 2 — prédiction moyenne** : M est-elle plus dispersée ? Moins concentrée ?
- **Exemple 3 — mauvaise prédiction** : M est-elle uniforme / incohérente ?
- Vérification domaine : 17h–19h est-ce une heure de pic pour ce type de fonction cloud ?
- [Figures à exporter depuis notebook `softcam-cluster4-v3-h1-analysis.ipynb`]

### 4.5 Discussion

**4.5.1 Synthèse des 7 hypothèses**

| Hypothèse | Résultat | Statut |
|-----------|---------|--------|
| H1.A | M pointe 17h-19h | ✅ (tendance — test diagonal à formaliser) |
| H1.B | Pearson=0.992 | ✅✅ |
| H1.C | R²=0.7563 vs 0.37 FAYAM | ✅✅ |
| H1.D | Cohérence inter-fonctions | ✅✅ |
| H1.E | ρ=−0.80 (n=5, non significatif) | ⚠️ tendance indicative |
| H1.F | ΔMAE=+5.37% / plafond 25% | ⚠️ contribution modeste mais réelle |
| H1.G | 97.1% préservation avec k=1 | ✅ |

**4.5.2 Limites reconnues**
1. Dissociation entraînement/inférence : mix optimal d'inférence ≠ mix d'entraînement — design assumé mais à justifier davantage
2. Un seul cluster testé (Cluster 4) — généralisation à C0/C6/C8 non validée
3. H1.E statistiquement fragile (n=5)
4. H1.F : contribution modeste (21% du plafond) — M raffine, ne domine pas
5. Évaluation qualitative incomplète (visualisations M)
6. TFT non comparé empiriquement

**4.5.3 Comparaison théorique H1 vs H2 (SHAP)**
- TimeSHAP : coût 2^n vs une passe → avantage [NOM-MODÈLE]
- TimeSHAP : approximation externe vs fidélité par construction → avantage [NOM-MODÈLE]
- TimeSHAP : pas de contrainte sur l'architecture → avantage déploiement (pas de retrain)

---

## Conclusion générale

> Fichier : `Conclusion/Conclusion.tex` — commande : `\myChapterStar`

**1. Rappel de la problématique**
- FaaS prédiction opaque → besoin d'explicabilité pour la confiance, le diagnostic, la conformité

**2. Réponse aux hypothèses**
- H1.A→H1.G : bilan honnête (ce qui est validé, ce qui est partiel)

**3. Contributions principales et plus-value**
- Architecture [NOM-MODÈLE] avec Evidence Layer M régularisée
- Protocole de validation en 7 hypothèses adapté aux séries temporelles
- R²=0.7563 (+38.6 pp vs FAYAM) avec explicabilité intrinsèque
- **Plus-value synthétique** : seule approche améliorant simultanément performance et explicabilité, à coût constant, sans approximation post-hoc

**4. Limites**
- Cluster unique, dissociation mix, évaluation qualitative incomplète

**5. Perspectives**
- Révision architecturale (dot product — decision en cours)
- Extension à Cluster 0, 6, 8
- Comparaison empirique avec TimeSHAP
- Application à d'autres datasets FaaS

---

## Annexes

**Annexe A — Hyperparamètres**
- Hyperparamètres FAYAM (retrouvés dans `tsf_transf.py`)
- HPO Optuna : meilleurs hyperparamètres trouvés, convergence des trials

**Annexe B — Extraits de code**
- `softcam_transformer_v3.py` : Evidence Layer (forward + loss)
- Fix #5 : correction de `generate()`

**Annexe C — Résultats complets par run**
- Tableau R²/Spearman/MAE pour tous les runs (A, B1, B2, B3, B5, B6, B7)

---

## Checklist retours encadreurs (2026-05-25)

> À cocher avant soumission du mémoire.

- [ ] Chaque choix architectural a une justification théorique littérature (pattern §3.1.4)
- [ ] Seuil R² dans la littérature documenté (§4.2.4)
- [ ] Mesure de confiance en M implémentée et rédigée (§4.3.7)
- [ ] Figures réelles notebooks exportées et intégrées (§4.4)
- [ ] Évaluation explicabilité quantitative complète (activation precision/sensitivity)
- [ ] Évaluation explicabilité qualitative rédigée (§4.4)
- [ ] Nom du modèle tranché et appliqué partout
- [ ] Dot product exploré, justifié ou écarté avec justification

---

## Décisions en attente avant rédaction de Chap3

| Décision | Options | Bloque quoi |
|----------|---------|------------|
| Nom du modèle | TET / ELT / autre | Titre, Chap3, Chap4, Résumé |
| Dot product vs mélange | Justification théorique requise | §3.2.3 |
| Évaluation qualitative | Figures à exporter | §4.4 |
| Test diagonal H1.A | Calcul `argmax(M[t])` | §4.3.1, §3.5 |

---

## Références clés à intégrer dans la bibliographie

| Clé | Référence | Utilisée dans |
|-----|-----------|--------------|
| Rudin2019 | Rudin (2019), Stop Explaining Black Box ML | Intro, Chap2 §2.1, 2.7 |
| Lipton2018 | Lipton (2018), Mythos of Model Interpretability | Intro, Chap2 §2.1 |
| Vaswani2017 | Vaswani et al. (2017), Attention is All You Need | Chap3 §3.2 |
| He2016 | He et al. (2016), ResNet | Chap3 §3.2.3 |
| Ba2016 | Ba et al. (2016), Layer Normalization | Chap3 §3.2.4 |
| ZouHastie2005 | Zou & Hastie (2005), ElasticNet | Chap3 §3.3 |
| Djoumessi2025 | Djoumessi & Berens (2025), SoftCAM | Chap2 §2.4.2, Chap3 §3.1 |
| Lim2019 | Lim et al. (2019), TFT | Chap2 §2.4.3 |
| JainWallace2019 | Jain & Wallace (2019), Attention ≠ Explanation | Chap2 §2.5 |
| DeYoung2020 | DeYoung et al. (2020), ERASER benchmark | Chap2 §2.6, Chap3 §3.5 |
| Ribeiro2016 | Ribeiro et al. (2016), LIME | Chap2 §2.3.1 |
| Lundberg2017 | Lundberg & Lee (2017), SHAP | Chap2 §2.3.2 |
| Bento2021 | Bento et al. (2021), TimeSHAP | Chap2 §2.3.3 |
| Zhou2016 | Zhou et al. (2016), CAM | Chap2 §2.4.1 |
| Selvaraju2017 | Selvaraju et al. (2017), GradCAM | Chap2 §2.2.4 |
| Bengio2009 | Bengio et al. (2009), Curriculum Learning | Chap3 §3.3.4 |
