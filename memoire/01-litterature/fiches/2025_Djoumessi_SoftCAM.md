---
type: fiche-lecture
article: 2025_Djoumessi_SoftCAM.pdf
auteurs: Kerol Djoumessi, Philipp Berens
annee: 2025
journal_conf: Preprint arXiv:2505.17748v1 (23 mai 2025, under review) — Hertie Institute for AI in Brain Health, University of Tübingen
date_lecture: 2026-04-27
pertinence: 5
tags: [self-explainable, intrinsic-interpretability, cam, cnn, elasticnet, vit-perspective, h1-nouvelle]
---

# Soft-CAM : rendre les modèles boîte-noire intrinsèquement explicables pour les décisions à fort enjeu

> Citation BibTeX-key : `Djoumessi2025SoftCAM` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Article fondateur de la nouvelle H1**. SoftCAM démontre qu'on peut transformer un classifieur CNN black-box en modèle *self-explainable* par une **modification architecturale minimale** — sans coût computationnel additionnel ni perte de précision. Notre H1 cherche à transposer cette idée au TimeSeriesTransformer de FAYAM.

## Problème traité

Les méthodes XAI dominantes pour les CNN (GradCAM, Integrated Gradients, Guided Backprop, ScoreCAM, LayerCAM) sont **post-hoc** : elles approximent a posteriori le raisonnement d'un modèle déjà entraîné. Conséquences listées par les auteurs (p. 1) : sensibilité aux perturbations, absence de fidélité prouvée, explications qui ne reflètent pas vraiment le processus décisionnel du modèle. Ces limites posent problème dans les domaines à fort enjeu (santé, justice, finance) où le règlement européen *AI Act* (2024) exige des décisions explicables.

Les auteurs proposent **SoftCAM**, une modification architecturale légère qui rend tout CNN « boîte-noire » intrinsèquement interprétable, **en un seul forward pass**, sans pénaliser la performance.

## Méthode

### Principe

CNN traditionnel : `Backbone → GAP (Global Average Pooling) → FCL (Fully Connected) → Softmax`.

SoftCAM (Fig. 1, p. 4) :
1. **Supprimer la couche GAP**.
2. **Remplacer la FCL par une convolution 1×1** avec **C kernels** (C = nombre de classes). Cette couche est nommée *class-evidence layer*.
3. La sortie est un volume `A ∈ ℝ^{M×N×C}` interprété directement comme **carte d'évidence par classe** (`A_c` = carte spatiale pour la classe c).
4. **Prédiction finale** :  ŷ = Softmax(AvgPool(A)) — donc l'AvgPool est rétabli **après** la couche d'évidence, pas avant.
5. Aucun paramètre supplémentaire vs FCL équivalente (eq. 3-4, p. 4).

### Régularisation ElasticNet (équation 5, p. 5)

```
L(y, ŷ) = CE(y, ŷ) + λ₁ · Σ |A_c^{i,j}| + λ₂ · Σ ||A_c^{i,j}||₂
```

- `λ₁` (Lasso, ℓ₁) → favorise la **parcimonie** (sparse SoftCAM) ; utile quand on veut une explication ciblée.
- `λ₂` (Ridge, ℓ₂) → réduit les activations non pertinentes sans les forcer à zéro (dense SoftCAM) ; utile quand l'oubli de zones critiques est plus pénalisant que les faux positifs (ex. multi-foyers d'infection en pneumonie).
- ElasticNet = combinaison des deux → équilibre.

### Évaluation expérimentale

- **3 datasets médicaux** : Kaggle Diabetic Retinopathy (fundus), Retinal OCT, RSNA Chest X-Ray.
- **Backbones** : ResNet-50 et VGG-16 (Torchvision, pré-entraînés ImageNet, fine-tunés 70 epochs, mini-batch=16, NVIDIA A40, PyTorch).
- **Hyperparamètres explicites** :
  - Lasso fundus VGG : `λ₁ = 1·10⁻⁶`. Lasso fundus ResNet : `λ₁ = 5·10⁻⁵`. Lasso OCT ResNet : `λ₁ = 9·10⁻⁴`. Lasso OCT VGG : `λ₁ = 3·10⁻⁶`.
  - Ridge ResNet : `λ₂ = 7·10⁻⁵`. Ridge VGG : `λ₂ = 2·10⁻⁴`.
  - Optimiseur : SGD + Nesterov momentum 0.9, LR 1·10⁻³ → 1·10⁻⁴ (cosine annealing clipped), weight decay 5·10⁻⁴.
- **5 baselines XAI post-hoc comparées** : GradCAM, Integrated Gradients (Captum), Guided BP, ScoreCAM, LayerCAM (TorchCAM).
- **4 métriques d'explicabilité** :
  - *Top-k localization precision* (k=30) : alignement avec les annotations vérité-terrain (patches 33×33).
  - *Activation precision* (eq. 6, p. 17) : proportion de masse positive de la saliency map qui tombe dans la région annotée.
  - *Activation sensitivity* (eq. 7, p. 17) — proposé par les auteurs : pénalise les faux négatifs (régions oubliées).
  - *Activation consistency* : proportion d'activations positives sur images malades vs négatives sur images saines.
  - *Faithfulness / sensitivity* : Area Under Deletion Curve (AUDC) — on retire progressivement les patches top-k et on mesure la chute de confiance.

## Résultats clés

### Performance préservée (Table 1, p. 6, classification binaire)
- VGG-16 : AUC 0,938 → SoftCAM dense **0,942** / sparse **0,938** sur fundus.
- ResNet-50 sur OCT : AUC 0,999 → dense **1,000** / sparse **1,000** (pas de dégradation).
- RSNA CXR : VGG AUC 0,989 → dense **0,999** / sparse **0,990**.
- **Conclusion** : la modification architecturale **ne dégrade pas la précision**, et l'améliore parfois légèrement.

### Explicabilité supérieure aux méthodes post-hoc
- *Top-k precision* : sparse SoftCAM ResNet **surpasse toutes les méthodes post-hoc** sur fundus et OCT (Fig. 3, p. 8).
- *Activation consistency* (Tables 2-3, p. 20) :
  - ResNet sparse SoftCAM sur fundus : `r⁺_LG = 0,55 ± 0,2` (positif sur malade) vs `r⁻_LG = 0,76 ± 0,2` (négatif sur sain). VGG sparse : 0,28 / 0,94. La parcimonie favorise une saliency plus concentrée.
- *Faithfulness* (AUDC) : sparse SoftCAM gagne notamment sur OCT et RSNA ; sur fundus il est second derrière Guided BP (admis par les auteurs).
- *Activation precision/sensitivity* sur RSNA (boîtes englobantes) : sparse SoftCAM = meilleure précision, ridge SoftCAM = meilleure sensibilité → *trade-off contrôlable par* `λ`.

### Coût computationnel
- **Single forward pass** : pas de back-pass ni de perturbation requise → contrairement aux méthodes gradient-based (un back-pass par classe) et gradient-free (multiples forward passes perturbés).
- Multi-classe : explications **par classe en un seul passage** (figure 5 p. 10). Avantage majeur en production.

## Limites identifiées par les auteurs

- **Résolution coarse** : pour ResNet-50 et VGG-16, la couche d'évidence opère sur des feature maps de taille 16×16 (entrée 512×512) → explications grain grossier. Limitation héritée du backbone, pas de SoftCAM.
- **Adapté aux backbones à branche unique**. Inception V3 (multi-branch) reste un défi (mentionné p. 4 in fine).
- **Trade-off `λ` task-specific** : le choix de `λ₁` / `λ₂` doit être fait par validation (Appendix C, p. 18).
- **Domaine d'évaluation limité** à l'imagerie médicale 2D ; pas de validation sur d'autres modalités.
- Perspective explicite des auteurs (p. 10) : *« In the future, we could explore the integration of SoftCAM with other standard architectures like ViT [19]. »* — **C'est exactement la porte ouverte que H1 cherche à pousser**, en transposant à un Transformer de séries temporelles.

## Limites identifiées par MOI (lecteur)

1. **Hypothèse implicite : les feature maps ont une structure spatiale exploitable**. Vrai pour CNN (filtres convolutifs préservent la localité). Pour un TimeSeriesTransformer, la structure est *séquentielle* (et tokenisée par patches temporels) — pas immédiatement compatible. Adaptation conceptuelle requise.
2. **Pas de Vision Transformer testé**. Les auteurs *suggèrent* l'extension à ViT mais ne la démontrent pas. Notre H1 sera donc une **première extension empirique** (à un cousin proche, le TimeSeriesTransformer de HuggingFace).
3. **ElasticNet sur les evidence maps**, pas sur les paramètres du modèle. La parcimonie agit sur les *activations*, ce qui est moins étudié que sur les poids (Lasso classique). À vérifier que c'est différentiable proprement (ils l'utilisent en backprop, donc oui).
4. **Évaluation focalisée classification**. SoftCAM produit une carte par classe. Pour la *régression de séries temporelles* (FAYAM prédit un nombre d'invocations), la notion de « classe » disparaît — il faudra définir un équivalent (par sortie temporelle ? par cluster DBSCAN ? par seuil de charge ?).
5. **Code « anonymous.4open.science » (preprint)** ; à valider qu'il sera publiquement accessible avant la fin de notre projet (sinon il faudra ré-implémenter).
6. **Pas de comparaison à TimeSHAP / KernelSHAP** : ces approches restent post-hoc mais sont la baseline naturelle pour le tabulaire/temporel. À garder en tête comme repli (notre H2).

## Lien avec H1 *(notre nouvelle hypothèse prioritaire)*

H1 reformulée = **Adapter SoftCAM (interprétabilité par modification architecturale + régularisation ElasticNet) au TimeSeriesTransformer de FAYAM, pour produire des cartes d'évidence temporelles par fonction et par cluster DBSCAN, sans surcoût computationnel.**

### Pistes d'adaptation au Transformer
- **Couche cible à remplacer** : la projection finale du décodeur (qui mappe `d_model=32` → distribution de probabilité sur la valeur future).
- **Forme « evidence map »** : pour une prédiction multi-horizons, la carte d'évidence pourrait être un tenseur `[predictionLength × contextLength]` indiquant la contribution de chaque pas passé à chaque pas futur — analogue spatial → temporel.
- **Régularisation ElasticNet** : applicable directement sur ce tenseur (Lasso → sparsité temporelle, mise en évidence des lags critiques ; Ridge → vue lissée).
- **Compatibilité avec les attentions** : on peut comparer la carte d'évidence apprise à la matrice d'attention décodeur — H3 (étude post-hoc des attention weights) deviendrait alors un **outil de validation** de H1.

### Risques
- **Incompatibilité conceptuelle** : la convolution 1×1 préserve les positions spatiales d'un CNN ; un Transformer mélange les positions par self-attention. La modification ne s'étend pas trivialement.
- **Coût d'entraînement** : si la modification dégrade la convergence, le budget temps (< 3 mois) est en jeu → c'est précisément pourquoi H2 (TimeSHAP) et H3 (attention weights) sont prévus comme replis ordonnés.

### Si H1 réussit
Contribution scientifique forte : **première transposition documentée d'un schéma SoftCAM-like à un Transformer de séries temporelles**, avec étude différentielle par cluster DBSCAN — angle non couvert par FAYAM ni par Djoumessi & Berens.

## Citations à réutiliser

> « These methods are often sensitive, unreliable, and fail to reflect true model reasoning, limiting their trustworthiness in critical applications. » (Abstract, p. 1)

> « Post-hoc saliency methods often lack faithfulness, reliability, and consistency, resulting in explanations that may not accurately reflect the model's decision-making process. » (p. 1)

> « SoftCAM turns classical CNNs into fully convolutional networks, generating class-specific evidence maps that are directly used for predictions. » (p. 2)

> « Our approach leverages a parameterized function `h_ψ` to directly produce class activation maps that are used for prediction. » (p. 4)

> « In the future, we could explore the integration of SoftCAM with other standard architectures like ViT. » (p. 10)

> « Our work presents a major step forward in the development of powerful self-explainable models, demonstrating that interpretable-by-design architectures can preserve, and in some cases even improve upon, the classification performance of state-of-the-art models. » (p. 10)

## Idées à creuser

- **Cartographier les briques du `TimeSeriesTransformer` HuggingFace** : identifier précisément la couche de projection finale du décodeur — c'est la candidate à remplacer par une opération « evidence-producing ».
- **Définir l'équivalent d'une « class evidence map » pour la régression** : carte temps-passé × temps-futur ? Carte par cluster DBSCAN ? Carte par seuil de charge (faible/moyenne/forte) ?
- **Récupérer le code SoftCAM** (`https://anonymous.4open.science/r/SoftCAM-E1A3/`) — vérifier la licence, en extraire la logique de la `class-evidence layer` et de la perte ElasticNet.
- **Comparer activation precision/sensitivity à des métriques de fidélité temporelle** (comprehensiveness/sufficiency de FAYAM-baseline). Si les définitions sont compatibles, on garde la même grille d'évaluation entre H1, H2 et H3 → comparaison directe.
- **Estimer le coût d'entraînement** : si SoftCAM-Transformer demande de réentraîner l'ensemble (et pas juste fine-tuner la dernière couche), prévoir le budget GPU dès la phase 1 (cluster HPC).
- **Plan B avant fin S6** : si l'adaptation conceptuelle bloque, basculer sur H2 (TimeSHAP) sans attendre, pour ne pas griller le calendrier.
