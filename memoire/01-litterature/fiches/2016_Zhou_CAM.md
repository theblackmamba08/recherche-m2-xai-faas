---
type: fiche-lecture
article: 2016_Zhou_CAM.pdf
auteurs: Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba
annee: 2016
journal_conf: CVPR 2016, Las Vegas — MIT / Institut de Robòtica i Informàtica Industrial
date_lecture: 2026-05-20
pertinence: 3
tags: [cam, class-activation-map, cnn, interpretability, visualization, h1-genealogie]
---

# Learning Deep Features for Discriminative Localization

> Citation BibTeX-key : `Zhou2016CAM` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🌱 **Article originel des Class Activation Maps (CAM).** Ancêtre direct de SoftCAM et donc de H1. Montre qu'une modification architecturale simple (GAP + couche linéaire) permet de produire des cartes de localisation *sans supervision spatiale*. Citer pour la généalogie.

## Problème traité

Les CNN pour la classification d'images sont des boîtes noires : on sait qu'ils classifient correctement, mais pas *où* ils regardent dans l'image. L'objectif est d'extraire des cartes de saillance classe-spécifiques directement à partir du modèle, sans annotation de localisation.

## Méthode

**Modification architecturale** : remplacer les couches Fully Connected finales d'un CNN par :
1. **Global Average Pooling (GAP)** sur les dernières feature maps `f_k(x, y)`.
2. **Couche linéaire** avec poids `w^c_k` pour la classe c.

**Class Activation Map** pour la classe c :
```
CAM_c(x, y) = Σ_k w^c_k · f_k(x, y)
```

La carte `CAM_c` est une combinaison linéaire *pondérée par les poids de classification* des feature maps — elle localise directement les régions qui activent la classe c.

**Clé** : les poids `w^c_k` sont appris par la tâche de classification (supervision faible) → la localisation est *gratuite*.

## Résultats clés

- Localisation précise sur ImageNet malgré l'entraînement en classification seule.
- Top-5 localization error de 37.1% (vs 40.4% pour la baseline sans GAP).
- Les CAMs révèlent des concepts sémantiques par couche — visualisations interprétables.

## Limites identifiées par les auteurs

- **Nécessite de modifier l'architecture** : remplacer les FC par GAP + linéaire, puis réentraîner.
- Résolution limitée à celle de la feature map (typiquement 7×7 pour une image 224×224).
- Ne fonctionne pas sur les architectures sans structure convolutive finale.

## Lien avec H1

**Point de départ de la généalogie** :
- CAM (2016) : map intrinsèque, modification architecturale, réentraînement nécessaire.
- Grad-CAM (2017) : map post-hoc, aucune modification, backward pass.
- SoftCAM (2025) : map intrinsèque, modification minimale, forward only + ElasticNet.
- **H1** : transposition du principe SoftCAM au domaine temporel.

Notre Evidence Layer `M = softmax(Linear(dec_output))` est l'analogue temporel de `w^c_k · f_k(x,y)` : une combinaison linéaire pondérée qui à la fois explique et prédit.

À citer en slide 6 (généalogie) pour montrer que H1 s'inscrit dans une lignée scientifique établie.

## Citations à réutiliser

> "We propose a technique called Class Activation Mapping (CAM), which allows the classification-trained CNN to be able to localize discriminative image regions used for predicting the class label." (Abstract)

> "Our technique produces a discriminative localization map by projecting back the weights of the output layer on the convolutional feature maps." (Abstract)

## Idées à creuser

- Slide généalogique CAM → Grad-CAM → SoftCAM → H1 : un axe temporel (2016-2017-2025-2026) avec les propriétés progressives (post-hoc → semi-intrinsèque → intrinsèque, modif. lourde → lourde → minimale → minimale).
