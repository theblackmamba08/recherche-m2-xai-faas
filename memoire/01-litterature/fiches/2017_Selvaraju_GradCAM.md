---
type: fiche-lecture
article: 2017_Selvaraju_GradCAM.pdf
auteurs: Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
annee: 2017
journal_conf: ICCV 2017, Venice — Georgia Tech / Virginia Tech
date_lecture: 2026-05-20
pertinence: 3
tags: [grad-cam, cam, cnn, saliency, visualization, h1-genealogie]
---

# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

> Citation BibTeX-key : `Selvaraju2017GradCAM` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🔗 **Maillon de la chaîne généalogique CAM → Grad-CAM → SoftCAM → H1.** Grad-CAM est l'extension de CAM (Zhou 2016) qui fonctionne sans modifier l'architecture — mais il reste post-hoc. SoftCAM (Djoumessi 2025) le supplante avec une approche intrinsèque.

## Problème traité

CAM (Zhou 2016) nécessite de modifier l'architecture CNN (remplacer les couches fully-connected par une GAP + couche linéaire) — ce qui requiert un réentraînement complet. Grad-CAM généralise CAM à *n'importe quelle* architecture CNN sans modification structurelle.

## Méthode

Pour une classe cible `c` et une couche convolutive `A^k` (k canaux) :

1. **Gradient** : calculer `∂y^c / ∂A^k_ij` — gradient de la sortie de classe par rapport aux activations.
2. **Poids** : `α^c_k = (1/Z) Σ_ij ∂y^c / ∂A^k_ij` — Global Average Pooling des gradients.
3. **Carte** : `Grad-CAM^c = ReLU(Σ_k α^c_k · A^k)` — combinaison linéaire pondérée, ReLU pour garder les influences positives.

**Post-hoc** : nécessite un backward pass par classe → coûteux en inférence.

## Résultats clés

- Localise correctement les régions discriminantes sur ImageNet, VQA, captioning.
- Visuellement plus nette que les saliency maps simples.
- Compatible avec toutes les architectures CNN (VGG, ResNet, Inception) sans modification.
- Human study : les humains préfèrent Grad-CAM aux saliency maps simples comme explication.

## Limites identifiées par les auteurs

- Résolution limitée par la taille de la feature map de la dernière couche convolutive.
- Post-hoc → pas de garantie de fidélité.
- Un backward pass par classe → coûteux pour la classification multi-classe.

## Lien avec H1

**Généalogie** : CAM (2016) → Grad-CAM (2017) → SoftCAM (2025) → H1.

Grad-CAM résout le problème d'applicabilité universelle de CAM, mais reste post-hoc. SoftCAM (Djoumessi 2025) reprend l'idée des cartes d'évidence de CAM et les rend intrinsèques — sans backward pass, avec fidélité par construction.

Notre H1 est l'adaptation de ce principe au domaine temporel (TimeSeriesTransformer).

À citer en slide 6 (généalogie CAM → Grad-CAM → SoftCAM) pour construire le fil de motivation.

## Citations à réutiliser

> "Grad-CAM uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept." (Abstract)

> "Grad-CAM is a strict generalization of CAM." (Section 3.2)

## Idées à creuser

- La progression CAM → Grad-CAM → SoftCAM illustre bien l'évolution vers l'intrinsèque : CAM est semi-intrinsèque (nécessite GAP), Grad-CAM est post-hoc (gradient), SoftCAM est pleinement intrinsèque (forward only).
- Pour la présentation, une slide généalogique avec ces 3 étapes + H1 serait très pédagogique.
