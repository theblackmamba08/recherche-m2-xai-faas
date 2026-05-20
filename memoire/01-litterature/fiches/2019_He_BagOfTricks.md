---
type: fiche-lecture
article: 2019_He_BagOfTricks.pdf
auteurs: Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
annee: 2019
journal_conf: CVPR 2019, Long Beach — Amazon Web Services
date_lecture: 2026-05-20
pertinence: 2
tags: [training, warm-up, learning-rate, schedule, batch-normalization, h1-training]
---

# Bag of Tricks for Image Classification with Convolutional Neural Networks

> Citation BibTeX-key : `He2019BagTricks` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🛠️ **Référence technique pour la justification du warm-up.** Documente empiriquement que le warm-up du learning rate (et par extension du mix dans notre cas) est un ingrédient critique pour stabiliser l'entraînement quand on introduit des composants hétérogènes. Citer pour légitimer le warm-up mix de Run B5.

## Problème traité

L'entraînement de CNN performants requiert de nombreux "trucs" pratiques (learning rate schedules, data augmentation, initialisation, etc.) qui sont souvent omis dans les articles et difficiles à reproduire. Ce papier compile et évalue systématiquement ces techniques.

## Méthode

Évaluation sur ResNet-50 / ImageNet de plusieurs techniques :
- **Large batch training** avec linear scaling rule pour le LR.
- **Warm-up du LR** : démarrer avec un LR faible et l'augmenter progressivement sur les premières époques.
- **Label smoothing**, **mixup**, **cosine LR decay**.
- **Knowledge distillation**.

### Warm-up (Section 2.2)

Avec un grand batch size, une mise à jour de gradient agressive dès la première époque peut faire diverger l'entraînement (les poids sont loin de leur optimum). Le warm-up consiste à :
1. Démarrer avec un LR faible (ou mix=0 dans notre cas).
2. Augmenter linéairement jusqu'à la valeur cible sur les `N` premières époques.

Résultat : stabilité de l'entraînement, convergence plus fiable.

## Résultats clés

- Warm-up + cosine decay + label smoothing + mixup → **+1.7 pp Top-1 accuracy** sur ResNet-50 ImageNet (76.4% → 78.1%).
- Le warm-up seul contribue significativement à la stabilité avec grand batch.

## Lien avec H1

**Analogie directe** avec notre warm-up du mix :

| Contexte | Warm-up classique (LR) | Notre warm-up (mix) |
|---|---|---|
| Paramètre | Learning Rate | evidence_mix |
| Problème évité | Divergence gradient avec grand LR | Collapse `dec_output` avec mix élevé |
| Stratégie | LR: 0 → LR_target sur N epochs | mix: 0 → 0.05 sur N epochs |
| Résultat | Stabilité de l'entraînement | Convergence de l'Evidence Layer |

Run B (mix=0.3 sans warm-up) → R²=−2.83. Run B5 (warm-up mix 0→0.05) → R²=0.71. **Le warm-up est l'ingrédient critique** — exactement comme documenté par He et al. pour le LR.

À citer en slides 10 et 17 pour légitimer la stratégie d'entraînement progressif.

## Citations à réutiliser

> "Training with a large learning rate at the beginning can lead to numerical instability. A common solution is to use a warm-up strategy." (Section 2.2)

> "We gradually increase the learning rate from 0 to the initial learning rate during the first few training epochs." (Section 2.2)

## Idées à creuser

- L'analogie LR warm-up / mix warm-up est pédagogiquement forte pour expliquer le Run B5 aux encadreurs : c'est une pratique établie de l'entraînement profond, appliquée à un nouveau paramètre.
- Smith 2017 (Cyclical Learning Rates) est une extension du même principe — à citer conjointement si on veut approfondir.
