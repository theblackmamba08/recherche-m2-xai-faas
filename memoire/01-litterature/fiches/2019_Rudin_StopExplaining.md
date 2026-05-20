---
type: fiche-lecture
article: 2019_Rudin_StopExplaining.pdf
auteurs: Cynthia Rudin
annee: 2019
journal_conf: Nature Machine Intelligence, 1(5), pp. 206–215 — Duke University
date_lecture: 2026-05-20
pertinence: 5
tags: [intrinsic-interpretability, post-hoc, black-box, self-explainable, high-stakes, h1-motivation]
---

# Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead

> Citation BibTeX-key : `Rudin2019Stop` *(à reporter dans `redaction/biblio/refs.bib`)*

> 💡 **Argument philosophique fondateur pro-intrinsèque.** Rudin argumente que pour les décisions à fort enjeu, les modèles intrinsèquement interprétables sont préférables aux explications post-hoc des boîtes noires. Justifie philosophiquement notre choix de H1 vs H2 (post-hoc).

## Problème traité

La pratique dominante en ML est de déployer des modèles noirs très performants, puis d'ajouter des explications post-hoc (LIME, SHAP, GRAD-CAM). Rudin argue que cette approche est fondamentalement défaillante pour les décisions à fort enjeu (médecine, justice, finance) : les explications post-hoc ne sont que des approximations d'un modèle qu'on ne comprend pas.

## Méthode / Argumentation

**Thèse principale** : il existe systématiquement des modèles intrinsèquement interprétables (arbres de décision, règles, GAMs, modèles linéaires) qui ont des performances comparables aux boîtes noires sur les données tabulaires structurées. Dans ce cas, utiliser une boîte noire + post-hoc est injustifié.

**Mythes réfutés** :
1. *"Les boîtes noires sont nécessaires pour la haute précision"* → faux sur les données tabulaires.
2. *"Les explications post-hoc sont fiables"* → faux : elles approximent le modèle, pas les données.
3. *"On peut toujours faire confiance à LIME/SHAP"* → faux : sensibles aux perturbations, non uniques.

**Propriétés d'une vraie explication** selon Rudin :
- Simulable par un humain.
- Basée sur des features sémantiquement signifiantes.
- Fidèle au modèle par construction (pas par approximation).

## Résultats clés

- Revue d'études montrant que des modèles interprétables (scoring lists, GAMs) atteignent des performances comparables aux boîtes noires sur des datasets médicaux et juridiques.
- Cas d'étude : récidive criminelle (COMPAS) — un modèle logistique simple atteint les mêmes performances que l'algorithme opaque avec des règles lisibles.

## Limites identifiées par les auteurs

- L'argument est plus solide pour les données *tabulaires* ; pour les séquences, images, textes, les modèles interprétables restent moins performants.
- Ne dit pas qu'on ne *peut* pas utiliser des boîtes noires — dit qu'on ne *devrait* pas pour des décisions à fort enjeu sans nécessité démontrée.

## Lien avec H1

**Argument philosophique pour notre choix** : H1 propose un modèle *architectural self-explainable* (intrinsèque) plutôt que d'appliquer SHAP/IG (post-hoc) sur le modèle FAYAM. Rudin 2019 justifie ce choix : si on peut avoir l'explication intégrée au modèle, on devrait le faire.

À citer en :
- Slide 3 (pourquoi expliquer ?) : argument pour l'explication intrinsèque.
- Slide 6 (promesse de l'intrinsèque) : référence centrale.
- Slide 33 (position self-explainability) : légitimité de l'approche.

## Citations à réutiliser

> "Explainability methods for black box models have a critical problem: they can provide explanations that look reasonable but are not faithful to what the model actually computes." (p. 207)

> "If a simple model can achieve the same accuracy as a complex model, there is no need to use the complex model." (p. 208)

> "An interpretable model is one that can be directly understood by a human, without the need for any additional explanation." (p. 206)

> "We should stop trying to explain black box models and instead design models that are interpretable from the start." (Conclusion)

## Idées à creuser

- Appliquer l'argument Rudin au contexte FaaS/serverless : les décisions d'auto-scaling sont-elles à fort enjeu ? Oui si elles impactent la QoS et les coûts.
- Contrepoint utile : SoftCAM montre qu'on peut avoir un modèle *expressif* (Transformer) ET intrinsèquement interprétable — réconciliant la puissance prédictive et l'explicabilité que Rudin oppose.
