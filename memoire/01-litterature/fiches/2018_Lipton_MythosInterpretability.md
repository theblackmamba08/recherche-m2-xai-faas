---
type: fiche-lecture
article: 2018_Lipton_MythosInterpretability.pdf
auteurs: Zachary C. Lipton
annee: 2018
journal_conf: ACM Queue, 16(3) / Communications of the ACM, 61(10) — Carnegie Mellon University
date_lecture: 2026-05-20
pertinence: 4
tags: [interpretability, transparency, post-hoc, intrinsic, framework, h1-motivation]
---

# The Mythos of Model Interpretability

> Citation BibTeX-key : `Lipton2018Mythos` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🗺️ **Cadre conceptuel pour la taxonomie XAI.** Lipton distingue transparence (intrinsèque) et post-hoc explainability, et énumère les propriétés désirables d'une explication. Utile pour positionner H1 dans le paysage XAI.

## Problème traité

Le terme "interprétabilité" est utilisé de manière vague et incohérente dans la littérature ML. Chaque article définit l'interprétabilité différemment. Lipton propose un cadre taxinomique pour clarifier ce qu'on veut dire quand on dit qu'un modèle est "interprétable".

## Méthode / Argumentation

### Taxonomie principale

**Transparence** (propriété du modèle lui-même) :
- *Simulabilité* : un humain peut simuler mentalement le modèle.
- *Décomposabilité* : chaque partie du modèle est intelligible séparément.
- *Transparence algorithmique* : l'algorithme d'apprentissage est compréhensible.

**Post-hoc explainability** (explication après-coup) :
- *Explication textuelle naturelle*.
- *Visualisations* (attention maps, saliency maps).
- *Explication par exemples* (prototypes, counterfactuals).
- *Explication locale* (LIME, SHAP).

### Propriétés désirables selon les cas d'usage

- Décisions à fort enjeu → transparence préférée.
- Débogage → post-hoc acceptable.
- Confiance utilisateur → dépend du contexte.

### Mise en garde principale

> Les explications post-hoc peuvent être *trompeuses* : elles expliquent le modèle, pas nécessairement le phénomène sous-jacent.

## Résultats clés

- Pas de résultats empiriques — papier de position.
- Influence majeure sur la communauté XAI, souvent cité pour la taxonomie.

## Lien avec H1

Permet de positionner H1 clairement dans la taxonomie :
- Notre Evidence Layer → **transparence décomposable** : M est une partie du modèle directement interprétable.
- TimeSHAP / IG → **post-hoc explainability**.
- La "fidélité par construction" de H1 correspond à la *simulabilité* de Lipton : on peut calculer exactement comment M contribue à la prédiction.

À citer en slides 3 (motivation), 6 (promesse intrinsèque) et 33 (position self-explainability).

## Citations à réutiliser

> "Calls for interpretability may be motivated by a desire to audit models for various reasons, such as ensuring that the model is not using spurious features." (Section 1)

> "Post-hoc explanations can be misleading — they explain the model, not the world." (Section 4)

> "Transparency and post-hoc explainability represent distinct notions of interpretability, often conflated in the literature." (Section 2)

## Idées à creuser

- La distinction transparence / post-hoc est exactement ce qui sépare H1 de H2 dans notre stratégie de recherche. À utiliser pour introduire le plan du chapitre contribution.
- Lipton 2018 + Rudin 2019 forment un tandem : Lipton clarifie le vocabulaire, Rudin prescrit le comportement.
