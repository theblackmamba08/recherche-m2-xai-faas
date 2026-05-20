---
type: fiche-lecture
article: 2020_Jacovi_FaithfullyInterpretable.pdf
auteurs: Alon Jacovi, Yoav Goldberg
annee: 2020
journal_conf: ACL 2020, Online — Bar-Ilan University
date_lecture: 2026-05-20
pertinence: 4
tags: [faithfulness, plausibility, interpretability, nlp, evaluation-xai, h1-validation]
---

# Towards Faithfully Interpretable NLP Systems: On the Faithfulness of Saliency Methods

> Citation BibTeX-key : `Jacovi2020Faithful` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Référence clé pour la définition de fidélité.** Jacovi & Goldberg distinguent *fidélité* (l'explication reflète le calcul réel du modèle) et *plausibilité* (l'explication semble raisonnable à un humain). Une explication peut être plausible sans être fidèle — et vice versa. Utile pour argumenter la fidélité par construction de M.

## Problème traité

Les méthodes d'explication en NLP (attention, saliency maps, gradient) sont évaluées soit par des humains (plausibilité) soit par des métriques automatiques (corrélation avec gradients). Mais ces deux dimensions ne mesurent pas la même chose. L'article propose un cadre formel pour distinguer *fidélité* et *plausibilité*.

## Méthode / Argumentation

### Définitions formelles

**Fidélité** (*faithfulness*) : une explication E est fidèle à un modèle F si E reflète fidèlement le *processus de calcul réel* de F. Autrement dit, E indique vraiment les features qui causent la prédiction de F.

**Plausibilité** (*plausibility*) : une explication E est plausible si elle semble raisonnable à un observateur humain — si elle correspond à ce que l'humain penserait être important.

### La disjonction clé

Une explication peut être :
- **Fidèle et plausible** (idéal) — rare.
- **Fidèle mais non plausible** — le modèle utilise des features "bizarres" mais correctement identifiées.
- **Plausible mais non fidèle** — l'explication a l'air sensée mais ne reflète pas le calcul réel.
- **Ni l'un ni l'autre** — inutile.

### Application aux méthodes NLP

- **Attention** → plausible (semble raisonnable) mais pas nécessairement fidèle (Jain & Wallace 2019).
- **Gradient-based** → plus fidèles (corrélés au calcul) mais moins plausibles visuellement.
- **Rationale extraction** → fidèle ET plausible si conçue correctement.

### Critères de fidélité proposés

1. *Sufficiency* : la rationale seule suffit à reproduire la prédiction.
2. *Comprehensiveness* : retirer la rationale dégrade la prédiction.
3. *Monotonicity* : ajouter plus de features importantes améliore la prédiction.

## Résultats clés

- Les méthodes d'attention sont systématiquement plus plausibles que fidèles.
- La fidélité et la plausibilité sont faiblement corrélées.
- Les évaluations humaines de plausibilité ne sont pas de bons proxy de fidélité.

## Lien avec H1

**Argument central de H1** : notre Evidence Layer est *fidèle par construction*. M est littéralement le coefficient algébrique dans `h = (1-mix)·dec_output + mix·LayerNorm(bmm(M, enc_hidden))`. Pas d'approximation, pas de post-hoc.

La distinction Jacovi & Goldberg permet de formuler précisément notre claim :
- On ne revendique pas que M est *plausible* (bien que H1.A et H1.D suggèrent que oui).
- On revendique que M est *fidèle* — par construction mathématique.

À citer en slide 26 (méthodologie de validation) : *"Suivant Jacovi & Goldberg 2020, nous distinguons fidélité et plausibilité. Notre test de fidélité repose sur comprehensiveness (H1.F) et sufficiency (H1.G)."*

## Citations à réutiliser

> "We define a saliency map to be faithful if it accurately reflects the model's reasoning process." (Section 2)

> "Plausibility is not faithfulness: a saliency map can be plausible without being faithful, and faithful without being plausible." (Abstract)

> "Faithfulness requires that the explanation reflects what the model actually computes, not what a human observer thinks the model should compute." (Section 2.1)

> "Comprehensiveness and sufficiency are the two key properties of faithful explanations." (Section 3)

## Idées à creuser

- Combiner avec DeYoung 2020 (ERASER) pour une double référence sur comprehensiveness/sufficiency — les deux sont complémentaires : ERASER définit les métriques, Jacovi & Goldberg définissent la philosophie.
- Slide 26 peut référencer les deux : ERASER pour le protocole expérimental, Jacovi & Goldberg pour le cadre conceptuel.
