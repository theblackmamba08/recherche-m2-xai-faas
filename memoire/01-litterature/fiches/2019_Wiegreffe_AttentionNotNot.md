---
type: fiche-lecture
article: 2019_Wiegreffe_AttentionNotNot.pdf
auteurs: Sarah Wiegreffe, Yuval Pinter
annee: 2019
journal_conf: EMNLP-IJCNLP 2019, Hong Kong — Georgia Tech / Ben-Gurion University
date_lecture: 2026-05-20
pertinence: 4
tags: [attention, xai, faithfulness, controversy, counterpoint, h1-motivation]
---

# Attention is not not Explanation

> Citation BibTeX-key : `Wiegreffe2019Attention` *(à reporter dans `redaction/biblio/refs.bib`)*

> ⚖️ **Contre-point équilibré de Jain & Wallace 2019.** Ne réhabilite pas entièrement l'attention comme explication, mais montre que les conditions de la critique sont restrictives. Utile pour nuancer le débat en soutenance.

## Problème traité

Jain & Wallace 2019 affirment que l'attention n'est pas une explication. Wiegreffe & Pinter questionnent la rigueur de cette conclusion : les critères utilisés pour définir une "explication" sont-ils les bons ? La méthode adversariale est-elle convaincante ?

## Méthode

- Reprennent les mêmes tâches NLP que Jain & Wallace.
- Proposent **4 tests alternatifs** pour évaluer si l'attention peut constituer une explication :
  1. Attention comme probabilité marginale sur les inputs.
  2. Calibration de l'attention : peut-on entraîner un modèle avec attention diversifiée ?
  3. Adversarial attention : tests plus stricts (imposer que les sorties soient vraiment identiques).
  4. Uniform attention : comparer à baseline uniforme.
- Critiquent la définition de "explication" utilisée par Jain & Wallace.

## Résultats clés

- **Les adversarial attentions de Jain & Wallace sont trop permissives** : leurs distributions alternatives n'ont pas besoin d'être réellement indistinguishables en sortie, seulement proches.
- Sous des tests plus stricts, l'attention est **plus contrainte** qu'annoncé.
- **Mais** : les auteurs concluent eux-mêmes que *"attention can be a valid explanation under certain conditions"* — pas une réhabilitation totale.
- La question reste ouverte : l'attention peut parfois être informative, mais la fidélité n'est pas garantie en général.

## Limites identifiées par les auteurs

- Le débat porte sur la *définition* d'une explication — pas tranché dans la littérature.
- Les résultats varient selon les tâches et les modèles.
- N'invalide pas la conclusion principale : l'attention ne garantit pas la fidélité.

## Lien avec H1

À utiliser pour nuancer le discours en soutenance : la controverse sur l'attention est réelle et non résolue. Notre position (Evidence Layer avec fidélité par construction) court-circuite ce débat — M est algébriquement exact, indépendamment des arguments de Jain & Wallace vs Wiegreffe & Pinter.

Citer en slide 5 après Jain & Wallace : *"même les contre-arguments reconnaissent que la fidélité de l'attention n'est pas garantie en général."*

## Citations à réutiliser

> "Attention can be a valid explanation of model behavior in some circumstances." (Conclusion)

> "We argue that the evidence provided by Jain and Wallace for their claims is not conclusive." (Abstract)

> "The question of whether attention is explanation may have no clean answer; it depends on the task, the model, and the definition of explanation." (Discussion)

## Idées à creuser

- Serrano & Smith 2019 donnent un troisième point de vue avec des expériences sur Transformer pur.
- Utiliser ce débat pour positionner H1 : notre contribution évite la question en rendant la fidélité structurelle.
