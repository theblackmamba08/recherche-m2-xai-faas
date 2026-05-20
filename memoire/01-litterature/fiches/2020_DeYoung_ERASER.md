---
type: fiche-lecture
article: 2020_DeYoung_ERASER.pdf
auteurs: Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace
annee: 2020
journal_conf: ACL 2020 — Northeastern University / Salesforce Research
date_lecture: 2026-05-20
pertinence: 5
tags: [faithfulness, comprehensiveness, sufficiency, benchmark, evaluation-xai, h1-validation]
---

# ERASER: A Benchmark to Evaluate Rationalized NLP Models

> Citation BibTeX-key : `DeYoung2020ERASER` *(à reporter dans `redaction/biblio/refs.bib`)*

> 📏 **Référence fondatrice pour les métriques H1.F et H1.G.** ERASER définit formellement comprehensiveness et sufficiency — les deux métriques que nous utilisons pour valider que M est une explication fidèle. À citer obligatoirement dans la section validation.

## Problème traité

Les explications en NLP (rationales, highlights) sont évaluées de manière incohérente dans la littérature. ERASER propose un benchmark unifié avec des définitions formelles et des datasets annotés, permettant de comparer les méthodes sur un pied d'égalité.

## Méthode

### Métriques clés (Section 2)

**Comprehensiveness** : une explication est comprehensive si la prédiction se **dégrade significativement** quand on supprime les tokens sélectionnés comme rationale.
```
Comprehensiveness = f(x) - f(x \ r)
```
où `r` est la rationale (tokens explicatifs) et `f(x \ r)` est la prédiction sans les tokens r.

**Sufficiency** : une explication est suffisante si la prédiction reste **stable** quand on ne garde que les tokens de la rationale.
```
Sufficiency = f(x) - f(r)
```
où `f(r)` est la prédiction sur la rationale seule.

- **Comprehensiveness haute** → les tokens retenus sont *nécessaires* à la prédiction.
- **Sufficiency haute** → les tokens retenus sont *suffisants* pour la prédiction.
- Une bonne explication maximise les deux simultanément.

### Datasets

8 datasets NLP avec annotations humaines de rationales : MultiRC, Evidence Inference, FEVER, Movies, BoolQ, CoS-E, eQASM, ERASER-SNLI.

## Résultats clés

- Les méthodes de rationale extraction (BERT-based, pipeline models) obtiennent des scores variables — pas de méthode clairement dominante.
- Comprehensiveness et sufficiency sont **partiellement corrélées** mais mesurent des propriétés distinctes.
- Les annotations humaines ont elles-mêmes un accord inter-annotateurs imparfait → upper bound non trivial.

## Lien avec H1

**Lien direct avec H1.F et H1.G** :
- Notre test H1.F (masque top-k de M, mesure Δ MAE) est une adaptation temporelle de la *comprehensiveness* d'ERASER. Les "tokens" deviennent des pas de temps historiques.
- Notre test H1.G (garde top-k de M, mesure préservation MAE) est une adaptation de la *sufficiency*.

À citer dans :
- Slide 26 (méthodologie de validation) : *"Nous reprenons les définitions formelles de comprehensiveness et sufficiency de DeYoung et al. 2020, adaptées au domaine temporel."*
- Slides 30 et 31 (H1.F et H1.G) : légitimité des tests dans la littérature XAI.

## Citations à réutiliser

> "We define comprehensiveness as the degree to which a rationale contains all the information necessary for the model to make its prediction." (Section 2.1)

> "We define sufficiency as the degree to which a rationale is sufficient to make the same prediction as the full input." (Section 2.1)

> "Comprehensiveness and sufficiency together characterize the quality of an explanation: a good explanation should be both comprehensive and sufficient." (Section 2.2)

## Idées à creuser

- L'adaptation temporelle de ces métriques à des séries temporelles n'est pas triviale : un token = un pas de temps ici, mais avec la corrélation temporelle, masquer top-k redistribue la masse sur les voisins (ceiling effect expliqué H1.F). À mentionner comme nuance de l'adaptation.
- ERASER est pour le NLP (classification) ; notre adaptation (régression de séries temporelles) est une contribution méthodologique en soi.
