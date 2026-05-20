---
type: fiche-lecture
article: 2019_Serrano_AttentionInterpretable.pdf
auteurs: Sofia Serrano, Noah A. Smith
annee: 2019
journal_conf: ACL 2019, Florence — University of Washington
date_lecture: 2026-05-20
pertinence: 4
tags: [attention, xai, faithfulness, transformer, interpretability, h1-motivation]
---

# Is Attention Interpretable?

> Citation BibTeX-key : `Serrano2019Attention` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🔍 **Troisième voix dans le débat attention/explication.** Teste directement sur des architectures Transformer (pas seulement LSTM). Conclut que l'attention est *partiellement* informative mais pas *fidèle* au sens strict.

## Problème traité

Jain & Wallace 2019 et Wiegreffe & Pinter 2019 débattent sur LSTM + attention additive. Mais les Transformers utilisent un mécanisme d'attention différent (multi-head, scaled dot-product). La critique s'applique-t-elle également ?

## Méthode

- Modèles : BERT et autres Transformers sur tâches NLP.
- **Expérience 1 : zeroing attention** — mettre à zéro certains poids d'attention et mesurer la variation de la prédiction.
- **Expérience 2 : gradient-attention comparison** — comparer les poids d'attention aux gradients de la perte par rapport aux inputs (mesure de saillance "ground truth").
- Évaluation via corrélation de rang entre les deux mesures.

## Résultats clés

- Mettre à zéro les poids d'attention les plus élevés **ne change souvent pas la prédiction** de manière significative → l'attention n'est pas causalement nécessaire.
- La corrélation entre poids d'attention et gradients est **faible à modérée** selon les couches et les têtes.
- **Conclusion** : les poids d'attention donnent une image *partiellement* utile du comportement du modèle, mais ne constituent pas des explications fidèles au sens de la causalité.

## Limites identifiées par les auteurs

- Résultats variables selon les couches (les dernières couches sont souvent plus informatives).
- L'analyse dépend du modèle et de la tâche.
- Pas de solution alternative proposée — diagnostic uniquement.

## Lien avec H1

Étend la critique de Jain & Wallace aux Transformers — exactement l'architecture que nous utilisons (TimeSeriesTransformer HuggingFace). Renforce l'argument : la cross-attention de notre modèle de base est sujette à la même critique, ce qui justifie l'Evidence Layer.

À citer en slide 32 (H1.B reframé) : la divergence M/cross-attention n'est pas une limite de M, c'est la cross-attention qui est non fiable.

## Citations à réutiliser

> "We find that attention weights are weakly correlated with gradient-based measures of importance, suggesting that they are not a reliable indicator of feature importance." (Abstract)

> "The extent to which attention weights are interpretable as explanations of model behavior remains an open question." (Conclusion)

## Idées à creuser

- Complémentaire à Jain & Wallace 2019 : les deux s'appliquent aux Transformers TS que nous utilisons.
- Argument H1 renforcé : M dans l'Evidence Layer est *mathématiquement* fidèle, là où l'attention est au mieux *corrélée*.
