---
type: fiche-lecture
article: 2019_Jain_AttentionNotExplanation.pdf
auteurs: Sarthak Jain, Byron C. Wallace
annee: 2019
journal_conf: NAACL-HLT 2019, Minneapolis — Northeastern University / Boston University
date_lecture: 2026-05-20
pertinence: 5
tags: [attention, xai, faithfulness, controversy, transformer, h1-motivation]
---

# Attention is not Explanation

> Citation BibTeX-key : `Jain2019Attention` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🎯 **Article fondateur de la motivation H1.** Démontre empiriquement que les poids d'attention ne constituent pas des explications fiables, ouvrant la porte à des alternatives comme notre Evidence Layer.

## Problème traité

Les poids d'attention dans les RNN et Transformers sont largement utilisés comme explications a posteriori : "le modèle prête attention à ce token, donc ce token est important". Mais est-ce vraiment vrai ? L'article teste cette hypothèse fondamentale.

## Méthode

- **Datasets** : 8 tâches NLP (classification de texte, NLI, QA) avec annotations de saillance (SNLI, SST, etc.).
- **Expérience 1 : adversarial attention** — construire une distribution d'attention différente qui produit la même prédiction. Si c'est possible → l'attention n'est pas l'unique explication.
- **Expérience 2 : permutation** — permuter ou uniformiser les poids d'attention. Mesure la variation de la sortie via Total Variation Distance (TVD).
- Modèles testés : LSTM + attention additive (Bahdanau).

## Résultats clés

- Il **existe** des distributions d'attention adversariales (très différentes de l'originale) qui produisent des prédictions quasi-identiques sur la majorité des exemples.
- Permuter ou uniformiser l'attention change la sortie de manière **non significative** sur la plupart des tâches.
- **Conclusion principale** : les poids d'attention ne peuvent pas être interprétés comme des mesures causales de l'importance des tokens. Attention ≠ explanation.

## Limites identifiées par les auteurs

- Résultats limités aux modèles LSTM + attention (pas Transformer pur).
- La critique porte sur l'utilisabilité des poids comme *explications*, pas sur leur utilité pour la prédiction.
- Wiegreffe & Pinter 2019 proposent une réfutation partielle.

## Lien avec H1

**Argument central** : si les poids d'attention ne sont pas des explications fiables, il faut un mécanisme qui garantit la fidélité par construction — c'est exactement ce que fait M dans notre Evidence Layer. M est le coefficient algébrique *exact* de `bmm(M, enc_hidden)`, pas une corrélation approximée.

À citer pour motiver H1 vs TFT (slide 5 de la présentation) et pour justifier que H1.B (divergence M/cross-attention) est une caractéristique et non un défaut.

## Citations à réutiliser

> "We show that attention weights are frequently uncorrelated with gradient-based measures of feature importance, and that they often fail to identify features considered important by other, more principled methods." (Abstract)

> "One can construct adversarial attention distributions — distributions that differ markedly from the original yet yield equivalent predictions." (Section 3)

> "Attention is not explanation." (Titre)

## Idées à creuser

- Voir aussi : Wiegreffe & Pinter 2019 (contre-point), Serrano & Smith 2019 (convergence).
- Le résultat s'applique aux LSTM ; pour le Transformer, la question reste partiellement ouverte — Serrano & Smith 2019 étendent la critique aux architectures Transformer.
