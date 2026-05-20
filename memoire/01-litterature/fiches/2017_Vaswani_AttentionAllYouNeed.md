---
type: fiche-lecture
article: 2017_Vaswani_AttentionAllYouNeed.pdf
auteurs: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
annee: 2017
journal_conf: NeurIPS 2017, Long Beach — Google Brain / Google Research
date_lecture: 2026-05-20
pertinence: 4
tags: [transformer, attention, architecture, encoder-decoder, multi-head, h1-architecture]
---

# Attention Is All You Need

> Citation BibTeX-key : `Vaswani2017Transformer` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🏛️ **Référence fondatrice du Transformer.** À citer pour justifier l'architecture encoder-décodeur de notre modèle de base (TimeSeriesTransformer HuggingFace) et le mécanisme d'attention que nous étendons via l'Evidence Layer.

## Problème traité

Les modèles séquence-à-séquence (RNN, LSTM, GRU) souffrent de deux limites : la difficulté de parallélisation (les calculs sont séquentiels) et la dégradation de l'information sur longues séquences (gradient vanishing). L'objectif est de proposer une architecture basée uniquement sur l'attention, sans récurrence ni convolution.

## Méthode

### Architecture Transformer

- **Encoder** : N=6 couches identiques, chacune composée de (1) Multi-Head Self-Attention, (2) Feed-Forward Network, avec résidus et LayerNorm.
- **Decoder** : N=6 couches, chacune composée de (1) Masked Multi-Head Self-Attention, (2) **Cross-Attention sur l'encodeur**, (3) Feed-Forward Network.
- **Attention scalée** : `Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V`.
- **Multi-Head** : h=8 têtes parallèles, chacune avec ses propres matrices `W_q, W_k, W_v`.
- **Positional Encoding** : encodage sinusoïdal des positions, car pas de récurrence.

### Points clés pour notre H1

- La **cross-attention du décodeur** (`Q` = sorties du décodeur, `K,V` = sorties de l'encodeur) est le mécanisme que H1 cherche à supplanter par l'Evidence Layer. C'est ce mécanisme qui, selon Jain & Wallace 2019, ne garantit pas la fidélité.
- Le `dec_output` utilisé dans notre formule `M = softmax(Linear(dec_output))` est précisément la sortie des couches décodeur de cette architecture.

## Résultats clés

- Sur traduction EN→DE (WMT 2014) : **BLEU = 28.4**, état de l'art de l'époque.
- Entraînement **8× plus rapide** que les meilleurs modèles récurrents grâce à la parallélisation.
- Généralise bien à d'autres tâches séquentielles.

## Limites identifiées par les auteurs

- Complexité O(n²) en mémoire et en calcul pour les longues séquences.
- Nécessite un encodage positionnel explicite (pas de notion de position intrinsèque).
- Pas conçu pour les séries temporelles — adaptation requise (c'est ce que fait HuggingFace TimeSeriesTransformer).

## Lien avec H1

Référence fondatrice à citer pour :
1. Slide 7 : justifier l'architecture de base (encoder-décodeur, cross-attention).
2. Slide 5 : montrer que la cross-attention du décodeur est exactement ce que Jain & Wallace critiquent.
3. Équation centrale : notre `M = softmax(Linear(dec_output))` est une modification structurelle du mécanisme de cross-attention standard `softmax(QKᵀ/√d_k)`.

## Citations à réutiliser

> "Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences." (Introduction)

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions." (Section 3.2)

> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms." (Abstract)

## Idées à creuser

- Le TimeSeriesTransformer HuggingFace (Rasul et al.) est une adaptation directe de cette architecture pour les séries temporelles — à citer conjointement.
- La comparaison entre notre `M = softmax(Linear(dec_output))` et `Attention = softmax(QKᵀ/√d_k)` est au cœur de l'argument de fidélité par construction (table slide 7).
