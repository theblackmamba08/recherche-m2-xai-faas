# DEBRIEF — Foundation Models for Time Series: A Survey

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global des encadreurs

Dernière séance de l'année académique 2025. Les encadreurs ont pris acte des difficultés d'implémentation LSTM et ont réorienté le travail vers l'architecture Transformer via le papier originel.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Difficultés d'implémentation LSTM : articles sans code, ~4h/époque, aucun résultat cohérent | Étudiant (rapport) | Situation exposée | Encadreurs ont pris acte et réorienté |
| 2 | **Se concentrer sur l'architecture Transformer** via le papier originel *Attention Is All You Need* (Vaswani et al., 2017) | Encadreurs | Noté | Lire, comprendre et présenter l'architecture complète du Transformer |

## Décision de fin d'année

- **Implémentation LSTM suspendue** : difficultés techniques trop coûteuses en temps (4h/époque sans résultat).
- **Pivot vers le Transformer** : priorité donnée à la compréhension profonde de l'architecture Transformer via *Attention Is All You Need*.
- **Fin de l'année académique 2025** : la suite du travail reprend en 2026.

## Action items (début d'année 2026)

- [ ] Lire et ficher **"Attention Is All You Need"** (Vaswani et al., 2017) — papier originel du Transformer.
- [ ] Préparer une présentation détaillée de l'architecture Transformer : encoder, decoder, multi-head attention, positional encoding, feed-forward.
- [ ] Relier l'architecture Transformer au `TimeSeriesTransformer` HuggingFace utilisé par FAYAM.

## Prochaine présentation (2026)

Présentation de l'architecture Transformer complète à partir du papier *Attention Is All You Need*, avec lien explicite vers les modèles de FAYAM.
