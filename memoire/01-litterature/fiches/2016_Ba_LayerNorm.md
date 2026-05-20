---
type: fiche-lecture
article: 2016_Ba_LayerNorm.pdf
auteurs: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
annee: 2016
journal_conf: arXiv:1607.06450 — University of Toronto / Google DeepMind
date_lecture: 2026-05-20
pertinence: 3
tags: [layer-normalization, training-stability, deep-learning, h1-fix, architecture]
---

# Layer Normalization

> Citation BibTeX-key : `Ba2016LayerNorm` *(à reporter dans `redaction/biblio/refs.bib`)*

> 🔧 **Référence technique pour Fix #4 de H1.** LayerNorm est l'ingrédient architectural qui a permis de stabiliser l'Evidence Layer (alignement statistique de h_evidence sur dec_output). À citer pour justifier ce choix technique.

## Problème traité

Batch Normalization (Ioffe & Szegedy 2015) améliore la stabilité de l'entraînement des réseaux profonds, mais a des limitations : elle dépend de la taille du batch, n'est pas applicable aux RNN (statistiques variables selon le pas de temps), et est difficile à utiliser en inférence online.

Layer Normalization propose une alternative qui normalise *le long des features* (et non du batch), résolvant ces problèmes.

## Méthode

Pour un vecteur d'activation `h` de dimension `d` :
```
LayerNorm(h) = γ · (h - μ) / (σ + ε) + β
```
où :
- `μ = mean(h)`, `σ = std(h)` — calculés sur les `d` features (pas sur le batch)
- `γ, β` — paramètres appris (scale + shift)
- `ε` — constante de stabilité numérique

**Propriété clé** : indépendant de la taille du batch → applicable aux séquences de longueur variable, aux RNN et aux Transformers.

## Résultats clés

- Améliore la convergence et la stabilité sur RNN, LSTM.
- Composant standard de l'architecture Transformer (Vaswani 2017 l'intègre directement après les sous-couches).
- Pas de coût computationnel significatif.

## Lien avec H1

**Fix #4 (Run B3)** : `h_evidence = bmm(M, enc_hidden)` et `dec_output` ont des statistiques très différentes (scales, variances). Sans alignement, le mix `(1-mix)·dec_output + mix·h_evidence` est déstabilisé. LayerNorm sur `h_evidence` avant le mix aligne les deux distributions :
```python
h_evidence_normed = self.evidence_norm(h_evidence)  # LayerNorm
h = (1 - mix) * dec_output + mix * h_evidence_normed
```

Résultat : R² passe de −1.97 (B2, sans LayerNorm) à −1.59 (B3, avec LayerNorm), et max_weight de M descend de 0.97 → 0.06 (M moins effondrée).

À citer en slide 13 (Run B3 : LayerNorm) et slide 7 (architecture finale).

## Citations à réutiliser

> "Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks." (Abstract)

> "Unlike batch normalization, layer normalization performs exactly the same computation at training and test times." (Section 3)

## Idées à creuser

- LayerNorm est aussi présent dans le Transformer de base (Vaswani 2017) — notre usage est cohérent avec le paradigme de l'architecture.
- L'alignement statistique h_evidence / dec_output est la raison fondamentale pour laquelle B3 améliore B2 : à documenter clairement dans le chapitre H1.
