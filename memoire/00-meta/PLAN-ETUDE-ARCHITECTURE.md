# Plan d'étude — Architecture Transformer & Pipeline données (5 jours)

> Initié : 2026-05-04  
> Objectif : cartographier précisément le `TimeSeriesTransformerForPrediction` HuggingFace  
> Livrable final : `memoire/02-baseline/CARTOGRAPHIE.md` (~3 pages + 5-6 schémas Excalidraw)

---

## Calendrier

| Jour | Tâche | Livrable |
|------|-------|----------|
| **J1** | Lancer Colab le matin. Lire le blog HuggingFace (Rasul & Rogge) + Illustrated Transformer (Jay Alammar). Dessiner en Excalidraw l'architecture globale encoder-decoder. | Esquisse architecture globale |
| **J2** | Étude pipeline de données (sections 4-5 du notebook). Instrumenter avec `print(shape)` à chaque étape de transformation. Schéma data flow avec shapes annotées. | Schéma pipeline données |
| **J3** | Étude encodeur. `print(model.encoder)` + lecture du source HuggingFace. Schéma d'une couche encodeur (MultiHeadAttention → AddNorm → FFN → AddNorm). | Schéma couche encodeur |
| **J4** | Étude décodeur + cross-attention + tête de distribution (StudentT). Schéma couche décodeur. | Schéma couche décodeur + tête |
| **J5** | Étude inférence + métriques (MASE, sMAPE, R², Spearman). Document de synthèse dans `02-baseline/MEMOIRE.md`. Ranger les esquisses dans `memoire/02-baseline/figures/`. | `CARTOGRAPHIE.md` finalisé |

---

## Les 3 questions à maîtriser

### A. Préparation des données
- Tracer le shape exact de chaque tenseur d'entrée de `model(...)`
- Comprendre InstanceSplitter : comment une série de 20 160 min → fenêtres `(past, future)`
- Comprendre `lags_sequence` pour `freq="1T"` (afficher avec `print(lags_sequence)`)
- Comprendre les 5 time features cycliques + age feature (log-scale)
- Comprendre le `mean_scaler` interne du modèle

### B. Architecture du modèle
- Embedder de features statiques catégorielles (`embedding_dimension=[2]`)
- 4 blocs encodeur : MultiHeadAttention → AddNorm → FFN → AddNorm
- 4 blocs décodeur : MaskedSelfAttn → AddNorm → CrossAttn → AddNorm → FFN → AddNorm
- Tête de distribution StudentT : `d_model=32` → paramètres `(loc, scale, df)`
- Où se trouve la projection finale du décodeur → **cible de H1 SoftCAM**

### C. Sorties et interprétation
- `model.generate()` → `out.sequences` shape `(batch, 100, 120)`
- Pourquoi médiane et pas moyenne ?
- Interpréter MASE > 1 (pire que naive), R² négatif (possible)
- Lien entre distribution prédictive et intervalles de confiance

---

## Ressources (strictement limitées à celles-ci)

| Ressource | Quoi lire |
|-----------|-----------|
| *The Illustrated Transformer* — Jay Alammar | Architecture Transformer visuelle |
| HuggingFace blog — Rasul & Rogge | Exactement le `TimeSeriesTransformer` utilisé |
| `modeling_time_series_transformer.py` (GitHub HF) | Source du modèle pour Phase 2 |
| *The Annotated Transformer* — Harvard NLP | Uniquement si besoin de PyTorch line-by-line |

---

## Outils de visualisation

| Usage | Outil | Quand |
|-------|-------|-------|
| Esquisses de travail | **Excalidraw** (excalidraw.com) | J1–J5 |
| Pipeline données, flèches, shapes | Excalidraw ou Mermaid | J2 |
| Heatmaps attention | Matplotlib + seaborn | Déjà dans notebook |
| Figures finales mémoire | **TikZ** (LaTeX vectoriel) | Phase 5 uniquement |

---

## Pièges à éviter

- Ne pas faire de TikZ maintenant (coûteux). Excalidraw d'abord.
- Ne pas lire plus de papers que la liste ci-dessus.
- Annoter les shapes sur chaque flèche des schémas — c'est ce qui fait la différence à la soutenance.
- Produire des esquisses fonctionnelles, pas des peintures.
- Tracker quotidiennement dans `02-baseline/MEMOIRE.md`.

---

## Suivi

- [ ] J1 terminé
- [ ] J2 terminé
- [ ] J3 terminé
- [ ] J4 terminé
- [ ] J5 terminé — `CARTOGRAPHIE.md` livré
