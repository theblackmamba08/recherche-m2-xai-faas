# Steps — code

## Phase 1 : reproduction baseline

- [ ] Importer le code FAYAM (Transformer uniquement) dans `src/models/`
- [ ] Ajouter scripts data-loading dans `src/data/`
- [ ] Vérifier reproductibilité : seed fixée, métriques stables sur 3 runs
- [ ] Tests unitaires de base dans `tests/`

## Phase 2 : H1

- [ ] `src/xai/attention_extract.py` : extrait `output_attentions=True`
- [ ] `src/xai/dbscan_clustering.py`
- [ ] `src/xai/faithfulness.py` : comprehensiveness + sufficiency
- [ ] `src/visualization/attention_heatmap.py`
- [ ] Notebook `notebooks/h1-analysis.ipynb`

## Phase 3 : H2 *(optionnel)*

- [ ] `src/xai/integrated_gradients.py` (Captum)
- [ ] Notebook comparaison
