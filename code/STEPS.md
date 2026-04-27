# Steps — code

> Refonte 2026-04-27 : nouveau pipeline H1 (SoftCAM-Transformer) → H2 (TimeSHAP) → H3 (attention).

## Phase 1 — Reproduction baseline FAYAM

- [ ] Importer le code FAYAM (Transformer uniquement, pas le CNN-LSTM) dans `src/models/fayam_transformer.py`
- [ ] Identifier les hyperparamètres manquants (`predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension` — non chiffrés p. 76 du mémoire FAYAM)
- [ ] `src/data/azure_loader.py` : chargement des fichiers Azure Functions 2019 (filtre HTTP, fusion 14 fichiers)
- [ ] `src/data/dbscan_clustering.py` : reproduction des 33 clusters HTTP
- [ ] **Activer `output_attentions=True` dès maintenant** pour disposer du matériel H3 sans réentraîner
- [ ] Reproductibilité : seed fixée, métriques stables sur 3 runs sur ≥ 3 datasets parmi les 18 (`HashFunction` table IX FAYAM p. 86)
- [ ] Tests unitaires de base dans `tests/`

## Phase 2 — H1 : SoftCAM-Transformer (priorité)

### Étude préalable
- [ ] Cartographier l'architecture HuggingFace `TimeSeriesTransformer` : localiser la projection finale du décodeur
- [ ] Récupérer et étudier le code SoftCAM (`https://anonymous.4open.science/r/SoftCAM-E1A3/`)

### Implémentation
- [ ] `src/models/softcam_transformer.py` : variante avec couche d'évidence (équivalent du `1×1 conv` SoftCAM mais en temporel)
- [ ] `src/training/elasticnet_loss.py` : perte combinée `prediction_loss + λ₁·||A||₁ + λ₂·||A||₂²`
- [ ] `src/xai/evidence_map.py` : extraction de la carte d'évidence à l'inférence
- [ ] `src/xai/faithfulness_temporal.py` : comprehensiveness + sufficiency adaptés à la régression temporelle
- [ ] `src/visualization/evidence_heatmap.py` : visualisation cartes temporelles
- [ ] Notebook `notebooks/h1-softcam-transformer.ipynb` : pipeline end-to-end

### Comparaison sanity check
- [ ] Comparer carte d'évidence apprise vs matrice d'attention décodeur (la cohérence est attendue ; divergence = signal d'alerte)

## Phase 3 — H2 : SHAP-based (repli)

- [ ] `src/xai/timeshap_runner.py` : TimeSHAP sur Transformer FAYAM
- [ ] `src/xai/kernelshap_runner.py` : KernelSHAP comparaison
- [ ] Notebook `notebooks/h2-shap-baseline.ipynb`

## Phase 4 — H3 : attention weights (dernier recours / sanity check)

- [ ] `src/xai/attention_extract.py` : déjà disponible si phase 1 a activé `output_attentions=True`
- [ ] `src/xai/cluster_attention_analysis.py` : agrégation par cluster DBSCAN
- [ ] Notebook `notebooks/h3-attention-analysis.ipynb`

## Tests

- [ ] `tests/test_data_loader.py`
- [ ] `tests/test_softcam_transformer.py` — vérifier dim sortie et shape evidence map
- [ ] `tests/test_elasticnet_loss.py` — vérifier différentiabilité et signe des gradients
