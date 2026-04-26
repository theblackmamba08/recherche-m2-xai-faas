# Steps — 03-contribution

## Phase H1

- [ ] Activer `output_attentions=True` sur Transformer FAYAM
- [ ] Sauvegarder les attention maps par échantillon (format à choisir : `.npz` / `.parquet`)
- [ ] Implémenter DBSCAN sur les charges (script dans `../../code/src/`)
- [ ] Visualiser attention moyenne par cluster (matplotlib)
- [ ] Implémenter `comprehensiveness` et `sufficiency`
- [ ] Tableau de résultats par cluster

## Phase H2 *(si H1 stable en S8)*

- [ ] Installer Captum
- [ ] Calculer IG sur 100 échantillons représentatifs
- [ ] Comparaison qualitative et quantitative IG vs attention
