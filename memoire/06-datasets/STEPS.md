# Steps — 06-datasets

- [x] Identifier la source du dataset FAYAM (Azure Functions Trace supposé — à confirmer encadreur)
- [x] Télécharger dans `raw/` — clusters 0, 4, 6, 8 (19 fonctions × 20 160 min)
- [x] Rédiger `DATA-CARD.md` (source, taille, schéma, profils)
- [ ] Confirmer la licence / source exacte avec l'encadreur
- [ ] Lancer le prétraitement → `processed/`
  - [ ] Normalisation (min-max ou z-score ?)
  - [ ] Découpage en fenêtres (contextLength × predictionLength — à caler avec les hyperparamètres FAYAM)
  - [ ] Split train/val/test
- [ ] Vérifier reproductibilité (seed, version du script)
