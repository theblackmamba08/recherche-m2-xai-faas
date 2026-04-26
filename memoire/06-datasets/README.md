# 06-datasets — Données

## Structure

- `raw/` — données brutes, **gitignored** (lourdes, parfois licenciées)
- `processed/` — données prétraitées, **gitignored**
- `scripts/` *(optionnel)* — scripts de prétraitement (s'ils sont triviaux ; sinon dans `../../code/src/data/`)
- `DATA-CARD.md` — fiche descriptive (source, taille, schéma, licence, prétraitement)

## Règle

Le code de prétraitement vit dans [`../../code/src/data/`](../../code/src/data/). Ce dossier ne contient que les **données** et leur **fiche**.
