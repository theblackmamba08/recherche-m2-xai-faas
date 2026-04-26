# memoire/ — Travail de recherche

Ce dossier contient **toute la matière scientifique** du mémoire (hors code source). La rédaction LaTeX finale est dans [`../redaction/`](../redaction/), le code dans [`../code/`](../code/).

## Structure

| Sous-dossier | Contenu |
|---|---|
| [`00-meta/`](00-meta/) | Journal de bord, roadmap, questions ouvertes, décisions méthodo |
| [`01-litterature/`](01-litterature/) | Articles lus (PDF + fiches structurées) |
| [`02-baseline/`](02-baseline/) | Reproduction du mémoire FAYAM, résultats de référence |
| [`03-contribution/`](03-contribution/) | Hypothèses (H1, H2), expériences, résultats originaux |
| [`05-ressources/`](05-ressources/) | Cours, tutoriels, snippets utiles |
| [`06-datasets/`](06-datasets/) | `raw/` (gitignored) + `processed/` (gitignored) + scripts de prétraitement |

> Les dossiers `04-redaction/` et `07-presentations/` du plan initial ont été remontés à la racine ([`../redaction/`](../redaction/) et [`../presentations/`](../presentations/)) pour découpler outillage LaTeX et matière scientifique.

## Convention par dossier

Chaque sous-dossier contient :
- `README.md` : rôle public (visible sur GitHub)
- `STEPS.md` : étapes restantes
- `MEMOIRE.md` : trace de ce qui a été fait
