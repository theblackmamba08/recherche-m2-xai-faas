# redaction/ — Mémoire LaTeX (template Université de Dschang)

Rédaction finale du mémoire. **Le template Dschang sera importé ici** dès qu'il sera disponible.

## Structure prévue

| Élément | Rôle |
|---|---|
| `main.tex` | Point d'entrée |
| `chapitres/` | Un fichier `.tex` par chapitre |
| `biblio/refs.bib` | Bibliographie BibTeX |
| `figures/` | Figures importées (PDF / PNG / TikZ) |
| `template-dschang/` *(à venir)* | Cls / sty fournis par l'université |
| `STEPS.md`, `MEMOIRE.md` | Pilotage |

## Workflow

1. **Écrire** dans `chapitres/` (1 fichier par chapitre).
2. Avant compilation : invoquer le skill **`biblio-sync`** pour vérifier que les `\cite{}` ont une fiche dans [`../memoire/01-litterature/fiches/`](../memoire/01-litterature/fiches/) ET une entrée dans `biblio/refs.bib`.
3. Compilation : skill **`latex-compile`** (gère `latexmk -pdf` + biber/bibtex).
4. Avant remise : skill **`audit-similarite`** + skill **`redaction-humaine`**.

## Compilation manuelle

```bash
cd redaction/
latexmk -pdf main.tex
# ou en mode continu :
latexmk -pvc -pdf main.tex
```

## Compilation dans Antigravity / VS Code

Extension recommandée : **LaTeX Workshop** (James-Yu).
Réglage `latex-workshop.latex.recipes` à pointer sur `latexmk -pdf`.
