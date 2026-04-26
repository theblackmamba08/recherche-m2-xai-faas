# presentations/ — Slides Beamer

Toutes les présentations du mémoire (entretiens encadreurs, séminaires, soutenance) suivent **le même flux** :

1. Création d'un dossier daté : `YYYY-MM-DD-thème-court/`
2. Skill **`brief-debrief-presentation`** → scaffold de `BRIEF.md` + `slides.tex` + (plus tard) `DEBRIEF.md`
3. Compilation Beamer via skill **`latex-compile`**
4. Après la présentation : remplir `DEBRIEF.md` (questions, action items, prochaine présentation)

## Convention de nommage

```
YYYY-MM-DD-mot-cle.....
2026-05-12-revue-litterature/
2026-06-30-mi-parcours/
2026-09-10-soutenance/
```

## Structure d'un dossier de présentation

```
2026-05-12-revue-litterature/
├── BRIEF.md          # AVANT : objectif, audience, points clés, questions anticipées
├── slides.tex        # Beamer
├── figures/          # figures spécifiques
├── build/            # PDF compilé (gitignore-friendly via .gitignore racine)
└── DEBRIEF.md        # APRÈS : questions reçues, action items, suite
```

## Template

Voir [`_template/`](_template/) — squelette à dupliquer / adapté par le skill `brief-debrief-presentation`.
