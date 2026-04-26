# Mémoire M2 — XAI pour FaaS

**Titre** : Prédiction des charges de travail et atténuation du cold start dans les architectures FaaS : approches basées sur le Deep Learning et l'explicabilité

**Étudiant** : *(à compléter)*
**Université** : Université de Dschang
**Année universitaire** : 2025-2026
**Encadrant** : *(à compléter)*

---

## Résumé

Ce projet étend les travaux baseline (FAYAM) sur la prédiction de charge FaaS en ajoutant des mécanismes d'explicabilité (XAI) au modèle Transformer. L'objectif est de rendre les prédictions interprétables sans dégrader les performances, via l'analyse des poids d'attention et leur évaluation par faithfulness (comprehensiveness + sufficiency), avec une lecture différentielle par cluster DBSCAN.

## Hypothèse principale (H1)

XAI par attention weights sur le Transformer HuggingFace de FAYAM — extraction, visualisation par cluster DBSCAN, évaluation faithfulness.

## Structure du dépôt

| Dossier | Rôle |
|---------|------|
| [`memoire/`](memoire/) | Recherche : littérature, baseline, contributions, datasets, ressources |
| [`redaction/`](redaction/) | Mémoire LaTeX (template Université de Dschang) |
| [`presentations/`](presentations/) | Slides Beamer (encadreurs, séminaires, soutenance) |
| [`code/`](code/) | **Source unique** du code (modèles, expériences, tests) |
| [`hpc/`](hpc/) | Scripts SLURM/sbatch — appellent `code/`, ne dupliquent rien |

Chaque dossier contient un `README.md` (rôle public), un `STEPS.md` (étapes restantes) et un `MEMOIRE.md` (trace de ce qui a été fait).

## Démarrer

```bash
# 1. Cloner le dépôt
git clone <url> recherche-m2-xai-faas
cd recherche-m2-xai-faas

# 2. Code Python
cd code
python -m venv .venv
source .venv/bin/activate   # ou .venv/Scripts/activate sous Windows
pip install -e .

# 3. Compiler le mémoire LaTeX
cd ../redaction
latexmk -pdf main.tex
```

## Licence

*(à définir — académique par défaut)*
