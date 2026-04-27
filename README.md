# Mémoire M2 — XAI pour FaaS

**Titre** : Prédiction des charges de travail et atténuation du cold start dans les architectures FaaS : approches basées sur le Deep Learning et l'explicabilité

**Étudiant** : *FOSSO TIOFOUET Jospin Cabrel*
**Université** : Université de Dschang
**Année universitaire** : 2025-2026
**Encadrant** : *Pr KENGNE TCHENDJI Vianney*

---

## Résumé

Ce projet étend les travaux baseline (FAYAM, Université de Dschang, 2024) sur la prédiction de charge FaaS en ajoutant une couche d'**explicabilité (XAI)** au modèle Transformer utilisé pour l'atténuation du cold start. L'objectif est de rendre les prédictions **interprétables sans dégrader les performances**, en privilégiant — autant que possible — une **interprétabilité intrinsèque** (le modèle explique sa décision en un seul forward pass, plutôt qu'a posteriori).

## Hypothèses de recherche

Les trois pistes ci-dessous forment un **continuum d'ambition décroissante**. Si la première bloque, la suivante prend le relais.

### H1 — *prioritaire* — XAI intrinsèque par modification architecturale

Adapter le principe **SoftCAM** (Djoumessi & Berens, arXiv:2505.17748, 2025) au `TimeSeriesTransformer` HuggingFace de FAYAM : remplacer la projection finale du décodeur par une opération qui produit nativement une *carte d'évidence temporelle*, régularisée par ElasticNet. Les auteurs SoftCAM mentionnent l'extension à ViT comme perspective ouverte ; nous proposons la **première transposition à un Transformer de séries temporelles** appliqué aux charges FaaS.

### H2 — *repli méthodologique* — XAI post-hoc fondée sur Shapley

Si H1 bute sur un verrou conceptuel ou technique, basculer sur des approches **SHAP-based** (TimeSHAP, KernelSHAP), conçues pour les séries temporelles et théoriquement bien fondées.

### H3 — *dernier recours* — XAI par poids d'attention

Si H1 et H2 échouent, étudier directement les **poids d'attention** du Transformer (`output_attentions=True`), avec analyse différentielle par cluster DBSCAN et évaluation par *comprehensiveness/sufficiency*. Cette piste, moins ambitieuse, peut aussi servir d'**outil de validation** de H1.

### Hypothèse abandonnée
Implémentation CNN-LSTM en PyTorch (Modèle 1 de FAYAM) : coût d'apprentissage trop élevé pour le calendrier (< 3 mois).

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
