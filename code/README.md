# code/ — Source unique du code

**Tout le code Python du mémoire vit ici.** [`../hpc/`](../hpc/) ne contient que des scripts SLURM qui invoquent ce dossier — **aucune duplication**.

## Structure

| Sous-dossier | Rôle |
|---|---|
| `src/` | Modules importables (modèles, data, XAI, métriques) |
| `tests/` | Tests `pytest` |
| `notebooks/` | Notebooks d'exploration / visualisation |
| `experiments/` | Configurations d'expériences + sorties (`runs/`, gitignored) |
| `pyproject.toml` | Dépendances et installation |

## Installation

```bash
cd code/
python -m venv .venv
source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -e .[dev]
```

## Lancer une expérience

```bash
python -m src.experiments.run_h1 --config experiments/configs/h1-baseline.yaml
```

Le skill **`experiment-tracker`** crée automatiquement `experiments/<date>_<nom>/run.md` avec config, hyperparams, hash git, dataset utilisé.

## Tests

```bash
pytest
```
