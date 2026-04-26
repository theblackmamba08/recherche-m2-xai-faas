# hpc/ — Scripts cluster (SLURM)

Ce dossier ne contient **que** des scripts batch et de l'environnement.
Le code Python vit dans [`../code/`](../code/) — règle de **source unique**.

## Structure

| Sous-dossier | Rôle |
|---|---|
| `slurm/` | Scripts `*.sbatch` à soumettre via `sbatch` |
| `envs/` | Specs d'environnement (`environment.yml`, `module-list.txt`) |
| `logs/` | Logs `slurm-*.out` (gitignored) |

## Conventions

- Chaque script `slurm/*.sbatch` `cd` dans `$SLURM_SUBMIT_DIR/../code/` puis lance Python.
- Variables `SBATCH` en tête : `--job-name`, `--time`, `--gres=gpu:1`, `--output=../logs/%x-%j.out`.
- Toujours `module purge && module load <stack>` avant l'exécution.

## Workflow type

```bash
ssh user@cluster
cd recherche-m2-xai-faas/
git pull
sbatch hpc/slurm/h1-baseline.sbatch
squeue -u $USER
tail -f hpc/logs/h1-baseline-<jobid>.out
```

## One-click local → cluster

Voir `hpc/sync.sh` *(à créer)* qui rsync le code vers le cluster et lance le sbatch en une commande.
