# Steps — hpc

- [ ] Identifier le cluster cible (Dschang ? Grid5000 ? Jean Zay ?)
- [ ] Lister les modules disponibles → `envs/module-list.txt`
- [ ] Écrire `envs/environment.yml` (ou requirements.txt) compatible cluster
- [ ] Premier `slurm/h1-baseline.sbatch` (run reproductible H1)
- [ ] Script `sync.sh` rsync local → cluster
- [ ] Tester le pipeline complet sur 1 epoch
