# Mémoire — 02-baseline

## 2026-04-26

- Dossier créé.
- Pas encore de reproduction lancée.

## 2026-04-28

- Code FAYAM extrait de `transformer_m2-main.zip` → copié dans `code/src/baseline/fayam/`.
- Hyperparamètres Transformer identifiés : `prediction_length=120`, `context_length=240`, `freq="1T"`, `encoder_layers=4`, `decoder_layers=4`, `d_model=32`.
- Dataset disponible sur HuggingFace Hub : `FaalSa/dataME`.
- **Discordance importante** : le mémoire dit DBSCAN mais le code utilise HDBSCAN — paramètres non documentés.
- Notes complètes dans `code/src/baseline/fayam/BASELINE.md`.
- Prochaine étape : lancer `tsf_transf.py` sur nos 4 clusters pour reproduire les métriques FAYAM (voir `STEPS.md`).
