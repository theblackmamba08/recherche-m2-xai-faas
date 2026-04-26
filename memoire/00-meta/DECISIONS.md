# Décisions méthodologiques

> Format : date — décision — alternatives écartées — justification.

## 2026-04-26 — Hypothèse prioritaire = H1 (attention + DBSCAN + faithfulness)

- **Alternatives écartées** :
  - H2 (Integrated Gradients via Captum) : reportée en bonus si H1 stable en S8.
  - H3 (CNN-LSTM PyTorch) : abandonnée — coût d'apprentissage trop élevé pour < 3 mois et niveau PyTorch débutant.
- **Justification** : HuggingFace expose `output_attentions=True` nativement → pas de ré-implémentation modèle. L'angle original est l'**analyse différentielle par cluster DBSCAN**, axe distinctif vis-à-vis de FAYAM.

## 2026-04-26 — Code unique dans `code/`, pas de duplication HPC

- **Alternatives écartées** : code dupliqué dans `hpc/` ; submodule git.
- **Justification** : éviter la divergence des versions, simplifier l'iteration locale ↔ cluster.

## 2026-04-26 — Fichiers Claude (CLAUDE.md, .claude/, memory/) non commités

- **Justification** : préférence personnelle. L'explicabilité publique passe par les `README.md` de chaque dossier.
