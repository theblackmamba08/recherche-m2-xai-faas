# Roadmap — < 3 mois jusqu'à soutenance

> Démarrage : 2026-04-26 — Soutenance cible : *(à fixer)*

## Phase 1 — Reproduction baseline (S1-S2)

- [ ] Cloner / installer le code FAYAM
- [ ] Reproduire les résultats Transformer (pas le CNN-LSTM)
- [ ] Documenter écarts éventuels dans `02-baseline/MEMOIRE.md`

## Phase 2 — H1 : Attention + DBSCAN + Faithfulness (S3-S6)

- [ ] Activer `output_attentions=True` sur le Transformer HuggingFace
- [ ] Extraire et stocker les attention maps par échantillon
- [ ] Clustering DBSCAN sur les charges → 3-5 profils
- [ ] Visualiser attention moyenne par cluster
- [ ] Implémenter faithfulness : comprehensiveness + sufficiency
- [ ] Analyse différentielle inter-clusters

## Phase 3 — Bonus H2 (S7-S8 si H1 stable)

- [ ] Integrated Gradients via Captum
- [ ] Comparer avec attention weights

## Phase 4 — Rédaction (S9-S12)

- [ ] Plan détaillé du mémoire
- [ ] Chapitre 1 : intro + état de l'art
- [ ] Chapitre 2 : méthode
- [ ] Chapitre 3 : résultats
- [ ] Chapitre 4 : discussion
- [ ] Relecture, audit similarité, soumission

## Jalons encadreurs

- *(à compléter après chaque entretien)*

---

## État courant

> 📍 **Première chose à lire en début de session.** Mis à jour à chaque fin de session par le hook Stop.

### Dernière session : 2026-04-26

- **Phase actuelle** : Pré-phase 1 (setup du dépôt)
- **Avancée récente** : Arborescence complète posée. Skills installés. Hook Stop actif (MEMOIRE.md + JOURNAL.md auto-vérifiés).
- **Bloquants** : aucun. En attente de :
  - Template LaTeX Dschang
  - Choix du cluster HPC cible
- **Prochain pas concret** : démarrer la **phase 1** (récupérer code FAYAM, reproduire la baseline Transformer).
- **Décisions du jour** : aucune nouvelle (cf. [DECISIONS.md](DECISIONS.md))
