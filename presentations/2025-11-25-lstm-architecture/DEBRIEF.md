# DEBRIEF — Architecture LSTM : du Neurone Biologique à la Mémoire Longue Terme

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global des encadreurs

L'encadreur a constaté que l'étudiant maîtrise l'idée d'ensemble de l'architecture LSTM et de sa généalogie. Cette validation a déclenché le passage à la phase pratique : fourniture de datasets et mission d'implémentation.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Validation de la compréhension globale de l'architecture LSTM | Encadreur | Démonstration jugée satisfaisante | Passer à l'implémentation |
| 2 | Chercher des **articles présentant des architectures LSTM pour la prédiction** de séries temporelles | Encadreur | Noté | Revue de littérature ciblée LSTM + prédiction |
| 3 | **Implémenter ces architectures** et les tester sur les datasets fournis | Encadreur | Noté | Coder, entraîner, évaluer |

## Articles recommandés

| Titre | Recommandé par | Priorité |
|-------|---------------|----------|
| *Foundation Models for Time Series: A Survey* | Dr DJOUMESSI Kerol | Haute |

## Ressources fournies par les encadreurs

- **Datasets** : fournis par l'encadreur à l'issue de la séance *(à documenter précisément dans `memoire/06-datasets/` : noms, sources, format, granularité)*.

## Action items (avant la prochaine présentation)

- [ ] Lire et ficher **"Foundation Models for Time Series: A Survey"** (recommandé par Dr DJOUMESSI) — le déposer dans `memoire/01-litterature/articles/` et invoquer le skill `fiche-lecture`.
- [ ] **Documenter les datasets** reçus dans `memoire/06-datasets/` : nom, source, format, granularité, taille.
- [ ] Faire une **revue de littérature** : chercher des articles présentant des architectures LSTM pour la prédiction de séries temporelles (priorité : FaaS/cloud, mais pas exclusif).
- [ ] **Implémenter au moins une architecture LSTM** de référence trouvée dans la littérature.
- [ ] Entraîner et évaluer sur les datasets fournis — métriques : sMAPE, RMSE, R² (alignées sur FAYAM).
- [ ] Préparer une présentation des résultats pour la prochaine séance.
- [ ] Étape suivante après LSTM : **étudier le Transformer** (TimeSeriesTransformer HuggingFace de FAYAM).

## Prochaine présentation

Présenter les articles LSTM trouvés, l'architecture implémentée, et les résultats de prédiction sur les datasets fournis.
