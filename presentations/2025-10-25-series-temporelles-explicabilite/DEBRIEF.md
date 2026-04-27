# DEBRIEF — Séries Temporelles : De la Prédiction à la Confiance par l'Explicabilité

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global des encadreurs

Nette amélioration par rapport à la présentation #1. Les termes "explicabilité" et "interprétabilité" sont apparus dans la présentation — point positif, même si leur maîtrise conceptuelle reste à approfondir par l'étudiant.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Qu'est-ce qu'on attend **en sortie** quand on donne une série temporelle à un modèle ? | Encadreur | "Une prédiction" | Trop réducteur : lire des articles sur les sorties possibles (prédiction ponctuelle, intervalle de confiance, classification, détection d'anomalie, génération…) |
| 2 | Présenter les **données dans un modèle précis** et dire comment on les obtient | Encadreurs | Non traité | Choisir un dataset concret (ex. FaaS/cloud), décrire sa structure, son acquisition, sa granularité |
| 3 | Les **images sont toujours insuffisantes** — effort à amplifier | Encadreurs | Reconnu (1 seul graphique présent) | Ajouter des figures : pipeline XAI, cartes d'attention, exemples SHAP/LIME, architecture Transformer |
| 4 | Les **domaines d'application** pas assez approfondis malgré leur présence | Encadreurs | Cités mais pas illustrés | Illustrer avec des cas réels et des résultats chiffrés (ex. détection anomalie réseau, prédiction charge cloud) |
| 5 | **Aucune référence bibliographique** encore — répétition du reproche de la séance #1 | Encadreurs | Reconnu | Obligation : inclure des citations IEEE dans toutes les présentations futures, sans exception |
| 6 | Prendre des **modèles existants et donner des résultats en termes d'explicabilité** | Encadreurs | Non traité | Présenter des résultats concrets : ex. "LIME appliqué à ARIMA donne telle importance aux 3 derniers lags" |
| 7 | Faire une **étude comparative des différentes approches d'explicabilité** | Encadreurs | Non traité | Tableau comparatif : LIME vs SHAP vs Attention vs CAM — avantages, inconvénients, applicabilité au temporel |

## Critiques / suggestions

- La **narrative** est meilleure (fil conducteur prédiction → confiance → XAI), mais la profondeur technique reste insuffisante.
- L'explicabilité est **évoquée** mais pas encore opérationnalisée : pas de méthode présentée, pas de résultat montré.
- **Aucun dataset concret** n'est utilisé pour illustrer les propos.
- Le **manque de références** est souligné pour la deuxième fois — point critique à corriger absolument.

## Action items (avant la prochaine présentation)

- [ ] Lire des articles sur les **types de sorties** d'un modèle de séries temporelles (prédiction, intervalle, classification, anomalie).
- [ ] Choisir un **dataset cloud/FaaS réel**, le décrire précisément (source, granularité, taille, format).
- [ ] Préparer un **tableau comparatif** des approches XAI pour le temporel : LIME, SHAP (TimeSHAP), Attention weights, CAM/SoftCAM — avantages, inconvénients, limites.
- [ ] Intégrer des **résultats concrets d'explicabilité** : au moins un exemple numérique ou visuel.
- [ ] **Multiplier les figures** : pipeline, architecture, cartes d'activation, sorties SHAP.
- [ ] **Inclure des références IEEE** dans chaque diapositive qui l'exige — obligation non négociable.

## Prochaine présentation

Passer de la théorie à la pratique : choisir un modèle, un dataset, une méthode XAI, et montrer des résultats. Comparaison des approches d'explicabilité avec preuves à l'appui.
