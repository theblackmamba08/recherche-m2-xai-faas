# DEBRIEF — Analyse et Modélisation des Séries Chronologiques

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Qu'est-ce que le **call-start** et les problèmes de montée en charge ? | Encadreurs | Pas de réponse satisfaisante à ce stade | Lire sur cold-start / warm-start dans les architectures FaaS |
| 2 | Les séries temporelles s'appliquent à bien d'autres secteurs que le cloud — pourquoi ne pas le mentionner ? | Encadreurs | Reconnu | Illustrer dès la prochaine présentation (énergie, médecine, télécoms…) |
| 3 | Qu'est-ce que l'**explicabilité des modèles** de façon générale, et en particulier pour les séries temporelles ? | Encadreurs | Pas de réponse structurée | Préparer une définition claire XAI/IML + lien séries temporelles |
| 4 | Pourquoi pas de **références bibliographiques** dans les slides ? | Encadreurs | Oubli | Inclure systématiquement des références à partir de la prochaine présentation |
| 5 | Absence de **schémas et de figures** — « une image vaut mille mots » | Encadreurs | Reconnu | Intégrer des figures illustratives (décomposition, pipeline, architecture) |
| 6 | S'intéresser aux différentes **méthodes d'interprétabilité et d'explicabilité** (avantages, inconvénients, limites) | Encadreurs | Noté | Faire une revue LIME, SHAP, Attention, Grad-CAM… pour la prochaine séance |

## Critiques / suggestions

- Présentation jugée **peu structurée** : pas de fil conducteur clair entre séries temporelles et le sujet de mémoire XAI-FaaS.
- **Aucun lien établi** entre les séries chronologiques et le contexte FaaS / cloud / prédiction de charge.
- **Aucune référence** citée : les encadreurs ont insisté pour que toute présentation future en comporte.
- Les slides manquent de **schémas et de spécifications** concrètes.
- Le terme "call-start" (cold-start) n'a pas été mentionné, alors qu'il est central dans les problématiques FaaS.

## Action items (avant la prochaine présentation)

- [ ] Lire et comprendre les notions de **cold-start / warm-start** dans les architectures FaaS.
- [ ] Lire sur **edge computing** et **fork computing** pour élargir le cadre applicatif.
- [ ] Préparer une section dédiée à l'**explicabilité et l'interprétabilité des modèles** : définitions, méthodes (LIME, SHAP, Attention…), avantages/inconvénients, limites.
- [ ] Montrer que les séries temporelles s'appliquent **au-delà du cloud** (énergie, santé, industrie…).
- [ ] **Inclure des références** bibliographiques IEEE dans toutes les présentations futures.
- [ ] **Ajouter des schémas** : pipeline d'analyse, architecture FaaS, décomposition d'une série.

## Prochaine présentation

Retravailler la présentation en intégrant les retours ci-dessus : ancrer les séries temporelles dans le contexte FaaS, introduire XAI/IML, inclure références et figures.