# DEBRIEF — Les Séries Temporelles : de la Prédiction à l'Explicabilité

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global des encadreurs

Nette amélioration constatée. La présentation couvre désormais un état de l'art XAI structuré (méthodes modèle-agnostiques, neuronales, spécialisées temporelles) avec des tableaux comparatifs. Les encadreurs ont posé de nombreuses questions sur les différentes méthodes présentées, signe d'un niveau de profondeur enfin atteint.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Questions approfondies sur **SHAP, LIME, Attention Mechanisms, Saliency Maps, Integrated Gradients, TimeSHAP, WindowSHAP, ShaTS** | Encadreurs | Réponses partielles — notions encore en cours de maîtrise | Approfondir chaque méthode, savoir les distinguer et les appliquer concrètement |
| 2 | S'intéresser aux **méthodes XAI appliquées aux modèles implémentés par FAYAM** — notamment SHAP et LIME sur les modèles LSTM et Transformer | Encadreurs | Noté | Lire les travaux de FAYAM, identifier les modèles exacts (LSTM + TimeSeriesTransformer HuggingFace), appliquer SHAP/LIME dessus |
| 3 | **Commencer par étudier l'architecture LSTM**, puis ensuite le Transformer — comprendre les entrées/sorties, la mémoire, le mécanisme d'attention | Encadreurs | Noté | Préparer une présentation dédiée : architecture LSTM (cellules, gates) → architecture Transformer (attention, encoder-decoder) |

## Critiques / suggestions

- Les méthodes XAI sont **présentées mais pas encore opérationnalisées** : on les cite, on les compare, mais on ne montre pas encore de résultats concrets sur un modèle et un dataset réels.
- La progression est visible, mais il faut maintenant **passer du théorique au pratique** : choisir un modèle (LSTM d'abord), un dataset FaaS/cloud réel, appliquer SHAP ou LIME, montrer les résultats.
- La compréhension fine des architectures LSTM et Transformer est un **prérequis identifié** avant d'aller plus loin en XAI.

## Action items (avant la prochaine présentation)

- [ ] Étudier en détail l'**architecture LSTM** : cellules LSTM, gates (input/forget/output), gestion de la mémoire à long terme, entrées/sorties.
- [ ] Étudier ensuite l'**architecture Transformer** (celui de FAYAM — `TimeSeriesTransformer` HuggingFace) : mécanisme d'attention, encoder-decoder, adaptation aux séries temporelles.
- [ ] Lire le mémoire FAYAM pour identifier précisément les modèles implémentés et les datasets utilisés.
- [ ] Préparer une **application concrète de SHAP et LIME** sur le LSTM de FAYAM — résultats à montrer lors de la prochaine séance.

## Prochaine présentation

Présenter les architectures LSTM et Transformer (dans cet ordre), puis montrer des résultats XAI concrets (SHAP/LIME) appliqués aux modèles de FAYAM sur un dataset cloud réel.
