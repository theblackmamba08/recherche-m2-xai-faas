# BRIEF — Foundation Models for Time Series: A Survey

> Rempli a posteriori (présentation déjà réalisée).

- **Date** : 2025-12-13 (samedi) — dernière présentation de l'année 2025
- **Audience** : Dr LACMOU ZEUTOUO Jerry, Dr DJOUMESSI Kerol, Pr KENGNE TCHENDJI Vianney, M. Yvan
- **Durée** : non précisée
- **Format** : Beamer (15 diapositives)

## Objectif

Présenter la fiche de lecture de l'article *Foundation Models for Time Series: A Survey* (recommandé par Dr DJOUMESSI lors de la séance #4). Rapporter également les difficultés rencontrées lors de l'implémentation LSTM. Dernière séance de l'année académique 2025.

## Points clés présentés

1. **Définition d'un modèle de fondation** : modèle d'IA pré-entraîné sur de vastes ensembles de données, puis adapté (fine-tuné) à des tâches spécifiques. Analogie avec ChatGPT/GPT pour les séries temporelles.
2. **Domaines d'application** : Finance, Santé, Énergie, Transports, IoT & Industrie.
3. **Enjeux des séries temporelles** : nature séquentielle, multivarié, irrégularités, bruit/non-stationnarité, haute dimensionnalité.
4. **Flashback historique** (généalogie des modèles) :
   - Méthodes statistiques (ARIMA) : avantages/limites.
   - RNN : capture dépendances temporelles, limite vanishing gradient → transition LSTM.
   - LSTM/GRU : résolvent vanishing gradient, limite parallélisme → besoin Transformer.
   - Transformers (2017) : self-attention, parallélisme, scalabilité — shift paradigmatique récurrent→attention.
5. **Foundation Models** : apprennent patterns partagés sur données massives, généralisation cross-domain, efficacité sur low-data.
6. **Taxonomie des modèles** : architecture (encoder-only, decoder-only, encoder-decoder), patch vs non-patch, objectif (MSE/NLL), univarié/multivarié, probabiliste, échelle.
7. **Exemples de modèles** : TTM (Non-Transformer), TimeGPT (Encoder-Decoder), MOMENT/MOIRAI (Encoder-Only), Timer-XL/Time-MOE (Decoder-Only), Chronos/AutoTimes (Adaptation LLM).
8. **Limites** : scalabilité O(n²) self-attention, manque support probabiliste/multivarié dans certains, coûts computationnels élevés.
9. **Conclusion** : Transformers = paradigme dominant ; Foundation Models = scalabilité + généralisation ; perspectives : hybrides, interprétabilité, multi-modal.

## Difficultés rapportées

- Implémentation LSTM : articles sans code partagé → obligation de tout recoder.
- Temps d'entraînement : ~4h par époque → aucun résultat cohérent obtenu.

## Demandes à l'audience

Retour sur les difficultés d'implémentation et orientation pour la suite.
