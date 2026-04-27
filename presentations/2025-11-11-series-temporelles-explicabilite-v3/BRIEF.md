# BRIEF — Les Séries Temporelles : de la Prédiction à l'Explicabilité

> Rempli a posteriori (présentation déjà réalisée).

- **Date** : 2025-11-11
- **Audience** : Dr LACMOU ZEUTOUO Jerry, Dr DJOUMESSI Kerol, Pr KENGNE TCHENDJI Vianney, M. Yvan
- **Durée** : ~1h
- **Format** : Beamer (31 diapositives) — références bibliographiques incluses

## Objectif

Présentation consolidée et approfondie des séries temporelles, avec introduction complète des méthodes d'explicabilité. Premier vrai état de l'art XAI présenté aux encadreurs : méthodes modèle-agnostiques, spécifiques aux réseaux neuronaux, et spécialisées pour les séries temporelles. Première présentation avec des tableaux comparatifs.

## Points clés présentés

1. **Introduction — 3 définitions** : statistique (Box & Jenkins, 1970), informatique (Chatfield, 2003), conceptuelle. Caractéristique fondamentale : dépendance temporelle entre observations.
2. **Domaines d'application** : Cloud Computing (cold start FaaS), Finance (Bitcoin/USDT), Santé, Énergie, Réseaux, Transport — avec exemples visuels réels (AWS CloudWatch ALB, coingecko, charge électrique nationale).
3. **Structure et composantes** : Tendance, Saisonnalité, Bruit — illustrés sur le dataset passagers aériens USA (1994-1997) avec décomposition graphique.
4. **Enjeux communs** : anticiper le futur dans le cloud (cold start), la finance, l'énergie. "Prévoir = Mieux décider."
5. **Modélisation déterministe** : régression linéaire/polynomiale, décomposition de Fourier, modèles additifs/multiplicatifs, lissage exponentiel simple.
6. **Modélisation stochastique classique** : AR, MA, ARMA, ARIMA, SARIMA.
7. **Deep Learning temporel** : RNN/LSTM/GRU (mémoire longue terme), Transformers (attention mechanism, state-of-the-art prévision longue), modèles hybrides ARIMA+LSTM.
8. **Limite de la modélisation** : "Une prédiction sans explication est un chiffre sans conviction." LSTM et Transformers = boîtes noires.
9. **Méthodes modèle-agnostiques** : SHAP (Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016), Permutation Importance, Partial Dependence Plot (PDP).
10. **Méthodes spécifiques aux réseaux neuronaux** : Attention Mechanisms (Bahdanau 2015, Vaswani 2017), Saliency Maps (Simonyan 2013), Integrated Gradients (Sundararajan 2017).
11. **Méthodes spécialisées séries temporelles** : TimeSHAP (Bento et al., 2021), WindowSHAP (Nayebi et al., 2022), ShaTS (2025).
12. **Tableaux comparatifs** : SHAP vs LIME / TimeSHAP vs WindowSHAP / ShaTS — points forts, limites, type de modèle ciblé.
13. **Limites de l'explicabilité** : véracité des explications, stabilité, l'explicateur lui-même peut être une boîte noire.
14. **Perspectives & Travaux futurs** : étude approfondie de SARIMA, LSTM, Transformers ; ouverture des boîtes noires ; apprentissage par expérimentation sur datasets réels.

## Questions anticipées

*(Non préparées formellement — état de l'art encore en construction.)*

## Demandes à l'audience

Validation de la revue XAI présentée et orientation vers les prochaines étapes concrètes.
