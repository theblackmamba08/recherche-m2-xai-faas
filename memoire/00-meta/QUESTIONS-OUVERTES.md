# Questions ouvertes

> À traiter avec les encadreurs ou à creuser. Statut : 🔴 ouverte / 🟡 en cours / 🟢 résolue.

## Méthodo — H1 (SoftCAM-Transformer)

- 🔴 **Cible architecturale exacte** : quelle couche du `TimeSeriesTransformer` HuggingFace remplacer par l'équivalent de la *class-evidence layer* ? (Hypothèse de travail : la projection finale du décodeur, mais à valider en lisant le code source HF.)
- 🔴 **Forme de l'evidence map en régression** : SoftCAM produit une carte par **classe** ; FAYAM fait de la **régression** (nombre d'invocations). Comment définir l'équivalent ? Trois pistes :
  - une carte par pas temporel futur (forme `[predictionLength × contextLength]`) ;
  - une carte par cluster DBSCAN (carte « profil de charge ») ;
  - une carte par seuil de charge (faible / moyen / fort).
- 🔴 **Régularisation ElasticNet** : valeurs initiales de `λ₁` / `λ₂` à tester ? SoftCAM utilise `λ₁ ∈ [10⁻⁶, 10⁻³]` selon dataset — ordre de grandeur pour temporel ?
- 🔴 **Métriques d'explication temporelles** : adapter *activation precision* / *activation sensitivity* aux séries temporelles (la « région annotée » devient quoi ?).
- 🔴 Faithfulness (comprehensiveness/sufficiency) : sur quels tokens (timesteps) l'appliquer ?

## Méthodo — H2 (TsSHAP / SHAPformer) si repli

- 🔴 **TsSHAP** : le surrogate XGBoost peut-il correctement imiter un Transformer sur les 18 fonctions FaaS (séries multivariées) ? Quelle fidélité obtenue ?
- 🔴 **TsSHAP** : univarié par construction — stratégie pour couvrir les 18 séries simultanément ?
- 🔴 **SHAPformer** : coût du réentraînement avec masked attention sur notre dataset FaaS ? Dépend du nombre de groupes N.
- 🔴 **SHAPformer** : code GitHub KIT compatible avec `TimeSeriesTransformer` HuggingFace ?

## Méthodo — H3 (attention) si dernier recours

- 🔴 Quel seuil DBSCAN (`eps`, `min_samples`) sur les charges FaaS ? Faut-il standardiser avant ?
- 🔴 Réutiliser tels quels les 33 clusters de FAYAM, ou en former de nouveaux ?

## Métriques transverses

- 🔴 Pour la performance prédictive, retenir lesquelles parmi sMAPE / RMSE / Normalized RMSE / R² / Spearman / Explained Variance ? Toutes pour comparabilité avec FAYAM, ou un sous-ensemble ?

## Données

- 🔴 Le dataset Azure Functions 2019 est-il intégralement accessible (cf. `HashFunction` du tableau IX FAYAM p. 86) ou faut-il en regénérer une partie ?
- 🔴 Le code FAYAM est-il fourni avec les hyperparamètres du Transformer (`predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension`) ? **Non chiffrés p. 76 du mémoire**.
- 🔴 Code source SoftCAM (`anonymous.4open.science/r/SoftCAM-E1A3/`) accessible publiquement ? Licence ?

## Rédaction

- 🔴 Le template Dschang impose-t-il un style biblio (IEEE, APA) ?

## Calendrier

- 🔴 Quel point de bascule formel H1 → H2 ? (proposition : si pas de prototype convergent à fin S6, basculer ; à valider avec encadreurs)
