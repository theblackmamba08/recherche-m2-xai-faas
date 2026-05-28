# Questions ouvertes

> À traiter avec les encadreurs ou à creuser. Statut : 🔴 ouverte / 🟡 en cours / 🟢 résolue.

## Retours présentation 2026-05-25 — encadreurs (Dr LACMOU, Dr DJOUMESSI, Pr KENGNE)

- 🔴 **Renommer le modèle** : "SoftCAM-Transformer" crée de la confusion. Trouver un nom qui référence exactement ce qui est fait. Options : `TemporalEvidenceTransformer` (TET), `EvidenceLayerTransformer` (ELT), autre. À trancher avec encadreurs.
- 🔴 **Révision architecturale — dot product** : remplacer `h=(1-mix)·dec_output + mix·LN(bmm(M,enc_hidden))` par un dot product entre `dec_output` et M. Justifier depuis la littérature avant d'implémenter.
- 🔴 **Évaluation explicabilité quantitative ET qualitative** : H1.F/G insuffisants seuls. Ajouter activation precision/sensitivity, visualisations de M sur exemples concrets, interprétation expert métier.
- 🔴 **Mesure de confiance en M** : comment quantifier la fiabilité de M ? Pistes : calibration (ECE/MCE), stabilité inter-runs, bootstrap sur M.
- 🔴 **Seuil R² dans la littérature** : existe-t-il un seuil reconnu ("bon modèle" si R² > X) pour les séries temporelles ? À chercher.
- 🔴 **Schémas visuels réels** : intégrer systématiquement les figures issues des notebooks dans les présentations (pas uniquement TikZ schématiques).

## Retours présentation 2026-05-20 — questions soulevées (pré-présentation)

- 🟢 **Définir "CAM en séries temporelles"** : CAM original = régions spatiales sur image. Ici = instants passés, pas de superposition, régression pas classification. À expliciter dans le mémoire.
- 🔴 **Nom "SoftCAM-Transformer" bien choisi ?** : confirmé par les encadreurs le 2026-05-25 — renommage requis. Voir section ci-dessus.
- 🟢 **7 hypothèses pas assez visibles** : présentées seulement au slide 26, sans annonce en amont. Résolu à l'oral pour cette session — à anticiper pour la soutenance.

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

- 🟢 **[Résolu 2026-04-28]** Le code FAYAM est-il fourni avec les hyperparamètres du Transformer ? → Oui : `prediction_length=120`, `context_length=240`, `freq="1T"`, `encoder_layers=4`, `decoder_layers=4`, `d_model=32` retrouvés dans `tsf_transf.py`.
- 🟡 Le dataset Azure Functions 2019 : 4 clusters (0,4,6,8) disponibles en CSV (19 fonctions). Les 14 clusters restants restent à récupérer si nécessaire — non bloquant pour la Phase 1.
- 🔴 Code source SoftCAM (`anonymous.4open.science/r/SoftCAM-E1A3/`) accessible publiquement ? Licence ?

## Rédaction

- 🔴 Le template Dschang impose-t-il un style biblio (IEEE, APA) ?

## Calendrier

- 🔴 Quel point de bascule formel H1 → H2 ? (proposition : si pas de prototype convergent à fin S6, basculer ; à valider avec encadreurs)
