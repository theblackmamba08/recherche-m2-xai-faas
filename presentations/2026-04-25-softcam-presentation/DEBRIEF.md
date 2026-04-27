# DEBRIEF — Soft-CAM : Rendre les Modèles Boîtes Noires Auto-Explicables

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global

Présentation **non formellement reçue** — Dr LACMOU l'a jugée incomplète. La fiche SoftCAM seule ne suffit pas : il faut la situer dans le panorama des méthodes post-hoc et la relier explicitement à l'architecture de FAYAM.

## Remarques de Dr LACMOU

| # | Remarque | Suite à donner |
|---|----------|----------------|
| 1 | Présentation incomplète : manque les **méthodes post-hoc** (TimeSHAP, SHAP…) pour contextualiser SoftCAM | Ajouter une section de comparaison SoftCAM vs post-hoc dans les slides |
| 2 | Faire des **comparaisons** entre SoftCAM et les approches post-hoc | Tableau comparatif : fidélité, coût, stabilité, applicabilité au temporel |
| 3 | **Embrayer sur l'architecture de FAYAM** : dans quelle mesure peut-on améliorer l'explicabilité de son modèle en s'inspirant de SoftCAM ? | Cartographier le `TimeSeriesTransformer` HuggingFace, identifier la couche cible |
| 4 | **Récupérer le code de FAYAM**, entraîner le modèle sur le dataset, ressortir les prédictions et les métriques | Phase 1 du ROADMAP — priorité immédiate |

## Plan de l'étudiant pour la suite

1. **Dataset** : récupérer le dataset fourni par l'encadreur, l'agréger et réaliser une étude exploratoire (EDA).
2. **Entraînement** : configurer et entraîner le `TimeSeriesTransformer` de FAYAM sur ce dataset.
3. **Métriques** : ressortir les scores de prédiction (sMAPE, RMSE, R², Spearman — alignés sur FAYAM).
4. **Présentation à compléter** : ajouter comparaison post-hoc + lien FAYAM avant la prochaine séance.

## Action items

- [ ] **Compléter la présentation** : ajouter section comparaison SoftCAM vs TimeSHAP/SHAP, puis lien vers architecture FAYAM.
- [ ] **Récupérer le code FAYAM** (`TimeSeriesTransformer` HuggingFace) et le cartographier.
- [ ] **EDA du dataset** : agréger, visualiser, comprendre la structure des données.
- [ ] **Entraîner le modèle** et ressortir prédictions + métriques (sMAPE, RMSE, R²).
- [ ] **Identifier la couche cible** dans le Transformer de FAYAM pour l'adaptation SoftCAM.

## Prochaine présentation

Présentation complétée (SoftCAM + post-hoc + FAYAM) + premiers résultats d'entraînement sur le dataset.
