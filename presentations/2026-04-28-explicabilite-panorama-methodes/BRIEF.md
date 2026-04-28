# BRIEF — Explicabilité : Panorama des méthodes XAI (LIME → SHAP → CAM → SoftCAM)

- **Date** : 2026-04-28 (mardi)
- **Audience** : Dr LACMOU ZEUTOUO Jerry (et Pr KENGNE si présent)
- **Durée estimée** : 35-45 min + questions
- **Format** : Beamer (35 diapositives) — thème Madrid, couleurs seahorse
- **Contexte** : suite directe de la présentation du 25/04/2026 (SoftCAM seul, jugée incomplète). Dr LACMOU avait demandé une présentation qui inclut les méthodes post-hoc de référence et montre pourquoi SoftCAM est préféré dans notre cadre.

## Objectif

Fournir un panorama complet des méthodes XAI pour les séries temporelles, en progressant de manière pédagogique de LIME (2016) jusqu'à SoftCAM (2025), en passant par toute la famille SHAP (KernelSHAP, TimeSHAP, WindowSHAP, TsSHAP, SHAPformer). Montrer que le choix de SoftCAM pour notre projet (H1) n'est pas arbitraire mais s'inscrit dans une progression logique, et que les méthodes SHAP constituent un repli robuste (H2).

## Plan de la présentation (7 sections)

1. **Accroche** (2 slides) : modèle boîte noire ≠ modèle de confiance ; XAI comme levier de confiance opérationnelle.
2. **Contexte FAYAM** (3 slides) : architecture Transformer HuggingFace, tâche FaaS, faille XAI identifiée.
3. **Taxonomie XAI** (3 slides) : ante-hoc vs post-hoc vs during-training ; local vs global ; agnostique vs spécifique.
4. **Famille SHAP** (12 slides) : LIME → KernelSHAP (Shapley) → TimeSHAP → WindowSHAP → TsSHAP → SHAPformer.
5. **Famille CAM** (5 slides) : CAM → GradCAM → SoftCAM (intrinsèque, ElasticNet).
6. **Tableau de synthèse** (2 slides) : comparaison sur 8 critères (tâche, architecture, coût, fidélité...).
7. **Retour à FAYAM** (4 slides) : grille de décision H1/H2, limites, prochaines étapes.

## Points clés à faire passer

- LIME = fondateur mais heuristique, pas adapté aux séries temporelles ni à la prévision.
- SHAP = solution théoriquement rigoureuse (valeurs de Shapley), mais conçue pour la classification ; seul TsSHAP cible nativement la prévision.
- SHAPformer = SHAP exact pour Transformer de prévision, 800× plus rapide (repli technique si H1 échoue).
- SoftCAM = seule approche intrinsèque applicable à un Transformer de prévision ; 1 forward pass ; préserve la performance.
- H1 (SoftCAM-Transformer) est le plan ; H2 (TsSHAP ou SHAPformer) est le repli robustement justifié.

## Métriques FAYAM à remplir

Le slide 2.3 contient des placeholders `\textit{(valeur)}` pour les métriques du Transformer FAYAM (sMAPE, RMSE, R², Spearman). À compléter après phase 1 (reproduction du code FAYAM).

## Statut

Présentation **compilée** (PDF 35 pages, 902 Ko). Prête pour la séance encadreur.
