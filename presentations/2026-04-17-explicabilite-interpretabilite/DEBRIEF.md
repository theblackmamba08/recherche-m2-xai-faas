# DEBRIEF — Explicabilité et Interprétabilité en IA : Fondations, Méthodes et Enjeux Éthiques

> Rempli a posteriori (2026-04-27) à partir des notes de séance.

## Retour global des encadreurs

Séance en présentiel avec Pr KENGNE TCHENDJI et Dr LACMOU ZEUTOUO uniquement. La maîtrise du cadre XAI est jugée suffisante pour passer à la contribution. Dr LACMOU a orienté directement vers l'article SoftCAM de Dr DJOUMESSI comme fondation pour la contribution.

## Questions / remarques reçues

| # | Question / Remarque | Personne | Réponse donnée | Suite à donner |
|---|---------------------|----------|----------------|----------------|
| 1 | Lire l'article **SoftCAM** de Dr DJOUMESSI (déjà fiché dans `memoire/01-litterature/`) | Dr LACMOU ZEUTOUO | Noté | Lire en profondeur — architecture, couche d'évidence, régularisation ElasticNet |
| 2 | S'inspirer de SoftCAM pour **rendre l'architecture de FAYAM explicable** | Dr LACMOU ZEUTOUO | Noté | Identifier le point d'ancrage dans le `TimeSeriesTransformer` HuggingFace — projection finale du décodeur |

## Décision clé issue de cette séance

**Convergence vers H1** : Dr LACMOU a explicitement demandé d'adapter le principe SoftCAM au modèle Transformer de FAYAM. C'est la formalisation de l'hypothèse H1 — interprétabilité intrinsèque par modification architecturale légère, inspirée de SoftCAM (Djoumessi & Berens, 2025).

## Action items

- [ ] Lire **en profondeur** l'article SoftCAM (`memoire/01-litterature/articles/2025_Djoumessi_SoftCAM.pdf`) : comprendre la couche d'évidence, la régularisation ElasticNet, les métriques faithfulness utilisées.
- [ ] Identifier comment transposer le principe SoftCAM au `TimeSeriesTransformer` HuggingFace de FAYAM : quelle couche remplacer, comment produire une carte d'évidence temporelle.
- [ ] Récupérer le code FAYAM et cartographier l'architecture Transformer (phase 1 du ROADMAP).

## Prochaine présentation

Présentation de la compréhension approfondie de SoftCAM + proposition de design pour l'adaptation au Transformer de FAYAM.
