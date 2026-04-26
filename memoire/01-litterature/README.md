# 01-litterature — État de l'art

## Structure

- `articles/` — PDFs des articles lus (nommer : `Année_PremierAuteur_MotsClés.pdf`)
- `fiches/` — fiches de lecture structurées (1 `.md` par article, même nom que le PDF)
- `synthese.md` — synthèse transversale (à créer quand ≥ 5 articles fichés)

## Workflow

1. Déposer le PDF dans `articles/`
2. Invoquer le skill **`fiche-lecture`** : il génère `fiches/<nom>.md` avec problème / méthode / résultats / limites / lien avec H1
3. Mettre à jour `synthese.md` quand un cluster thématique se dégage

## Axes de recherche prioritaires

1. Prédiction de charge FaaS (cold start, autoscaling)
2. Transformers pour séries temporelles
3. XAI : attention weights, faithfulness (comprehensiveness/sufficiency)
4. Clustering temporel (DBSCAN, k-shape)
