# 03-contribution — Apport original (H1, H2)

Hypothèses, design d'expériences, résultats **originaux** (au-delà de FAYAM).

## Hypothèses

### H1 — Attention + DBSCAN + Faithfulness *(prioritaire)*

> Les poids d'attention du Transformer FaaS, agrégés par cluster DBSCAN de profils de charge, révèlent des stratégies de prédiction interprétables et faithful.

**Mesures** :
- *Comprehensiveness* : drop de performance quand on retire les top-k tokens d'attention
- *Sufficiency* : performance résiduelle quand on garde uniquement les top-k tokens

### H2 — Integrated Gradients *(bonus)*

> IG via Captum produit des attributions cohérentes avec les attention weights.

## Contenu attendu

- `experiences/` — protocoles d'expérience (1 sous-dossier par expé)
- `resultats/` — sorties brutes (CSV/JSON), figures
- `analyses/` — notebooks d'analyse `.ipynb`
- `STEPS.md`, `MEMOIRE.md`

## Lien avec le code

Le code source vit dans [`../../code/`](../../code/). Ici on stocke uniquement les **artefacts** d'expériences et les analyses. Chaque expérience est tracée par le skill `experiment-tracker`.
