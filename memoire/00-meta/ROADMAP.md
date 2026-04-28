# Roadmap — < 3 mois jusqu'à soutenance

> Démarrage : 2026-04-26 — Pivot stratégique : 2026-04-27 — Soutenance cible : *(à fixer)*

> ⚠️ **Refonte du 2026-04-27** : après lecture de SoftCAM (Djoumessi & Berens, 2025), les hypothèses sont reformulées. La piste « attention weights » devient un repli, pas le plan principal. Voir [DECISIONS.md](DECISIONS.md) pour la justification.

## Phase 1 — Reproduction baseline (S1-S2)

- [ ] Récupérer le code FAYAM (Transformer uniquement, pas le CNN-LSTM)
- [ ] Identifier les hyperparamètres manquants (`predictionLength`, `contextLength`, `lagsSequence`, `cardinality`, `embeddingDimension` — non chiffrés p. 76 du mémoire FAYAM)
- [ ] Reproduire les résultats Transformer sur ≥ 3 datasets parmi les 18 (en utilisant les `HashFunction` du tableau IX p. 86)
- [ ] **Activer dès maintenant `output_attentions=True`** pour disposer du matériel H3 sans réentraîner
- [ ] Cartographier précisément l'architecture du `TimeSeriesTransformer` HuggingFace : identifier la **projection finale du décodeur** (cible de H1)
- [ ] Documenter les écarts éventuels dans `02-baseline/MEMOIRE.md`

## Phase 2 — H1 : SoftCAM-Transformer (S3-S6)

> **Objectif** : transposer le principe SoftCAM (couche d'evidence + ElasticNet) du CNN au TimeSeriesTransformer.

- [ ] Récupérer le code de référence SoftCAM (`https://anonymous.4open.science/r/SoftCAM-E1A3/`) — étudier la `class-evidence layer` et la perte ElasticNet
- [ ] Définir l'équivalent temporel d'une *class evidence map* (carte temps-passé × temps-futur ? par cluster DBSCAN ? par seuil de charge ?) → **point de design critique à débattre avec encadreurs**
- [ ] Implémenter `src/models/softcam_transformer.py` : variante du `TimeSeriesTransformer` avec couche d'évidence et perte ElasticNet
- [ ] Entraîner sur ≥ 3 datasets représentatifs (un par profil DBSCAN : peu fréquente / régulière / populaire)
- [ ] Évaluer **performance prédictive** (sMAPE, RMSE, R², Spearman — mêmes métriques que FAYAM) → la performance ne doit pas dégrader
- [ ] Évaluer **qualité d'explication** (faithfulness/comprehensiveness/sufficiency, activation precision/sensitivity adaptés au temporel)
- [ ] Comparer la carte d'évidence apprise à la matrice d'attention décodeur (sanity check, et préparation à H3 comme outil de validation)

### Critère de bascule vers H2
Si à la fin de S6 (≈ 2 semaines de prototypage) l'adaptation SoftCAM→Transformer ne converge pas ou dégrade fortement la précision : **basculer sur H2 sans attendre**.

## Phase 3 — H2 : SHAP-based (repli, S7-S8 si H1 bloque)

- [ ] Étudier **TimeSHAP** (Bento et al.) — application directe au Transformer FAYAM (post-hoc)
- [ ] Comparer KernelSHAP / TimeSHAP sur les 3 datasets
- [ ] Reprendre la grille d'évaluation faithfulness pour comparabilité H1 ↔ H2

## Phase 4 — H3 : étude des attention weights (dernier recours OU outil de validation)

- [ ] Extraire les attention maps (déjà disponibles si phase 1 a activé `output_attentions=True`)
- [ ] Clustering DBSCAN sur les charges → 3-5 macro-profils dérivés des 33 clusters FAYAM
- [ ] Visualiser attention moyenne par cluster
- [ ] Calculer comprehensiveness + sufficiency par profil
- [ ] **Si H1 ou H2 a abouti** : H3 sert à *valider* la cohérence entre carte d'évidence apprise et attention apprise (et n'est plus une contribution principale).

## Phase 5 — Rédaction (S9-S12)

- [ ] Plan détaillé du mémoire
- [ ] Chapitre 1 : intro + problématique
- [ ] Chapitre 2 : état de l'art (FAYAM + SoftCAM + TimeSHAP + littérature attention temps réel)
- [ ] Chapitre 3 : méthode (l'hypothèse retenue parmi H1/H2/H3, avec justification du choix au regard du calendrier)
- [ ] Chapitre 4 : résultats
- [ ] Chapitre 5 : discussion + limites
- [ ] Relecture, audit similarité, soumission

## Jalons encadreurs

- 2026-04-27 : pivot stratégique acté (passage à H1 = SoftCAM-Transformer).
- *(à compléter après chaque entretien)*

---

## État courant

> 📍 **Première chose à lire en début de session.** Mis à jour à chaque fin de session par le hook Stop.

### Dernière session : 2026-04-28 (session 12 — finalisée)

- **Phase actuelle** : présentation panorama XAI prête à l'oral — 47 slides + script complet.
- **Avancée récente** : `SPEECH.md` créé (~30 s/slide, ~20 min, tableau temps cibles par section, conseils de rythme).
- **Prochain pas** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer, métriques.

### Avant-dernière session : 2026-04-28 (sessions 10–11)

- **Phase actuelle** : présentation panorama XAI **100 % finalisée** — 47 slides, 760 Ko, 0 erreur.
- **Avancée récente** : style tcolorbox uniforme (42→47 slides) ; frame Références (10 entrées) ; slides de transition automatiques.
- **Prochain pas** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer, métriques.
