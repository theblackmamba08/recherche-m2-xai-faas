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

### Dernière session : 2026-05-23 (session 86 — Q&A Run B collapse vulgarisé)

- **Phase actuelle** : Phase 2 — préparation orale, consolidation des explications pédagogiques.
- **Avancée** : Explication Run B collapse reformulée en français simple (analogie conduite/rétroviseur). Argument dissociation entraînement/inférence consolidé pédagogiquement.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle aux encadreurs.
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-23 (session 85 — Q&A perturbation + TFT)

- **Phase actuelle** : Phase 2 — consolidation des arguments XAI pour la soutenance.
- **Avancée** : LIME/SHAP/TimeSHAP expliqués (coût 2^n, fidélité approximée). TFT défini comme concurrent direct — argument : attention non régularisée vs M contrainte par ElasticNet.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle aux encadreurs.
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-23 (session 84 — Q&A méthodes post-hoc par gradient)

- **Phase actuelle** : Phase 2 — lecture littérature XAI en parallèle de la finalisation présentation.
- **Avancée** : Vanilla Gradients / SmoothGrad / Integrated Gradients / GradCAM expliqués. Lien GradCAM→SoftCAM consolidé. Argument fidélité par construction réaffirmé contre les gradients.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle aux encadreurs.
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-23 (session 83 — Q&A interprétabilité/explicabilité + correction slide 4)

- **Phase actuelle** : Phase 2 — finalisation présentation + lecture littérature.
- **Avancée** : Distinction interprétabilité/explicabilité clarifiée (Rudin). Slide 4 corrigé : "Débogage" → "Diagnostic".
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle aux encadreurs.
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-23 (session 82 — Q&A Rudin 2019)

- **Phase actuelle** : Phase 2 — lecture littérature en parallèle de la finalisation présentation.
- **Avancée** : Explication des 3 propriétés Rudin (simulable / features signifiantes / fidèle par construction). Lien avec H1 : M satisfait la fidélité par construction — argument défensif contre SHAP/LIME.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle aux encadreurs.
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-28 (session 85 — PLAN-MEMOIRE.md enrichi)

- **Phase actuelle** : Phase 5 (rédaction) — plan finalisé, prêt à soumettre aux encadreurs.
- **Avancée** : `PLAN-MEMOIRE.md` enrichi avec retours encadreurs : plus-value 3 axes, tableau comparatif XAI, pattern théorie→application obligatoire, §4.2.4 seuil R², §4.3.7 confiance en M, évaluation qualitative renforcée, checklist 8 points.
- **Prochain pas** :
  1. 🔴 Soumettre `PLAN-MEMOIRE.md` aux encadreurs pour validation.
  2. 🔴 Trancher : nom du modèle + dot product vs mélange.
  3. 🔴 Commencer rédaction Chap4 (résultats disponibles).
  4. 🟡 Exporter figures notebooks (§4.4 évaluation qualitative).
  5. 🟡 Documenter seuil R² dans la littérature (§4.2.4).

### Dernière session : 2026-05-28 (session 84 — Template Dschang + PLAN-MEMOIRE.md)

- **Phase actuelle** : Phase 5 (rédaction) amorcée — plan soumis aux encadreurs.
- **Avancée** : Template Dschang intégré dans `redaction/`. `PLAN-MEMOIRE.md` créé — 4 chapitres, contenu par section, références, volume estimé, ordre de rédaction.
- **Prochain pas** :
  1. 🔴 Soumettre `PLAN-MEMOIRE.md` aux encadreurs pour validation.
  2. 🔴 Trancher : nom du modèle + dot product vs mélange (décisions bloquantes pour Chap3).
  3. 🔴 Commencer la rédaction par Chap4 (résultats disponibles).
  4. 🟡 Exporter figures notebooks pour Chap4 §4.4 (évaluation qualitative).

### Dernière session : 2026-05-25 (session 83 — Q&A XAI fondamental + correction slide)

- **Phase actuelle** : Phase 2 — révision architecturale et évaluation explicabilité en attente.
- **Avancée** : Q&A approfondi sur les fondements XAI (Rudin, méthodes gradient/perturbation, TFT, Spearman, R², sparse/concentré). Slide corrigé : "Débogage" → "Diagnostic". Recompilation OK (43 pages, 0 erreur).
- **Prochain pas** :
  1. 🔴 Chercher la justification théorique du dot product dans la littérature (avant tout code).
  2. 🔴 Proposer 2-3 noms alternatifs à "SoftCAM-Transformer" aux encadreurs.
  3. 🔴 Compléter l'évaluation explicabilité : quantitatif (activation precision/sensitivity) + qualitatif (visualisations M).
  4. 🔴 Exporter figures notebooks pour les prochaines présentations.
  5. 🟡 Si dot product décevant → présenter résultats actuels avec cadre théorique renforcé.

### Dernière session : 2026-05-25 (session 82 — Débrief présentation encadreurs)

- **Phase actuelle** : Phase 2 — révision architecturale et évaluation explicabilité demandées.
- **Avancée** : Présentation tenue. DEBRIEF.md créé. Retours encadreurs documentés dans QUESTIONS-OUVERTES.md et DECISIONS.md.
- **Prochain pas** :
  1. 🔴 Chercher justification théorique du dot product dans la littérature (avant tout code).
  2. 🔴 Proposer 2-3 noms alternatifs à "SoftCAM-Transformer" aux encadreurs.
  3. 🔴 Compléter l'évaluation de l'explicabilité : quantitatif (activation precision/sensitivity) + qualitatif (visualisations M).
  4. 🔴 Exporter figures notebooks pour les prochaines présentations.
  5. 🟡 Si dot product décevant → présenter résultats actuels (R²=0.7563) avec cadre théorique renforcé.

### Dernière session : 2026-05-22 (session 81 — SPEECH.md complet)

- **Phase actuelle** : Phase 2 — présentation H1 prête, speech disponible.
- **Avancée** : `SPEECH.md` créé — script intégral 35 slides en français naturel, première personne. Présentation maintenant outillée pour la soutenance encadreurs.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🟡 Débriefer après la présentation réelle (questions encadreurs).
  4. 🟡 Trancher : garder "SoftCAM-Transformer" ou renommer "Temporal Evidence Map".

### Dernière session : 2026-05-21 (session 80 — Slide 35 restructuré + Q&A conceptuel)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Slide 35 restructuré en questions ciblées (feedback encadreurs + décisions ouvertes). Q&A : "idiosyncratique", "calibrated self-explainability" expliqués.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🔴 Finir slides de conclusion/bilan H1.A–H1.G (slides 32+).
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Dernière session : 2026-05-21 (session 79 — Diagramme TikZ architecture + slide loss + Q&A)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Slide 7 — architecture redessinée en TikZ (5 blocs, flèche enc_hidden, badge H1). Slide 8 — loss reformulée en français simple avec analogies. Slide 4 — 4 stakeholders complets. Q&A : Run A vs Run B expliqués, gap R²=0.37→0.53 justifié (C4 seul + HPO).
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🔴 Finir slides de conclusion/bilan H1.A–H1.G (slides 32+).
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Dernière session : 2026-05-21 (session 78 — Finitions slides + réécriture "positions et limites")

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Slide 30 simplifié en français courant. Slides 33/34/35 réécrits à la 1ère personne (ton humain). Slide 7 : figure architecture PDF incluse. Session Q&A amorcée.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal) dans le tableau slide 26.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🔴 Finir slides de conclusion/bilan H1.A–H1.G.
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Dernière session : 2026-05-21 (session 77 — Figures TikZ slides 30/31 H1.F/H1.G)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Slide 30 (H1.F) — courbe ΔMAE vs k (échelle log-manuelle), zone softcam!15, plafond warning 25%, bracket fayam "79% restant", max "+5.37%". Slide 31 (H1.G) — bar chart 4 valeurs (k=1/5/10/50) avec axe y zoomé 90–105% et marque brisée, ligne 100% success, message visuel "97% avec k=1 seul". 43 pages, 0 erreur.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans le tableau slide 26 + `H1-narration.md`.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🔴 Finir les slides de conclusion/bilan H1.A–H1.G (slides 32+).
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 76 — Figures TikZ slides 27/28/29 + discussion H1.E)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Bar chart H1.C (seuils vs résultats), slide H1.D simplifié (heatmaps schématiques retirées), scatter H1.E (ρ=-0.80, n=5). Limite n=5 identifiée : ρ non significatif statistiquement — H1.E à présenter comme tendance indicative appuyée par H1.F/G.
- **Prochain pas** :
  1. 🔴 Produire figures slides 30 (H1.F comprehensiveness) et 31 (H1.G sufficiency).
  2. 🔴 Reformuler H1.A (test diagonal `argmax(M[t]) ≈ heure(t)`) dans `slides.tex`.
  3. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 75 — Figures TikZ slides 22/24/25 + réforme hypothèses slide 26)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : 4 figures TikZ produites (slide 22 recalibré à scale=12.8/unité ; waterfall 4 paliers slide 24 ; bar chart FAYAM vs B5 + ligne TFT slide 25 ; tableau booktabs hypothèses slide 26). H1.A identifiée comme mal formulée — le test diagonal `argmax(M[t]) ≈ heure(t)` reste à faire.
- **Prochain pas** :
  1. 🔴 Reformuler H1.A (test diagonal) dans `slides.tex` et `H1-narration.md`.
  2. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  3. 🔴 Produire les figures restantes (slides 27+).
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 74 — Figures TikZ slides 12-13)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Figures TikZ produites pour slides 12 (schedules mix/γ avec vraies époques du notebook) et 13 (histogrammes Avant/Après LayerNorm montrant le collapse vs distribution de M).
- **Prochain pas** :
  1. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  2. 🔴 Produire les figures placeholders restantes (slides 14+).
  3. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 73 — Affinages slides 5-6 + explications pédagogiques)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Exemples enrichis sur slides 5 (panorama XAI) pour les 3 familles. Slide 6 (attention ≠ explication) testée en tableau/colonnes — rejetée pour erreurs Beamer, retour à la version bullet. Explication en français simple de "distribution adversariale" (Jain & Wallace 2019).
- **Prochain pas** :
  1. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  2. 🔴 Produire les ~10 figures placeholders.
  3. 🟡 Continuer la lecture/explication des slides suivants.
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 72 — Affinages slides 4-5 + explications pédagogiques)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Slide 4 affiné (Débogage plus concret, Auditeur retiré des Stakeholders). Explication pédagogique des 3 familles XAI temporelles (slide 5) : post-hoc gradient, post-hoc perturbation, intrinsèque — positionnement stratégique de H1.
- **Prochain pas** :
  1. 🔴 Confirmer le nombre de fonctions FAYAM → corriger slide 3.
  2. 🔴 Produire les ~10 figures placeholders.
  3. 🟡 Continuer la lecture/explication des slides suivants.
  4. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-21 (session 71 — Correction slide 3 + lecture slide 4)

- **Phase actuelle** : Phase 2 — présentation H1 en cours de finalisation.
- **Avancée** : Chiffre "~1900 fonctions Lambda" retiré de la slide 3 (non vérifiable dans nos fichiers). Explication du slide 4 ("Pourquoi expliquer ?") : 3 motivations universelles (Lipton 2018, Rudin 2019, EU AI Act 2024) + stakeholders FaaS.
- **Prochain pas** :
  1. 🔴 Confirmer le nombre exact de fonctions FAYAM auprès des encadreurs → compléter slide 3.
  2. 🔴 Produire les ~10 figures placeholders.
  3. 🟡 Fixer date de présentation avec les encadreurs.

### Session précédente : 2026-05-20 (session 70 — Contenu complet slides H1 + footer noir/blanc)

- **Phase actuelle** : Phase 2 — H1 verrouillée. Présentation encadreurs finalisée (contenu).
- **Avancée** :
  - `slides.tex` entièrement meublé (35 frames → contenu réel : textes, tableaux, boîtes tcolorbox, TikZ, citations). Compile : 44 pages, 967 Ko.
  - Footer ajusté : section centrale en noir sur fond blanc (`\setbeamercolor{title in head/foot}{fg=black,bg=white}`).
  - ⚠️ Erreur détectée slide 3 : "~1900 fonctions Lambda" non vérifiable dans nos fichiers — à confirmer dans le mémoire FAYAM avant présentation.
- **Prochain pas** :
  1. 🔴 Vérifier le nombre exact de fonctions dans le dataset FAYAM (mémoire FAYAM ou encadreurs) — corriger slide 3.
  2. 🔴 Produire les ~10 figures placeholders (waterfall MAE, bar charts H1.C/E/F/G, schémas TikZ run timeline).
  3. 🟡 Fixer date de présentation avec encadreurs.
  4. 🟢 Démarrer rédaction LaTeX chapitre H1 (`redaction/`).

### Session précédente : 2026-05-20 (session 69 — Squelette présentation H1 35 slides + 15 articles)

- **Phase actuelle** : Phase 2 — H1 verrouillée. Production des supports de présentation et de la bibliographie élargie.
- **Avancée** :
  - 15 nouveaux articles intégrés au corpus (`memoire/01-litterature/articles/`) avec fiches de lecture (`memoire/01-litterature/fiches/`) : controverse attention (Jain/Wiegreffe/Serrano), Vaswani Transformer, ERASER, Rudin, LayerNorm, ElasticNet, CAM/GradCAM, SHAP, IG, Bag of Tricks, Lipton, Jacovi & Goldberg. PDFs renommés selon nomenclature projet.
  - Plan détaillé de présentation à 35 slides élaboré (7 actes, ~40 min, citations partout).
  - `slides.tex` squelette complet posé : préambule hérité du panorama explicabilité + 35 frames vides avec titres et citations placeholder. Compile à 44 pages, 440 Ko.
  - Décision : meubler par Acte I d'abord (slides 2-8) pour valider style avant de propager.
- **Commit** : `2baf747` poussé sur main (42 fichiers, 1627 insertions).
- **Prochain pas** :
  1. 🔴 Meubler Acte I (slides 2-8) : contexte FaaS, motivation XAI, controverse attention, architecture, loss.
  2. 🟡 Puis Actes II→VII en propageant le style validé.
  3. 🟡 Produire les ~10 figures synthétiques (waterfall, bar charts, schémas TikZ).
  4. 🟢 Démarrer rédaction LaTeX chapitre H1 en parallèle de la finalisation slides.

### Session précédente : 2026-05-20 (session 68 — Revue critique H1.A→H1.G + brief encadreurs)

- **Phase actuelle** : Phase 2 — H1 verrouillée. Revue critique complète de toutes les hypothèses. Brief présentation encadreurs prêt.
- **Avancée** :
  - Revue au peigne fin de H1.G → H1.A dans l'ordre inverse, avec nuances méthodologiques :
    - H1.A : test diagonal (même heure de la journée) manquant — résultat 37% vs 25% indicatif seulement.
    - H1.B : reframé — divergence M/cross-attention attendue et souhaitable (comparer M à cross-attention est circulaire vu Jain & Wallace 2019). Mix d'inférence n'affecte pas M en teacher-forced.
    - H1.D : Pearson=0.992 partiellement confondant (paramètres partagés + inputs similaires par construction du cluster). Comparaison inter-clusters manquante.
    - H1.C : gain décomposé (+13.46 pp régularisation + 9.17 pp utilisation M).
    - H1.E : ρ=−0.80 clair, pas de nuance majeure.
    - H1.F/G : ✅ à mix=0.25, ceiling effect bien compris.
  - `H1-narration.md` mis à jour : grille reformatée (H1.F/G séparés), section "Nuances par hypothèse" ajoutée, section 7 (limites) étendue à 6 points.
  - Brief encadreurs créé : `presentations/2026-05-20-resultats-h1-softcam/BRIEF.md` (5 points clés, 5 questions anticipées, 3 demandes).
- **Bilan H1.A → H1.G final** :

  | H | Verdict | Note |
  |---|---------|------|
  | A | ✅ indicatif | test diagonal manquant — limite à mentionner |
  | B | ⚠️ reframé | divergence M/cross-att attendue (Jain & Wallace) |
  | C | ✅✅ | R²=0.7563, Spearman=0.9169 |
  | D | ✅ nuancé | Pearson=0.992 — confondant partiel avec setup |
  | E | ✅ | ρ R²↔entropy = -0.80 |
  | F | ✅ | +5.37% Δ MAE max à mix=0.25 |
  | G | ✅ | 97.13% préservation à k=1 |

- **Prochain pas** :
  1. 🔴 Démarrer rédaction LaTeX chapitre H1 (`redaction/`) sur la base de `H1-narration.md`.
  2. 🔴 Remplir `slides.tex` et fixer date avec encadreurs pour présentation H1.
  3. 🟡 Mettre à jour `JOURNAL.md` (entrée session 68).
  4. 🟢 (Optionnel) Tester H1.F sans renormalisation pour quantifier redondance temporelle.

### Session précédente : 2026-05-20 (session 66 — Notebook H1.F/G revisités à mix multiples)

- **Phase actuelle** : Phase 2 — H1. Validation rétrospective des hypothèses initiales en commençant par la fin (H1.G → H1.F → H1.E → H1.D → H1.C → H1.B → H1.A).
- **Avancée** :
  - Discussion sur H1.G d'abord : le verdict ⚠️ initial venait du ceiling effect à mix=0.05 (h_evidence pèse 5% → impact prédictif plafonné à ~5%). À mix=0.25 (config finale), le plafond passe à ~25% et le test devient informatif.
  - Notebook `softcam-cluster4-v3-h1fg-revisited.ipynb` créé (33 cellules) : 3 mix [0.05, 0.10, 0.25] × 7 k × 5 fonctions × 2 hypothèses = 210 forwards teacher-forced (~5 min Colab T4). Plafonds théoriques tracés en figure. Verdicts auto : H1.F PASS si Δ MAE max > 5%, H1.G PASS si préservation > 90% à k=10.
  - Note méthodologique : `predict_with_M_override` passe par `forward()` teacher-forced — Fix #5 ne joue pas (seul `model.evidence_mix` configure le régime).
  - Commit `e7c7813` + push.
- **Prochain pas** :
  1. 🔴 Colab T4 Run All sur `softcam-cluster4-v3-h1fg-revisited.ipynb` (~5 min).
  2. 🔴 Archiver HTML + JSON dans `code/experiments/runs/2026-05-20_softcam-cluster4-v3-h1fg-revisited/`.
  3. 🔴 Interpréter verdicts à mix=0.25 — mettre à jour `H1-narration.md` avec les nouveaux chiffres.
  4. 🟡 Enchaîner sur H1.E (entropy vs R²) — déjà validée ✅ mais à passer au peigne fin.

### Session précédente : 2026-05-20 (session 65 — Consolidation documentaire H1)

- **Phase actuelle** : Phase 2 — H1 verrouillée + capitalisée. Prête pour rédaction LaTeX.
- **Avancée** :
  - Discussion sur la self-explainability de la dissociation entraînement/inférence → défense en 4 points formalisée.
  - 4 écritures consolidatrices : `DECISIONS.md` (entrée 2026-05-20), `memory/project_h1_finalized_config.md` (nouvelle mémoire persistante), `memory/project_phase1_baseline_results.md` (chiffres corrigés post Fix #5), `memoire/03-contribution/notes/H1-narration.md` (squelette argumentatif chapitre H1).
  - Tableau de comparaison final consigné : FAYAM 0.3701 → Run A 0.5299 → B5+mix=0 0.6646 → **B5+mix=0.25 = 0.7563**. Gain décomposé : +13.46 pp régularisation + 9.17 pp utilisation effective de M.
- **Prochain pas** :
  1. 🟢 Récupérer `reeval_fix5.json` détaillé depuis Drive (per-function).
  2. 🔴 Démarrer rédaction LaTeX chapitre H1 dans `redaction/` sur la base de `H1-narration.md`.
  3. 🟡 Créer `run.md` consolidé dans `2026-05-20_softcam-cluster4-v3-reeval-fix5/`.
  4. 🟡 Mettre à jour `memoire/03-contribution/MEMOIRE.md` avec la configuration finale.

### Session précédente : 2026-05-20 (session 64 — Ré-évaluation exécutée : B5+mix=0.25 reste champion)

- **Phase actuelle** : Phase 2 — H1. Configuration finale verrouillée : **B5 + inférence mix=0.25 → R²=0.7563**. Pas de retrain B8.
- **Résultats Fix #5** (R² optimal par checkpoint) :

  | Modèle | Entraîné mix | Mix optimal | R² | Δ vs B5+0.25 |
  |--------|--------------|-------------|-----|--------------|
  | **B5** | 0.05 | **0.25** | **+0.7563** | — |
  | B6 | 0.10 | 0.50 | +0.5975 | −15.88 pp |
  | B7 | 0.15 | 0.50 | +0.4684 | −28.79 pp |

- **Trouvailles majeures** :
  - B7 récupéré : R² passé de −1.62 (buggé) à +0.4684 — `h_evidence` portait l'essentiel masqué par le bug.
  - Pattern : optimum d'inférence ≈ 5× mix d'entraînement.
  - Monter le mix d'entraînement dégrade monotone le R² pic → retrain à mix élevé non viable.
- **R² final SoftCAM** : 0.7563 vs FAYAM 0.37 → **+103 pp**. Contribution causale de M : +9 pp (mix=0 → mix=0.25 sur B5).
- **Prochain pas** :
  1. 🟢 Récupérer JSON détaillé `reeval_fix5.json` depuis Drive (per-function R²/Spearman).
  2. 🔴 Mettre à jour les mémoires persistantes : configuration finale H1 verrouillée.
  3. 🔴 Démarrer rédaction chapitre H1 : Méthode (SoftCAM-Transformer) + Résultats (H1.A/C/D/E + ré-évaluation) + Discussion (mix d'inférence ≠ mix d'entraînement, distinction structurelle/causale).
  4. 🟡 Créer `run.md` consolidé dans le dossier reeval-fix5.

### Session précédente : 2026-05-20 (session 63 — Notebook ré-évaluation B5/B6/B7 Fix #5)

- **Phase actuelle** : Phase 2 — H1. Décision retrain B8 mise en attente du diagnostic Fix #5 sur B6/B7.
- **Scrutation** :
  - B5 n'utilise PAS d'early stopping — le checkpoint est l'état brut après 51 epochs, pas une sélection par métrique fausse. Le bug `generate()` n'a corrompu que la métrique finale rapportée, pas le checkpoint.
  - R²=0.48 de B6 et R²=-1.62 de B7 mesuraient `dec_output` seul (mix_eff=0 à cause du bug). Diagnostic en creux : plus mix d'entraînement monte, plus `dec_output` s'effondre.
- **Avancée** : notebook `softcam-cluster4-v3-reeval-fix5.ipynb` créé (31 cellules) : 3 checkpoints × 7 mix = 21 évaluations, matrice + figures, décision automatique selon que B5+mix=0.25 (R²=0.7563) est battu. Sanity checks au démarrage (Fix #5 actif + existence des 3 checkpoints). Commit `f21d3a0` + push.
- **Prochain pas** :
  1. 🔴 Colab T4 Run All sur `softcam-cluster4-v3-reeval-fix5.ipynb` (~10 min, zéro réentraînement).
  2. 🔴 Archiver HTML + JSON + figures dans `code/experiments/runs/2026-05-20_softcam-cluster4-v3-reeval-fix5/`.
  3. 🔴 Décider B8 : si B6 ou B7 dépasse B5+mix=0.25 → fini. Sinon → fine-tune B8 mix=0.05→0.25.

### Session précédente : 2026-05-20 (session 62 — Ablation mix Fix #5 : SCÉNARIO A confirmé)

- **Phase actuelle** : Phase 2 — H1. Question causale tranchée. M contribue réellement.
- **Résultats** (checkpoint B5, `generate()` patché) :

  | mix | R² | Δ vs mix=0 |
  |-----|-----|------------|
  | 0.000 | 0.6646 | — |
  | 0.050 | 0.7130 | +4.8 pp ← entraîné |
  | 0.250 | 0.7563 | +9.2 pp ← pic |
  | 0.500 | 0.4367 | −22.8 pp |
  | 1.000 | −0.1332 | — |

- **Verdicts** :
  - **Scénario A confirmé** : M est causalement nécessaire (+4.84 pp R² quand Evidence Layer active).
  - Vrai R² de B5 = **0.7130** (pas 0.6628 — qui était mesuré avec le bug generate()).
  - Zone optimale non exploitée : pic à mix=0.25 (R²=0.7563) alors que modèle entraîné à mix=0.05.
  - H1.C se renforce : +93 pp au-dessus de FAYAM (0.37).
- **Prochain pas** :
  1. 🔴 Décider : utiliser mix=0.25 en inférence sur B5 (gratuit, +4.3 pp) OU retrain B8 avec mix cible=0.25 (plus cohérent) ?
  2. 🔴 Mettre à jour `JOURNAL.md` + mémoire persistante.
  3. 🟡 Créer `run.md` bilan dans le dossier mix-ablation.
  4. 🟢 Démarrer rédaction chapitre H1.

### Session précédente : 2026-05-19 (session 61 — Fix #5 : override `generate()` + push)

- **Phase actuelle** : Phase 2 — H1. Fix #5 déployé, en attente des vrais résultats de l'ablation mix.
- **Avancée** :
  - Override `generate()` implémenté dans `SoftCAMTransformerV3ForPrediction` : seul changement vs HF = `output_params(dec_last_hidden[:, -1:])` à la place de `parameter_projection(...)`. L'Evidence Layer est maintenant active à l'inférence.
  - Commit `3f0c51c` + push GitHub. Inclut le notebook `softcam-cluster4-v3-mix-ablation.ipynb`.
- **Prochain pas** :
  1. 🔴 Colab : Disconnect & delete runtime → T4 → `git pull` → Run All sur `softcam-cluster4-v3-mix-ablation.ipynb`.
  2. 🔴 Récupérer HTML + `mix_ablation.json` → archiver dans `code/experiments/runs/2026-05-19_softcam-cluster4-v3-mix-ablation/`.
  3. 🔴 Interpréter le vrai verdict A/B → adapter la narration du chapitre H1.

### Session précédente : 2026-05-19 (session 60 — Ablation mix exécutée → BUG CRITIQUE dans `generate()`)

- **Phase actuelle** : Phase 2 — H1. **Crise méthodologique** : la couche d'évidence est inactive à l'inférence sur toutes les runs antérieures.
- **Trouvaille** :
  - Mix sweep exécuté sur Colab T4. Résultat : R²=0.6646 et Spearman=0.9188 *identiques aux 9 valeurs* de mix ∈ [0.0 … 1.0], y compris mix=1.0.
  - Diagnostic : HuggingFace `generate()` (ligne 1679 de `modeling_time_series_transformer.py`) appelle `parameter_projection` directement, court-circuitant notre override `output_params`. **L'Evidence Layer n'est pas active à l'inférence.**
  - HTML archivé dans `code/experiments/runs/2026-05-19_softcam-cluster4-v3-mix-ablation/`.
- **Impact sur l'existant** :
  - H1.C (R²=0.6628, ρ=0.9222 de B5) n'évalue pas réellement la SoftCAM ; seulement un Transformer entraîné sous pression evidence dont l'inférence est baseline-équivalente.
  - Différences B5/B6/B7 reflètent les régimes d'entraînement, pas l'effet du mix à l'inférence.
  - H1.A/D/E restent valides (teacher-forced via `model.explain()`).
- **Prochain pas** :
  1. 🔴 Override `generate()` dans `SoftCAMTransformerV3ForPrediction` → copier la boucle HF + remplacer `parameter_projection` par `output_params`.
  2. 🔴 Re-charger checkpoint B5, lancer ré-évaluation avec le `generate()` patché (sanity check : R² doit rester ≥ 0.30 — sinon SoftCAM dégrade vraiment).
  3. 🔴 Relancer le notebook mix-ablation avec le fix → vrai verdict A/B.
  4. 🟡 Décider en fonction des résultats : poursuivre H1 tel quel, reformuler la contribution, ou pivoter.

### Session précédente : 2026-05-19 (session 59 — Notebook ablation `evidence_mix` créé)

- **Phase actuelle** : Phase 2 — H1. Question causale ouverte : M implique-t-elle la prédiction, ou est-elle un sous-produit de `dec_output` ?
- **Avancée** :
  - Discussion conceptuelle : H1.F/G inconclants à mix=0.05 ne distinguent pas Scénario A (M causale, plafond mécanique) de Scénario B (M cosmétique, prédiction passe par `dec_output`). Distinction posée : *interprétabilité structurelle* (M révèle des patterns) vs *interprétabilité causale* (M pilote la décision).
  - Décision : avant rédaction, faire l'ablation directe `model.evidence_mix=0.0` en inférence sur B5 — test gratuit, ~1h Colab, zéro réentraînement.
  - Notebook `softcam-cluster4-v3-mix-ablation.ipynb` créé (32 cellules) : sanity check + balayage 9 valeurs mix ∈ [0.0 … 1.0] + verdict automatique (seuil ΔR² 3pp / 5% relatif) + per-function + visualisations + prédictions `.npy` sauvegardées.
- **Prochain pas** :
  1. 🔴 Push GitHub → ouvrir `softcam-cluster4-v3-mix-ablation.ipynb` sur Colab T4 → Run All (~10-15 min).
  2. 🔴 Récupérer HTML + `mix_ablation.json` + figures → archiver dans `code/experiments/runs/2026-05-19_softcam-cluster4-v3-mix-ablation/`.
  3. 🔴 Interpréter le verdict empirique Scénario A vs B → ajuster la narration du chapitre H1 en conséquence.
  4. 🟢 Démarrer rédaction chapitre H1 sur la base du verdict.

### Session précédente : 2026-05-19 (session 58 — Résultats H1.A–H1.G analysés)

- **Phase actuelle** : Phase 2 — H1 complète. Toutes les hypothèses évaluées.
- **Verdicts** :
  - H1.A ✅ : argmax M dans pic 17-19h = 37% vs baseline 25% — M oriente vers le pic journalier.
  - H1.B ⚠️ : ρ moyen argmax(M) vs argmax(CA) = 0.16 — hétérogène (fn953 ρ=0.86, fn949 ρ=-0.65).
  - H1.D ✅✅ : Pearson hors-diag = 0.992 — M est cluster-level, argument le plus fort.
  - H1.E ✅ : Spearman R²↔entropy = -0.80 — M plus piquée pour meilleures prédictions.
  - H1.F/G ⚠️ : ceiling effect confirmé à mix=0.05 — impact prédictif non mesurable (~3% max).
- **Prochain pas** :
  1. 🔴 Créer `run.md` dans le dossier de run avec le bilan complet H1.A–H1.G.
  2. 🔴 Démarrer rédaction chapitre H1 (Méthode + Résultats + Discussion).
  3. 🟡 Mettre à jour `memoire/00-meta/JOURNAL.md`.

### Session précédente : 2026-05-19 (session 57 — HTML h1-analysis archivé)

- **Phase actuelle** : Phase 2 — H1. Résultats du notebook d'analyse disponibles (HTML), JSON manquant.
- **Avancée** : HTML `softcam-cluster4-v3-h1-analysis` archivé dans `code/experiments/runs/2026-05-19_softcam-cluster4-v3-h1-analysis/`.
- **Prochain pas** :
  1. 🔴 Récupérer `h1_analysis.json` depuis Drive → déposer dans `results/` → analyser les verdicts H1.A–H1.G.
  2. 🔴 Créer `run.md` dans le dossier de run avec les chiffres clés.
  3. 🟢 Démarrer rédaction chapitre H1.

### Session précédente : 2026-05-19 (session 56 — H1.F + H1.G formalisés, notebook d'analyse H1 prêt)

- **Phase actuelle** : Phase 2 — H1 VALIDÉ (H1.C). Outillage de vérification des hypothèses H1.A–H1.G prêt à tourner sur Colab.
- **Avancée** :
  - Discussion critique : H1.C seul ne rend pas le travail crédible. H1.A/B/D/E restaient ouverts ; H1.F (comprehensiveness) et H1.G (sufficiency) ont été ajoutés formellement aux hypothèses opératoires, avec note sur le ceiling effect à 5% (mix=0.05).
  - `softcam_transformer_v3.py` étendu : `_M_override` + `predict_with_M_override()` (forward teacher-forced avec M injectée) pour pouvoir tester H1.F / H1.G sans réentraîner.
  - Générateur `_generate_softcam_cluster4_v3_h1_analysis.py` → notebook `softcam-cluster4-v3-h1-analysis.ipynb` (38 cellules) : charge checkpoint B5 depuis Drive (pas de réentraînement), sanity check, extraction M en `.npy`, H1.A (argmax→heures), H1.B (M vs cross_attentions), H1.D (Pearson 5 fn), H1.E (entropy vs R²), H1.F/G (k-sweep mask/keep top-k), JSON synthèse.
- **Prochain pas** :
  1. 🔴 Push GitHub (v3 modifié + notebook H1 analyse + générateur).
  2. 🔴 Sur Colab : ouvrir `softcam-cluster4-v3-h1-analysis.ipynb` → T4 → Run All (~5-10 min, pas de réentraînement).
  3. 🔴 Télécharger HTML + `h1_analysis.json` + figures → interpréter manuellement les verdicts H1.A–H1.G.
  4. 🟢 Démarrer rédaction chapitre H1 sur la base des résultats.

### Session précédente : 2026-05-19 (session 55 — Runs B6 + B7 archivés + analyse définitive mix)

- **Phase actuelle** : Phase 2 — H1 VALIDÉ. Exploration mix terminée. B5 (mix=0.05) est l'optimum confirmé.
- **Résultats** :
  - B6 (mix=0.10) : R²=**0.4782**, Spearman=0.9171 → PASS mais −18 pp vs B5. 949/951 régressent fortement.
  - B7 (mix=0.15) : R²=**−1.6244**, Spearman=0.7685 → FAIL. Bascule non-linéaire entre 0.10 et 0.15.
  - Confirmation : maximiser mix n'est pas le bon objectif — M à mix=0.05 est déjà fidèle par construction.
  - M de B6/B7 plus diffuse (max_weight 0.092/0.089 vs 0.19 en B5) — le gradient aplatit M pour limiter son impact.
- **Prochain pas** :
  1. 🔴 Analyser cartes M de B5 (H1.A : profil argmax, lecture interprétative).
  2. 🔴 Démarrer rédaction chapitre H1 (Méthode + Résultats).
  3. 🟡 Tenter un autre cluster (C0 ou C8) si le temps le permet.

### Session précédente : 2026-05-18 (session 54 — Runs B6 + B7 générés — exploration mix)

- **Phase actuelle** : Phase 2 — H1 VALIDÉ. Exploration de mix=0.10 et 0.15 pour chercher un éventuel gain marginal.
- **Avancée** :
  - `_generate_softcam_cluster4_v3_runB6.py` (mix=0.10) et `_generate_softcam_cluster4_v3_runB7.py` (mix=0.15) créés à partir du template B5.
  - Seul `EVIDENCE_MIX_TARGET` change (0.05 → 0.10 / 0.15). Tous les autres réglages identiques (warm-up, anneal γ, LayerNorm v3, SEED=998, 51 epochs).
  - Notebooks générés : `softcam-cluster4-v3-runB6.ipynb` et `softcam-cluster4-v3-runB7.ipynb`.
- **Prochain pas** :
  1. 🔴 Push GitHub (nouveaux notebooks + générateurs).
  2. 🔴 Lancer B6 sur Colab T4 → comparer R² à B5 (0.6628).
  3. 🟡 Si B6 meilleur → lancer B7. Si B6 pire → B5 reste le best et on passe à l'analyse des cartes M.
  4. 🟢 Quand exploration terminée : analyser cartes M (H1.A), démarrer rédaction H1.

### Session précédente : 2026-05-18 (session 53 — **PASS H1.C** ✅ — H1 défendu)

- **Phase actuelle** : Phase 2 — **H1 VALIDÉ sur Cluster 4**. SoftCAM-Transformer fonctionne.
- **Avancée majeure** :
  - **Run B5 PASS H1.C** : R²=**0.6628**, Spearman=**0.9222**. Les deux gates dépassés (≥0.30 et ≥0.85).
  - R² > Run A (0.53) > FAYAM (0.37). Spearman = FAYAM.
  - Toutes les 5 fonctions ont R² positif et Spearman ≥ 0.89.
  - Recette gagnante : warm-up mix (→0.05) + anneal γ (→1e-3) + LayerNorm + mix très petit (0.05). Les 4 ingrédients sont nécessaires.
- **Prochain pas** :
  1. 🟢 Analyser cartes M (H1.A profil argmax, lecture interprétative).
  2. 🟢 Tenter Cluster 0/8 si possible (sinon documenter limitations).
  3. 🟢 Démarrer rédaction chapitre H1 (Méthode + Résultats).
  4. 🟢 Préparer présentation pour l'encadreur (annonce du PASS).

### Session précédente : 2026-05-18 (session 52 — Run B4 catastrophe, Run B5 prêt — dernier test)

- **Phase actuelle** : Phase 2 — H1. 5 configurations FAIL. Run B5 est le dernier test avant pivot H2.
- **Avancée** :
  - Run B4 (mix=0.10 sans schedules) : R²=**−3.58**, Spearman=**0.44**. PIRE que Run B3 (−1.59). Sans warm-up, le décodeur n'apprend jamais proprement.
  - **Diagnostic confirmé** : le warm-up est l'ingrédient critique. L'analyse linéaire que j'avais faite était incomplète (oubliait que dec_output évolue end-to-end avec h_evidence).
  - **Run B5 généré** : warm-up + anneal γ + LayerNorm + **mix=0.05** (plus petit que tous les runs). C'est la meilleure combinaison testable.
- **Prochain pas** :
  1. 🔴 Push GitHub (Run B5).
  2. 🔴 Colab : `softcam-cluster4-v3-runB5.ipynb` → Run All.
  3. 🟡 PASS H1.C → H1 défendable (mix=0.05 + carte M exacte).
  4. 🟡 FAIL → 5 configurations FAIL = preuve empirique d'incompatibilité architecturale. **Pivot H2 (TsSHAP / SHAPformer)** documenté comme contribution scientifique négative.

### Session précédente : 2026-05-18 (session 51 — Run B3 FAIL, découverte structurelle, Run B4 prêt)

- **Phase actuelle** : Phase 2 — H1 en cours. 4 variantes essayées, problème structurel identifié.
- **Avancée majeure** :
  - Run B3 (v3 LayerNorm, warm-up + anneal) : R²=−1.59, Spearman=0.78. Le LayerNorm a bien corrigé le collapse de M (max_weight 0.97 → 0.06), mais R² ne remonte presque pas.
  - **Analyse structurelle** : `parameter_projection(0.7·dec + 0.3·h_evidence)` est linéaire. h_evidence (single softmax) est structurellement plus pauvre que dec_output (cross-attention multi-tête multi-couches). À mix=0.3, on impose 30% de la prédiction à venir d'une branche faible → R² ne peut pas atteindre 0.30.
  - **Run B4 généré** : mix=0.10 constant, γ=0, modèle v3. Estimation R² ≈ 0.9×0.53 + 0.1×(-1) ≈ 0.38 → PASS attendu.
- **Prochain pas** :
  1. 🔴 Push GitHub (notebook + v3 + run.md) — IMPORTANT.
  2. 🔴 Sur Colab : `softcam-cluster4-v3-runB4.ipynb` → Disconnect → Run All (~15 min).
  3. 🟡 PASS H1.C → H1 défendable avec petit mix : *« mix=0.10 préserve la précision tout en fournissant une carte M algébriquement exacte »*.
  4. 🟡 FAIL → Run B5 (mix=0.05) ou Run B6 (architectural : M remplace la cross-attention du dernier décodeur).

### Session précédente : 2026-05-18 (session 50 — SoftCAM v3 + Run B3 prêt)

- **Phase actuelle** : Phase 2 — H1 en cours. Fix #4 implémenté dans `softcam_transformer_v3.py`, Run B3 prêt.
- **Avancée** : `SoftCAMTransformerV3ForPrediction` créé (hérite v2 + `evidence_norm = LayerNorm(d_model)` sur `h_evidence` avant mix). Notebook Run B3 généré (36 cellules, commit `96b62a0`). Schedules Run B2 conservés.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → `code/notebooks/softcam-cluster4-v3-runB3.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~15-20 min sur T4).
  3. 🟡 PASS H1.C (R²≥0.30, Spearman≥0.85) → analyser cartes M, rédiger contribution.
  4. 🟡 FAIL → pivot H2 (TsSHAP/SHAPformer).

### Session précédente : 2026-05-18 (session 49 — Run B2 archivé, Fix #4 planifié)

- **Phase actuelle** : Phase 2 — H1 en cours. Run B et Run B2 FAIL, Fix #4 architectural identifié.
- **Avancée** :
  - Run B (mix=0.3 constant, γ=1e-3 constant) : R²=−2.83, Spearman=0.33. *Attention collapse* : M quasi-Dirac sur s≈211, max_weight=0.85.
  - Run B2 (warm-up mix + annealing γ) : R²=−1.97, Spearman=0.80 (progrès +85/+47 pp). M encore effondrée (s≈25, max_weight=0.97) mais Spearman proche du gate 0.85.
  - Cause persistante : `h_evidence = bmm(M, enc_hidden)` n'est pas dans le même espace statistique que `dec_output` → la tête `parameter_projection` produit une mauvaise échelle → R² négatif mais Spearman OK.
  - Résultats archivés : HTML + test_metrics.json + run.md dans `code/experiments/runs/2026-05-18_softcam-cluster4-v2-runB2/`.
- **Prochain pas** :
  1. 🔴 **Fix #4** : ajouter `nn.LayerNorm(d_model)` sur `h_evidence` avant le mix dans `softcam_transformer_v2.py` (2 lignes : `__init__` + `output_params`).
  2. 🔴 Générer Run B3 (même schedules Run B2 + LayerNorm) + lancer sur Colab T4.
  3. 🟡 Si PASS H1.C → analyser cartes M.
  4. 🟡 Si FAIL → pivot H2 (TsSHAP/SHAPformer).

### Session précédente : 2026-05-18 (session 47 — audit Run B + fix docs entropie)

- **Phase actuelle** : Phase 2 — Run B audité au peigne fin, prêt à lancer sur Colab.
- **Audit OK** : tous les fixes de Run A présents (re-seed RNG, 4+4 layers, seed=998, split [:-120], pas de val_loader). Shapes `model.explain()` et batch mismatch enc/dec vérifiés.
- **Bugs de doc corrigés** : cellule 9 et plot cellule 10 affichaient `+ elastic − entropy → max entropie → sparsité`. Or le modèle calcule `forecast + elastic + entropy` (somme) — on **minimise** l'entropie pour obtenir des M piqués. Commit `08be6b3` poussé.
- **Caveat noté** : fenêtre de test = 22h-24h (jour 14), pas 17h-19h. H1.A teste où M se concentre pour cette fenêtre, pas le pic journalier en direct.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → ouvrir `code/notebooks/softcam-cluster4-v2-runB.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~15-20 min sur T4).
  3. 🟡 PASS H1.C (R²≥0.30 ET Spearman≥0.85) → analyser cartes M (H1.A/H1.D).
  4. 🟡 FAIL H1.C → essayer mix=0.1 (Run C) avant pivot H2.

### Session précédente : 2026-05-18 (session 46 — Run A PASS + Run B généré)

- Run A fix5 validé : R²=0.5299, Spearman=0.9176.
- Run B notebook généré (36 cellules, commit `62e38ed`).

### Session précédente : 2026-05-18 (session 44 — Run A strict baseline, val_loader retiré)

- **Phase actuelle** : Phase 2 — Run A reconstruit en strict baseline FAYAM, prêt pour 4e exécution.
- **Avancée** :
  - Retrait complet du val_loader (val_rows, val_dataset, val_loader, monitoring val_r2/val_spear, plot val) dans `_generate_softcam_cluster4_v2_runA.py`. Élimine la consommation de RNG par `.generate()` à chaque epoch.
  - Notebook régénéré et poussé sur GitHub (commit `0b49075` sur main).
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → main** → ouvrir `code/notebooks/softcam-cluster4-v2-runA.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All** (~8-10 min sur T4).
  3. 🟡 Si PASS (R²≈0.37, Spearman≈0.92) → pipeline saine, on lance Run B (`use_evidence_layer=True, mix=0.3`).
  4. 🟡 Si encore FAIL → investigation plus poussée (lags, scaler, time features).

### Session précédente : 2026-05-18 (session 43 — Run A 3e exécution + diagnostic RNG drift)

- **Phase actuelle** : Phase 2 — sanity check Run A toujours FAIL, cause RNG identifiée.
- **Résultats Run A (4+4, seed=998, train split corrigé)** :
  - R² = 0.0529 (FAYAM 0.3701, -31.72 pp)
  - Spearman = 0.9052 (FAYAM 0.9201, -1.49 pp) → ordre quasi correct, magnitude sous-estimée
  - Forte amélioration vs runs précédents (R² évolue de -0.19 → -0.46 → 0.05)
- **Cause identifiée** : le baseline FAYAM n'a PAS de val_loader. Run A appelle `.generate()` (sampling stochastique 100 trajectoires) sur val à chaque epoch → consomme RNG → diverge la dynamique d'entraînement. Code v2 avec `use_evidence_layer=False` est mathématiquement identique au baseline (vérifié ligne 269 de `softcam_transformer_v2.py`).
- **Prochain pas** :
  1. 🔴 Retirer val_loader + val_dataset du notebook Run A (strict baseline).
  2. 🔴 Régénérer notebook, push GitHub, recharger sur Colab, Run All.
  3. 🟡 PASS attendu si l'hypothèse RNG est correcte.

### Session précédente : 2026-05-18 (session 42 — fix train split, notebook 100 % FAYAM)

- **Phase actuelle** : Phase 2 — sanity check Run A, 3e correctif appliqué.
- **Avancée** :
  - Run v2.1 (4+4, seed=998) analysé : R²=-0.4604, Spearman=0.8698. Dernier bug identifié : train split `target[:-240]` au lieu de `target[:-120]` (FAYAM). On entraînait sur 120 points de moins.
  - Fix : ligne 217 du générateur corrigée → `target[:-PREDICTION_LENGTH]`. Notebook régénéré, commit `977073f`.
  - Tous les écarts vs FAYAM sont maintenant corrigés : architecture (4+4), seed (998), train split (target[:-120]).
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → branche main** → `code/notebooks/softcam-cluster4-v2-runA.ipynb`.
  2. 🔴 **Runtime → Disconnect and delete runtime** → **Run All**.
  3. 🟡 PASS attendu : R²≈0.37, Spearman≈0.92. Si PASS → Run B (`use_evidence_layer=True, mix=0.3`).

### Session précédente : 2026-05-18 (session 41 — archivage HTML Run A, Colab cache détecté)

- **Phase actuelle** : Phase 2 — sanity check Run A pas encore validée.
- **Avancée** :
  - HTML du run Colab archivé dans `code/experiments/runs/2026-05-17_softcam-cluster4-v2-runA/`.
  - Diagnostic : Colab a exécuté l'ANCIEN notebook (cache) — `ENCODER_LAYERS=2` visible dans le HTML. Résultats inchangés : R²=-0.1861, Spearman=0.9190.
- **Prochain pas** :
  1. 🔴 Sur Colab : **File → Open → GitHub → branche main** → ouvrir `code/notebooks/softcam-cluster4-v2-runA.ipynb` (version corrigée 4+4, seed=998).
  2. 🔴 **Runtime → Disconnect and delete runtime** puis **Run All**.
  3. 🟡 Vérifier dans la cellule config que `ENCODER_LAYERS = 4` avant de lancer.

### Session précédente : 2026-05-18 (session 40 — comparaison FAYAM + seed fix)

- **Phase actuelle** : Phase 2 — sanity check Run A prêt, notebook entièrement aligné sur FAYAM.
- **Avancée** :
  - Comparaison FAYAM Table VII finalisée : R²=0.3701 / Spearman=0.92 sur C4 sont cohérents avec la fourchette FAYAM (-0.164 → 0.958). Notre Spearman=0.92 dépasse 5/6 datasets FAYAM.
  - Dernier écart corrigé : `SEED = 2026` → `SEED = 998` dans `_generate_softcam_cluster4_v2_runA.py`. Notebook régénéré, commit `459730b`. Le notebook est maintenant 100 % aligné avec le protocole FAYAM (architecture 4+4, seed 998).
- **Prochain pas** :
  1. 🔴 Upload `softcam-cluster4-v2-runA.ipynb` sur Colab (File → Open → GitHub → branch main).
  2. 🔴 T4 GPU → Run All (~10-15 min) → vérifier PASS : R² ≥ 0.30, Spearman ≥ 0.85.
  3. 🟡 Si PASS → sanity check validée, pipeline SoftCAMV2 saine → lancer Run B (`use_evidence_layer=True`, `mix=0.3`).
  4. 🟡 Si FAIL → dernier verrou = train split (target[:-240] vs target[:-120]) à investiguer.

### Session précédente : 2026-05-17 (session 36 — H1 v2 code)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer **v2** (diagnostic-friendly).
- **Avancée** :
  - Diagnostic du NO-GO v1 : hypothèse principale = **information bottleneck** (v1 remplace `dec_output` par `bmm(M, enc_hidden)`, bypass tout le travail du décodeur ; `parameter_projection` reçoit des stats très différentes de ce pour quoi il était initialisé).
  - Code v2 écrit dans `code/src/models/softcam_transformer_v2.py` (à part, v1 intact). Ajoute `use_evidence_layer: bool` (toggle parent FAYAM strict) et `evidence_mix: float ∈ [0,1]` (interpolation `h = (1-mix)·dec_output + mix·bmm(M,enc)`).
- **Prochain pas** : décider entre 4 runs A/B chaînés dans un notebook ou Run A seul (sanity check parent FAYAM avec proto exact 51 epochs full, pas d'early stop).

### Session précédente : 2026-05-17 (session 35 — premier run H1)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. **GATE H1.C échoué**, investigation requise avant pivot.
- **Résultats** :
  - Test R² = **-6.1565** (FAYAM 0.37 → -652 pp)
  - Test Spearman = **-0.8731** (FAYAM 0.92 → -179 pp)
  - **Anti-corrélation systématique** (per-series -0.85 à -0.90) → bug architectural, pas un problème de convergence
  - best val R² = 0.0837 (epoch 8), early stop epoch 18, training 5.5 min
- **Archive** : `code/experiments/runs/2026-05-17_04-52_softcam-cluster4-h1-v1/` (HTML + run.md, gitignored)
- **Prochain pas** :
  1. 🔴 **Sanity check forward parent FAYAM** sur Cluster 4 sans evidence layer — si lui converge, bug 100% dans notre code H1
  2. 🔴 **Inspection visuelle de M** (heatmaps `.npy` sur Drive) — détecter softmax dégénéré
  3. 🔴 **Test unitaire fin** : `model.explain()` == forward standard ? Signes corrects ? `encoder_last_hidden_state` capturé au bon moment par le hook lors de `model.generate()` ?
  4. 🟡 Si bug trouvé après 1-2 jours → H1 v2. Sinon → **pivot H2 (TimeSHAP)**.

### Session précédente : 2026-05-17 (session 34 — fin)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook définitivement prêt.
- **Avancée** : Fix final cellule clone — `get_ipython().system()` remplace `subprocess.run()`. Colab peut maintenant cloner sans problème de TTY/credentials (commit `83a843e`).
- **Prochain pas** : Coller le code corrigé dans la cellule Colab → Run → vérifier GATE H1.C (R² ≥ 0.30, Spearman ≥ 0.85).

### Session précédente : 2026-05-17 (session 34 — suite)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook définitivement prêt pour Colab.
- **Avancée** : Double fix cellule clone — `os.system()` → `subprocess.run()` (session 34), puis suppression de `capture_output=True` (session 34 suite) qui causait `fatal: could not read Username` même sur repo public.
- **Prochain pas** : relancer le notebook sur Colab T4 (Runtime → Disconnect and delete runtime → re-upload → Run All) → vérifier GATE H1.C.

### Session précédente : 2026-05-17 (session 34)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Notebook prêt à tourner, bug Colab corrigé.
- **Avancée récente** :
  - Fix cellule git clone de `softcam-cluster4.ipynb` : `os.system()` → `subprocess.run()` avec capture et vérification du returncode (commit `7aeef09`).
  - Notebook poussé sur GitHub — prêt pour Colab T4.
- **Prochain pas** :
  1. 🔴 Sur Colab : Runtime → Disconnect and delete runtime → re-upload notebook → T4 GPU → Run All
  2. 🔴 Vérifier le **GATE H1.C** : R² ≥ 0.30 et Spearman ≥ 0.85 sur le test set
  3. 🟡 Si GO → analyser heatmaps M + comparer à `cross_attentions` (gratuit)
  4. 🟡 Si NO-GO → pivot immédiat vers H2 (TimeSHAP)

### Session précédente : 2026-05-16 (session 33)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. **Code implémenté**, prêt à tourner sur Colab T4.
- **Avancée récente** :
  - **Compréhension complète de l'architecture** validée par dialogue pédagogique (encoder + decoder + evidence layer). Architecture désormais maîtrisée pour la soutenance.
  - **Code H1 produit** :
    - `code/src/models/softcam_transformer.py` (~360 l.) — `SoftCAMTransformerConfig`, `SoftCAMTSPredictionOutput`, `SoftCAMTransformerForPrediction` (sous-classement HF, hook encodeur, override `output_params` → forward ET generate fonctionnent).
    - `code/tests/test_softcam_transformer.py` (10 tests pytest).
    - `code/notebooks/softcam-cluster4.ipynb` (33 cellules, généré par `_generate_softcam_cluster4.py`) — clone repo + GATE H1.C explicite + extraction & heatmaps des cartes M.
  - **Caveat L1 documenté** : sous softmax `mean(|M|) = 1/ctx` est constant (gradient nul) ; le vrai sparsity-inducing est l'entropie `γ·H(M)` (hyperparam exposé).
  - **Mémoires persistantes sauvegardées** (session 32 → 33) : argumentation H1, controverse Jain & Wallace, TFT concurrent, plan validation.
- **Prochain pas** :
  1. 🔴 Commit + push GitHub (`code/src/models/`, tests, notebook + générateur, doc updates)
  2. 🔴 Upload `softcam-cluster4.ipynb` sur Colab T4 → Runtime → Run All
  3. 🔴 Vérifier le **GATE H1.C** : R² ≥ 0.30 et Spearman ≥ 0.85 sur le test set
  4. 🟡 Si GO → analyser heatmaps M + comparer à `cross_attentions` (gratuit)
  5. 🟡 Si NO-GO → pivot immédiat vers H2 (TimeSHAP)

### Session précédente : 2026-05-16 (session 32)

- **Phase actuelle** : Phase 2 — H1 SoftCAM-Transformer. Fiche TFT rédigée. Priorité 🔴 close.
- **Avancée récente** :
  - **Fiche TFT créée** : `memoire/01-litterature/fiches/2019_Lim_TFT.md` — architecture complète (VSN, GRN, IMHA), résultats 9 datasets, critique Jain & Wallace 2019, tableau comparatif TFT vs H1.
  - **Argument H1 consolidé** : M[t,s] = décomposition linéaire exacte (fidèle par construction) vs IMHA TFT = poids d'attention non fidèles.
  - **Phrase-clé soutenance** prête : "...c'est une décomposition linéaire prouvée, pas une attribution approximée."
- **Prochain pas** :
  1. 🟡 **Ajouter TFT au panorama XAI v3** (canevas Dr LACMOU 4 points + plus-value de transition)
  2. ✅ **Implémenter `evidence_layer`** dans `src/models/softcam_transformer.py` (fait session 33)

### Session précédente : 2026-05-14 (session 30 — clôture Phase 1 bis)

- **Phase actuelle** : **Phase 1 bis CLOSE.** HPO Cluster 4 exécutée et archivée. Décision baseline actée. En route vers Phase 2 (H1 SoftCAM-Transformer) + lecture TFT.
- **Avancée récente** :
  - Run `optimized-cluster4.ipynb` exécuté sur Colab T4, résultats rapatriés et archivés dans [`code/experiments/runs/2026-05-14_optimized-cluster4/`](../../code/experiments/runs/2026-05-14_optimized-cluster4/).
  - **HPO** : 15 trials, 4 complétés (73% pruned dont OOM). Best val R²=0.5347 avec `d_model=128, context=240, encoder_layers=4, lr=6.41e-4`.
  - **Test** : R²=**-1.39** (vs FAYAM=0.37) → dégradation -1.76pp. Cause : `context=240` (OOM-contraint) perd la périodicité 24h.
  - **Décision** : **FAYAM original conservé comme baseline** (seuil +20pp non atteint).
  - **Insight majeur** : `d_model=32` de FAYAM justifié empiriquement — seul `d_model` compatible avec `context=1440` sans OOM T4. Réponse à la question encadreur.
  - **Ablation 949** : dédié R²=0.215 > multi-opt R²=-1.257 → "écrasée par multi-task" (contamination context=240 à noter).
- **Prochain pas** (par ordre de priorité) :
  1. 🔴 **Lire le papier TFT** (Lim et al. 2019, arxiv:1912.09363) et créer `memoire/01-litterature/fiches/2019_Lim_TFT.md`
  2. 🟡 Re-positionner la motivation H1 dans `memoire/03-contribution/MEMOIRE.md` (alternative à TFT, fidélité par construction vs attribution attention)
  3. 🟡 Ajouter TFT au panorama XAI v3 avec canevas Dr LACMOU (contexte/problèmes/transposabilité/limites)
  4. 🟢 Démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) — cartographier `parameter_projection` HuggingFace (cible de l'evidence layer SoftCAM)

### Session précédente : 2026-05-09 (session 28)

- **Phase actuelle** : préparation finale de la séance Dr LACMOU (panorama XAI v2 + SPEECH section architecture).
- **Avancée récente** :
  - `SPEECH-architecture.md` créé pour la section 10 (Architecture SoftCAM-Transformer) : 10 slides commentées avec texte à dire, transitions, anticipation de 5 questions probables, notes Cabrel.
- **Prochain pas** : Cabrel relit le speech avant la séance ; en parallèle, démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) (cartographier `parameter_projection` HuggingFace).

### Session précédente : 2026-05-09 (session 27)

- **Phase actuelle** : **Phase 1 terminée**, transition vers Phase 2. Présentation panorama XAI v2 prête.
- **Avancée récente** :
  - **Présentation Panorama XAI v2** construite et compilée : `presentations/2026-05-09-panorama-xai-v2/` (64 pages, 908 Ko). Canevas Dr LACMOU appliqué uniformément (5 environnements TeX dédiés : `ctxbox/probbox/transposbox/limitbox/plusbox`).
  - **Zooms approfondis** : 7 slides SHAPformer (architecture training masking + inférence 2^N + résultats + position semi-intrinsèque + discussion *« peut-on rendre 1-pass ? »*) et 6 slides SoftCAM (architecture intrinsèque + ElasticNet + résultats CNN + perspective ViT/Transformer).
  - **Section Architecture SoftCAM-Transformer (10 slides)** : motivation, schéma TikZ complet, entrées/sorties par bloc, code PyTorch evidence layer, équation centrale, ElasticNet visualisé, 3 variantes A/B/C, exemple carte d'évidence sur C4, hypothèses H1.A--H1.E.
  - SVG architecture créé : `memoire/03-contribution/figures/softcam-transformer-architecture.svg` (réutilisable mémoire + Beamer).
  - Discussion conceptuelle actée : Variante B (carte × embeddings encodeur) retenue pour le design SoftCAM-Transformer ; idée *« SHAPformer self-explainable »* reléguée au chapitre Discussion (dénature SHAPformer si poussée plus loin).
- **Prochain pas** : démarrer **Phase 2** — J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) : cartographier `parameter_projection` HuggingFace pour localiser la cible exacte de l'evidence layer.

### Session précédente : 2026-05-08 (session 26)

- **Phase actuelle** : **Phase 1 terminée**, transition vers Phase 2. Retour panorama XAI traité.
- **Avancée récente** :
  - DEBRIEF de la présentation panorama XAI du 28/04/2026 rédigé. Retour Dr LACMOU : présentation trop technique, étudiant en peine à l'oral.
  - **Cadre prescrit pour toute méthode XAI** : 4 points (contexte / problèmes résolus / transposabilité / limites) + plus-value de transition. Acté dans [DECISIONS.md](DECISIONS.md) (entrée 2026-05-08) et mémoire persistante.
- **Prochain pas** : (1) reprendre les slides du panorama selon le cadre Dr LACMOU avant la prochaine séance ; (2) en parallèle, démarrer **Phase 2 — H1** : J1 du [`PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md) (architecture `TimeSeriesTransformer` HuggingFace).

### Session précédente : 2026-05-05 (session 25)

- **Phase actuelle** : **Phase 1 terminée** — 4 runs dédiés archivés, cible H1 actée.
- **Avancée récente** :
  - Synthèse comparative des 4 baselines dédiés (C0/C4/C6/C8). **Seul C4 apprend** : R²=0.37, Spearman=0.92, sMAPE=0.22.
  - C0 = échec inattendu (R²≈0, modèle prédit la moyenne — probable problème scaler interne sur magnitude ~120 000). C6/C8 = trivial predictors (zero-inflated).
  - **Cible primaire H1 = Cluster 4** actée dans [DECISIONS.md](DECISIONS.md). Hypothèses opératoires H1.A–H1.E fixées (carte d'évidence sur heures du profil journalier, attention sur lags 1440/2880, non-dégradation précision, cohérence intra-cluster, test best/worst sur fn 953/949).
  - C0 archivé comme cas-limite pour chapitre Discussion (diagnostic scaler à creuser).
- **Prochain pas** : démarrer **Phase 2 — H1** : étude architecture `TimeSeriesTransformer` HuggingFace (J1 du `PLAN-ETUDE-ARCHITECTURE.md`) pour localiser la projection finale du décodeur (cible SoftCAM).

### Session précédente : 2026-05-05 (session 24)

- **Phase actuelle** : **Phase 1** — run C8 archivé, C0 à lancer.
- **Avancée récente** :
  - Run `baseline-cluster8` exécuté et archivé (MASE=0.44, sMAPE≈2.0, R²=-0.79). Modèle dédié = même résultat que mixte → C8 intrinsèquement difficile (zero-inflated).
  - Conclusion (depuis invalidée — voir session 25) : *"C0 est la seule cible viable pour H1"*.
- **Prochain pas** : uploader `baseline-cluster0.ipynb` sur Colab T4 → Run All.

### Session précédente : 2026-05-05 (session 23)

- **Phase actuelle** : **Phase 1** — runs par cluster engagés (C0 notebook créé, C6 archivé).
- **Avancée récente** :
  - Notebook `baseline-cluster0.ipynb` créé : modèle dédié C0, plots par fonction (zoom 6 h + vue 24 h + comparaison 3 fonctions), attention par fonction.
  - Dossier run `2026-05-05_baseline-cluster6/` archivé (zip Drive + HTML). Diagnostic C6 : zero-inflated → exclu de H1.
  - Notebook `baseline-cluster8.ipynb` créé (6 fonctions : 964, 965, 967, 968, 969, 977 — ~5 inv/min, ~25 % zéros).
- **Prochain pas** : uploader `baseline-cluster8.ipynb` sur Colab + Run All (T4 GPU). Ensuite lancer `baseline-cluster0.ipynb`. Archiver résultats et comparer métriques par cluster avec FAYAM Table VII.

### Session précédente : 2026-05-05 (session 22)

- **Phase actuelle** : **Phase 1 terminée** — dossier de run complet (HTML + CSV/JSON + PNG).
- **Avancée récente** :
  - `scatter_metrics.png`, `loss_curve.png`, `forecast_samples.png` archivés dans `results/`.
  - Run `2026-05-05_baseline-fayam-local-clusters` entièrement archivé localement.
- **Prochain pas** : démarrer **Phase 2 — H1** : lire `PLAN-ETUDE-ARCHITECTURE.md` (J1), puis implémenter SoftCAM-Transformer sur C0/C8.

### Session précédente : 2026-05-05 (session 17)

- **Phase actuelle** : **Phase 1** — notebook adapté pour les 4 clusters locaux (19 séries), run à relancer.
- **Avancée récente** :
  - Constat : `FaalSa/dataME` ne contient qu'1 série dans le test split → comparaison FAYAM impossible avec ce dataset.
  - Notebook `baseline-fayam-transformer.ipynb` adapté en 6 cellules : nouveau `RUN_NAME=2026-05-05_baseline-fayam-local-clusters`, chargement des 4 CSV depuis Drive, métriques avec colonnes `cluster`/`function_id`, nouvelle cellule de synthèse par cluster.
  - Pipeline GluonTS et architecture inchangés.
- **Prochain pas** : re-upload notebook sur Colab + Run All. Comparer ensuite par cluster avec FAYAM Table VII.

### Session précédente : 2026-05-04 (session 16)

- **Phase actuelle** : **Phase 1** — notebook Colab prêt, run à lancer. Plan d'étude architecture engagé.
- **Avancée récente** :
  - Clarification origine dataset `FaalSa/dataME` : pipeline FAYAM (Azure Trace → HDBSCAN → dataToHub.py → HuggingFace).
  - Plan d'étude architecture 5 jours créé → voir [`memoire/00-meta/PLAN-ETUDE-ARCHITECTURE.md`](PLAN-ETUDE-ARCHITECTURE.md).
  - Bug `TypeError: 'Axes' not iterable` corrigé dans `baseline-fayam-transformer.ipynb` (cellule 31).
  - Run Colab exécuté avec succès. HTML archivé. Métriques reportées dans `run.md` : MASE=0.8169, sMAPE=0.2903, RMSE=4.0750, R²=0.5845, Spearman=0.8342.
- **Prochain pas** : démarrer J1 du plan d'étude architecture — lecture Rasul & Rogge + esquisse encoder-decoder en Excalidraw.

### Session précédente : 2026-05-02 (session 15 — suite 4)

- **Phase actuelle** : **Phase 1** — notebook Colab prêt, run tracé, à lancer.
- **Avancée récente** :
  - Skill `experiment-tracker` exécuté : `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/` (run.md + command.sh + diff.patch).
  - Notebook `code/notebooks/baseline-fayam-transformer.ipynb` généré (37 cellules) — pipeline FAYAM complet + RMSE/R²/Spearman + extraction attention `output_attentions=True` post-training.
- **Prochain pas** : uploader `baseline-fayam-transformer.ipynb` sur Google Colab (T4 GPU) → Run All → remplir la table métriques dans `run.md`.

### Avant-dernière session : 2026-04-28 (session 13 — complète)

- **Phase actuelle** : **Phase 1 amorcée** — baseline FAYAM intégrée + EDA des 4 clusters terminée.
- **Avancée récente** : `code/notebooks/EDA_clusters.ipynb` créé (39 cellules, 11 sections) — analyse par fonction ET par cluster : statistiques descriptives, séries temporelles (heatmaps 14×1440, profils journaliers), zéros, distributions, ACF/FFT, ADF, cohérence intra/inter-cluster, recommandations prétraitement.
- **Prochain pas** : (1) exécuter l'EDA notebook pour valider les graphiques ; (2) lancer `tsf_transf.py` sur les 4 clusters (dataset HuggingFace `FaalSa/dataME`) et reproduire les métriques FAYAM (sMAPE, RMSE, R², Spearman).

### Avant-dernière session : 2026-04-28 (session 12 — finalisée)

- **Phase actuelle** : présentation panorama XAI prête à l'oral — 47 slides + script complet.
- **Avancée récente** : `SPEECH.md` créé (~30 s/slide, ~20 min, tableau temps cibles par section, conseils de rythme).
- **Prochain pas** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer, métriques.
