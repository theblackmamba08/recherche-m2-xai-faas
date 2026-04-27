# Journal de bord

> Une entrée par session significative. Format : date, durée, contenu, blocages.

## 2026-04-26 — Setup initial du dépôt

- **Durée** : ~1h
- **Fait** :
  - Arborescence complète du dépôt créée (65 fichiers)
  - 7 skills installés (rédaction-humaine, fiche-lecture, brief-debrief-presentation, experiment-tracker, biblio-sync, latex-compile, audit-similarite)
  - Hook Stop actif : vérifie MEMOIRE.md des 5 dossiers + JOURNAL.md global + section « État courant » de ROADMAP.md
  - Garde-fous éthiques intégrés (zéro génération masquée, refus skill anti-détection IA)
- **Décisions** : code unique dans `code/` ; fichiers Claude non commités ; un seul `presentations/` à la racine ; pas de STEPS/MEMOIRE racine (rôle joué par 00-meta/ROADMAP et 00-meta/JOURNAL).
- **Prochaine session** : récupérer le template LaTeX de Dschang, démarrer la phase 1 (reproduction FAYAM).

## 2026-04-27 — Fiche FAYAM + pivot stratégique XAI

- **Durée** : ~2h
- **Fait** :
  - Fiche de lecture du **mémoire FAYAM** (baseline architecturale) rédigée. Identification de la faille XAI à exploiter : 33 clusters DBSCAN sous-exploités, hyperparamètres Transformer non chiffrés (manque à combler).
  - Article **SoftCAM** (Djoumessi & Berens, arXiv:2505.17748v1, mai 2025) intégré, renommé `2025_Djoumessi_SoftCAM.pdf`, fiché.
  - **Pivot stratégique majeur** discuté avec les encadreurs et acté :
    - **H1 (NOUVELLE)** : adapter SoftCAM (interprétabilité intrinsèque + ElasticNet) au `TimeSeriesTransformer` HuggingFace de FAYAM.
    - **H2 (NOUVELLE)** : SHAP-based (TimeSHAP, KernelSHAP) en repli.
    - **H3 (NOUVELLE)** : attention weights + DBSCAN + faithfulness (ancienne H1) en dernier recours / sanity check.
  - Synchronisation complète : `README.md`, `CLAUDE.md`, `ROADMAP.md`, `DECISIONS.md`, `QUESTIONS-OUVERTES.md`, `01-litterature/MEMOIRE.md`, `03-contribution/MEMOIRE.md` + `STEPS.md`, `code/MEMOIRE.md` + `STEPS.md`, mémoire persistante `project_strategy.md`.
- **Décisions** : pivot stratégique XAI (cf. [DECISIONS.md](DECISIONS.md), entrée 2026-04-27).
- **Bloquants identifiés** : code FAYAM + hyperparamètres Transformer manquants ; vérifier accessibilité du code SoftCAM (`anonymous.4open.science`).
- **Prochaine session** : démarrer la **phase 1** — récupérer le code FAYAM, cartographier le `TimeSeriesTransformer` HuggingFace (localiser la projection finale du décodeur, cible H1).

## 2026-04-27 — Archivage présentation #1 (séance encadreurs du 23/10/2025)

- **Durée** : ~30 min
- **Fait** :
  - Convention d'archivage des présentations expliquée (dossier daté `YYYY-MM-DD-theme/`, BRIEF + DEBRIEF + slides.pdf).
  - Dossier `presentations/2025-10-23-series-chronologiques/` créé avec `BRIEF.md`, `DEBRIEF.md` et `slides.pdf` (31 diapositives Beamer, thème "Analyse et Modélisation des Séries Chronologiques").
  - Retours encadreurs documentés dans DEBRIEF : cold-start, XAI/IML, références obligatoires, schémas, edge computing / fork computing.
  - `presentations/MEMOIRE.md` mis à jour avec l'entrée de la présentation #1.
- **Prochaine session** : archiver les présentations suivantes ; commencer la phase 1 (code FAYAM).

## 2026-04-27 — Archivage présentation #2 (séance encadreurs du 25/10/2025)

- **Durée** : ~20 min
- **Fait** :
  - Dossier `presentations/2025-10-25-series-temporelles-explicabilite/` créé avec `BRIEF.md`, `DEBRIEF.md` et `slides.pdf` (18 diapositives Beamer).
  - Titre : "Séries Temporelles : De la Prédiction à la Confiance par l'Explicabilité" — nette amélioration par rapport à #1, XAI introduite mais non opérationnalisée.
  - Retours documentés : types de sorties d'un modèle à approfondir, dataset concret à choisir, étude comparative XAI (LIME/SHAP/Attention/CAM), références IEEE obligatoires, davantage de figures.
  - `presentations/MEMOIRE.md` mis à jour.
- **Prochaine session** : archiver les présentations suivantes puis démarrer la phase 1.

## 2026-04-27 — Archivage présentation #3 (séance encadreurs du 11/04/2026)

- **Durée** : ~20 min
- **Fait** :
  - Dossier `presentations/2025-11-11-series-temporelles-explicabilite-v3/` créé avec `BRIEF.md`, `DEBRIEF.md` et `slides.pdf` (31 diapositives Beamer).
  - Première vraie revue XAI présentée : SHAP, LIME, Attention, Saliency Maps, IG, TimeSHAP, WindowSHAP, ShaTS + tableaux comparatifs.
  - Action items encadreurs : étudier architecture LSTM puis Transformer (FAYAM), appliquer SHAP/LIME sur les modèles de FAYAM.
  - `presentations/MEMOIRE.md` mis à jour.
- **Prochaine session** : archiver les présentations restantes puis démarrer la phase 1.

## 2026-04-27 — Archivage présentation #4 (séance encadreurs du 25/04/2026)

- **Durée** : ~15 min
- **Fait** :
  - Dossier `presentations/2025-11-25-lstm-architecture/` créé avec `BRIEF.md` et `DEBRIEF.md` (pas de PDF disponible).
  - Présentation : généalogie neurone biologique → perceptron → MLP → RNN → LSTM, limites à chaque étape.
  - Encadreur a validé la compréhension d'ensemble et fourni des datasets + mission d'implémentation LSTM.
  - `presentations/MEMOIRE.md` mis à jour.
- **Prochaine session** : documenter les datasets reçus, chercher articles LSTM prédiction, implémenter.

## 2026-04-27 — Archivage présentation #5 (séance du 13/12/2025 — dernière de l'année 2025)

- **Durée** : ~15 min
- **Fait** :
  - Dossier `presentations/2025-12-13-foundation-models-survey/` créé avec `BRIEF.md`, `DEBRIEF.md` et `slides.pdf` (15 diapositives).
  - Présentation : fiche de lecture *Foundation Models for Time Series: A Survey* + rapport des difficultés LSTM (4h/époque, pas de résultats).
  - Décision encadreurs : suspendre LSTM, se concentrer sur *Attention Is All You Need* (Transformer originel).
  - Chronologie des présentations clarifiée : #3 = 13/12/2025 (fin 2025), #4 et #5 = avril 2026.

## 2026-04-27 — Archivage présentation #9 (25/04/2026) — SoftCAM, séance la plus récente

- Dossier `presentations/2026-04-25-softcam-presentation/` créé (24 diapositives, Dr LACMOU uniquement).
- Présentation jugée incomplète : manque comparaison post-hoc (TimeSHAP/SHAP) et lien avec architecture FAYAM.
- Action items immédiats : compléter la présentation, récupérer code FAYAM, EDA dataset, entraîner le modèle, ressortir les métriques.

## 2026-04-27 — Archivage présentation #8 (17/04/2026) — convergence H1 confirmée

- Dossier `presentations/2026-04-17-explicabilite-interpretabilite/` créé (24 diapositives, séance présentielle Pr KENGNE + Dr LACMOU).
- **Jalon majeur** : Dr LACMOU a formellement demandé d'adapter SoftCAM (Djoumessi 2025) au Transformer de FAYAM — H1 validée par les encadreurs.
- `presentations/MEMOIRE.md` et `ROADMAP.md` mis à jour.

## 2026-04-27 — Archivage présentations #6 et #7 (janv. 2026) + correction dates #3 et #4

- Présentations #6 (17/01/2026) et #7 (31/01/2026) archivées : démonstrations Colab d'implémentation Transformer (encodeur-décodeur + approche alternative) sur le dataset encadreur. Pas de PDF — BRIEF/DEBRIEF minimalistes, à compléter quand les notebooks seront retrouvés.
- Dates corrigées : présentations #3 et #4 renommées de 2026-04-11/25 → 2025-11-11/25. Toutes les références internes mises à jour.
- Chronologie complète : 5 présentations en 2025 (oct. × 2, nov. × 2, déc. × 1), 2 en janv. 2026.

## 2026-04-27 — Complément DEBRIEF présentation #4

- Article *Foundation Models for Time Series: A Survey* (recommandé par Dr DJOUMESSI Kerol) ajouté dans le DEBRIEF de la présentation #4.
- Action item créé : lire + ficher via skill `fiche-lecture` dans `memoire/01-litterature/`.
