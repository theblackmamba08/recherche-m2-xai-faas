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

## 2026-04-28 — Script de présentation (SPEECH.md)

- **Durée** : ~20 min
- **Fait** :
  - `SPEECH.md` créé dans `presentations/2026-04-28-explicabilite-panorama-methodes/` : speech complet slide par slide (~30 s/slide, ~20 min totales).
  - Un bloc par slide dans l'ordre exact, avec phrases de transition pour les 7 slides automatiques.
  - Tableau récapitulatif des temps cibles par section + conseils de rythme (section 4 abrégeable si retard).
- **Prochaine session** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer.

## 2026-04-28 — Références + finalisation présentation panorama XAI

- **Durée** : ~30 min (clôture de session)
- **Fait** :
  - Frame `Références` ajoutée à la fin de `slides.tex` (`allowframebreaks`, 10 entrées : LIME, KernelSHAP, TimeSHAP, WindowSHAP, TsSHAP, SHAPformer, CAM, GradCAM, SoftCAM, FAYAM).
  - Recompilation confirmée : **47 pages, 760 Ko, 0 erreur LaTeX** (5 slides de références auto-générés).
  - Suppression des 9 dossiers `figures/` vides dans tous les dossiers de présentations.
- **État final** : présentation **47 slides** complète et prête pour Dr LACMOU.
- **Prochaine session** : démarrer la **phase 1** — code FAYAM, dataset FaaS, entraînement Transformer, métriques.

## 2026-04-28 — Uniformisation style tcolorbox + gestion overflows

- **Durée** : ~1h (suite de session)
- **Fait** :
  - Style `tcolorbox` (bleu/vert/rouge/orange/gris) étendu aux slides 14→42 (sections 4c à 7 complètes).
  - Recompilation confirmée : 42 pages, 0 erreur LaTeX, PDF 693 Ko.
  - Stratégies de gestion des overflows documentées : `[shrink=X]`, `\footnotesize`, `boxsep`, découpage en deux frames.
- **Prochaine session** : corriger les overflows éventuels, puis démarrer la phase 1 (code FAYAM).

## 2026-04-28 — Panorama XAI : fiches de lecture + présentation complète + slides de transition

- **Durée** : ~4h (session complète)
- **Contexte** : présentation SoftCAM du 25/04 jugée incomplète par Dr LACMOU. Session consacrée à construire la base littéraire manquante, générer la présentation complète, puis l'affiner.
- **Fait** :
  - **5 articles ajoutés** dans `memoire/01-litterature/articles/` et fichés : TimeSHAP (4/5), WindowSHAP (3/5), TsSHAP (5/5), SHAPformer (5/5), LIME (2/5).
  - **Présentation Beamer générée et compilée** : `presentations/2026-04-28-explicabilite-panorama-methodes/slides.tex` (42 slides, PDF 974 Ko).
  - **Slides de transition** ajoutés via `\AtBeginSection` (préambule) : 1 slide de transition par section, TOC avec section courante en clair et autres grisées (`sectionstyle=show/shaded`).
  - Correction bug TikZ : ajout `decorations.pathreplacing` dans `\usetikzlibrary`.
  - Utilisateur prend la main sur le LaTeX pour ajustements visuels — reprendra lors d'une prochaine session.
- **Décisions** : TsSHAP = H2 prioritaire ; SHAPformer = H2 bonus ; LIME = référence historique uniquement.
- **Bloquants** : métriques FAYAM (slide 2.3) = placeholders `\textit{(valeur)}` à remplir après phase 1 ; article KernelSHAP sans PDF archivé.
- **Prochaine session** : démarrer la **phase 1** — récupérer le code FAYAM, reproduire les résultats Transformer.
- **Maintenance** : hook Stop résolu — `presentations/MEMOIRE.md` resynchronisé avec le journal global.

## 2026-04-28 — Panorama XAI : fiches de lecture + présentation complète (réponse Dr LACMOU)

- **Durée** : ~3h
- **Contexte** : présentation SoftCAM du 25/04 jugée incomplète par Dr LACMOU (manquait la comparaison avec les méthodes post-hoc et le lien avec FAYAM). Session consacrée à construire la base littéraire manquante et générer la présentation complète.
- **Fait** :
  - **5 articles ajoutés** dans `memoire/01-litterature/articles/` et fichés :
    - `2021_Bento_TimeSHAP.pdf` — pertinence 4/5 ; SHAP séquentiel pour RNN/classification ; coalition pruning (138.5 → 14.0 coalitions, η=0.025)
    - `2023_Nayebi_WindowSHAP.pdf` — pertinence 3/5 ; fenêtrage temporel ; variante Dynamic évite l'hypothèse anciens événements sans importance
    - `2023_Raykar_TsSHAP.pdf` — pertinence 5/5 ; **seule méthode SHAP conçue pour la prévision** ; surrogate XGBoost + TreeSHAP sur backtested forecasts
    - `2025_Hertel_SHAPformer.pdf` — pertinence 5/5 ; SHAP exact sampling-free pour Transformer de prévision ; 800× plus rapide que Permutation Explainer ; code GitHub disponible
    - `2016_Ribeiro_LIME.pdf` — pertinence 2/5 ; référence fondatrice XAI post-hoc ; repère historique pour l'état de l'art
  - **Présentation Beamer générée et compilée** : `presentations/2026-04-28-explicabilite-panorama-methodes/slides.tex` (35 slides, 902 Ko PDF, thème Madrid + seahorse, 7 sections).
    - Plan : Accroche → FAYAM et sa faille XAI → Taxonomie XAI → Famille SHAP (LIME→KernelSHAP→TimeSHAP→WindowSHAP→TsSHAP→SHAPformer) → Famille CAM (CAM→GradCAM→SoftCAM) → Tableau de synthèse → Retour à FAYAM avec grille de décision H1/H2
    - Correction bug TikZ : ajout `decorations.pathreplacing` dans `\usetikzlibrary`
- **Décisions** : TsSHAP identifié comme H2-prioritaire (seule méthode SHAP pour forecasting) ; SHAPformer identifié comme H2-bonus (SHAP exact pour Transformer) ; LIME = référence historique uniquement, pertinence directe nulle.
- **Bloquants restants** : KernelSHAP n'a pas d'article PDF archivé (présent dans la présentation via les fiches déjà lues) ; métriques FAYAM dans le slide 2.3 sont des placeholders `\textit{(valeur)}` à compléter après phase 1.
- **Prochaine session** : démarrer la **phase 1** — récupérer le code FAYAM, reproduire les résultats Transformer.

## 2026-04-28 — Phase 1 amorcée : datasets + baseline FAYAM

- **Durée** : ~1h
- **Fait** :
  - 4 clusters CSV reçus (clusters 0, 4, 6, 8 — 19 fonctions × 20 160 min, Azure Functions Trace 2019, fréquence minute) → déposés dans `memoire/06-datasets/raw/` + `DATA-CARD.md` rédigée.
  - Code FAYAM (`transformer_m2-main.zip`) analysé et intégré dans `code/src/baseline/fayam/` (référence immuable). Hyperparamètres Transformer retrouvés : `prediction_length=120`, `context_length=240`, `freq="1T"`, `encoder_layers=4`, `decoder_layers=4`, `d_model=32`. Dataset sur HuggingFace Hub : `FaalSa/dataME`.
  - Discordance DBSCAN/HDBSCAN détectée (mémoire FAYAM dit DBSCAN, le code utilise `hdbscan`) — à éclaircir avec l'encadreur.
- **Prochaine session** : lancer `tsf_transf.py` sur les 4 clusters et reproduire les métriques FAYAM (sMAPE, RMSE, R², Spearman).

## 2026-04-28 — EDA complète des 4 clusters (notebook)

- **Durée** : ~30 min (fin de session 13, suite directe de la phase 1)
- **Fait** :
  - `code/notebooks/EDA_clusters.ipynb` créé — 39 cellules, 11 sections couvrant l'analyse **par fonction** (19 séries) ET **par cluster** (0, 4, 6, 8).
  - Sections : vue d'ensemble → statistiques descriptives → séries temporelles (14 j + zoom 3 j + heatmap 14×1440 + profil journalier) → analyse des zéros (taux + runs consécutifs) → distributions (KDE, boxplot, skewness/kurtosis) → périodicité (ACF 2880 lags + FFT + top périodes) → stationnarité (ADF + diff(1) si non-stationnaire) → cohérence intra-cluster (Pearson + Spearman + distance euclidienne normalisée) → comparaison inter-cluster (CV, burstiness B=(σ-μ)/(σ+μ), profils normalisés overlay) → synthèse & recommandations prétraitement.
  - Recommandations clés documentées : ordre d'entraînement C0→C4→C8→C6, conserver les zéros natifs pour les clusters 6 et 8, normaliser par fonction, activer `output_attentions=True` dès le premier run.
- **Prochaine session** : exécuter le notebook pour vérifier les graphiques, puis lancer `tsf_transf.py` sur les 4 clusters.
- `code/MEMOIRE.md` mis à jour en conséquence.

## 2026-04-29 — Registre de résultats EDA (session 14)

- **Durée** : ~20 min
- **Fait** :
  - `code/experiments/eda/REGISTER.md` créé — tableau cumulatif mis à jour automatiquement à chaque run.
  - `EDA_clusters.ipynb` étendu à 47 cellules : cellule de détection Colab/Drive (index 3), 4 cellules de capture (overview, zéros, FFT top périodes, ADF stationnarité), section 12 (sauvegarde JSON Drive + REGISTER.md + `files.download()`).
  - Compatible local et Colab : le chemin de sauvegarde s'adapte automatiquement.
- **Prochaine session** : exécuter le notebook sur Colab, vérifier le JSON produit, puis lancer `tsf_transf.py`.
- `code/MEMOIRE.md` mis à jour en conséquence (session 14).
- Cellule rappel export HTML ajoutée en fin de notebook (48 cellules au total). Stratégie archivage finalisée : JSON auto + HTML manuel dans `code/experiments/eda/`.
- Documentation EDA complétée : `code/notebooks/README.md` créé (guide autonome pour utilisateur externe ou Claude futur), `STEPS.md` mis à jour (4 étapes Phase 1 cochées), `REGISTER.md` enrichi avec pointeurs.
- Push GitHub effectué (commit `4b8b4bb`) — token HF redacté dans `dataToHub.py` avant commit.

## 2026-04-27 — Complément DEBRIEF présentation #4

- Article *Foundation Models for Time Series: A Survey* (recommandé par Dr DJOUMESSI Kerol) ajouté dans le DEBRIEF de la présentation #4.
- Action item créé : lire + ficher via skill `fiche-lecture` dans `memoire/01-litterature/`.
