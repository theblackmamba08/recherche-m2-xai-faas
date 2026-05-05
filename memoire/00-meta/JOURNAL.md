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

## 2026-04-29 — Débogage Colab (session 14 — suite 2)

- **Durée** : ~10 min
- **Fait** :
  - Bug récurrent `KeyError: 'Zeros (%)'` sur la cellule capture zéros — cause : Colab avait encore l'ancienne version du notebook uploadée avant le fix `a113c93`.
  - Correction manuelle indiquée pour la session en cours + note ajoutée : toujours re-télécharger `EDA_clusters.ipynb` depuis GitHub avant d'uploader sur Colab.
- **Prochaine session** : re-uploader la version correcte, finaliser le run, archiver JSON + HTML dans `code/experiments/eda/`.

## 2026-05-02 — Finalisation archivage EDA + planification rapport (session 15 — suite)

- **Durée** : ~15 min
- **Fait** :
  - HTML renommé proprement : `EDA_clusters.ipynb - Colab.html` → `EDA_clusters_2026-05-02.html` (+ dossier `_files` renommé en cohérence). `REGISTER.md` mis à jour.
  - Décision : créer `memoire/02-baseline/EDA_RAPPORT.md` — synthèse narrative de l'EDA (contexte, résultats chiffrés, profil par cluster, décisions prétraitement) pour présentation aux encadreurs et base du chapitre données du mémoire.
- **Prochaine session** : rédiger `EDA_RAPPORT.md`, puis lancer `tsf_transf.py`.

## 2026-05-02 — Rapport EDA + clôture session 15

- **Durée** : ~20 min
- **Fait** :
  - `memoire/02-baseline/EDA_RAPPORT.md` rédigé (7 sections, chiffres réels du JSON) : contexte, description données, stats descriptives par fonction, stationnarité (19/19 ADF p≈0), périodicité (24h universelle), zéros, profil par cluster, décisions prétraitement, ordre d'entraînement C0→C4→C8→C6, résumé exécutif.
  - Commité et poussé (commit `f0f9e73`).
  - Audit complet des fichiers de suivi : ROADMAP, JOURNAL, DECISIONS, QUESTIONS-OUVERTES, memory/ tous mis à jour.
- **Prochaine session** : lancer `tsf_transf.py` sur les 4 clusters, reproduire les métriques FAYAM.

## 2026-05-02 — Refactoring DATA_DIR + comparaison notebooks (session 15 — suite 2)

- **Durée** : ~20 min
- **Fait** :
  - Notebook Colab exécuté (49 cellules, 31 outputs) comparé au local (48 cellules, propre) — seule vraie différence : `DATA_DIR` hardcodé dans la cellule d'imports.
  - Refactoring : cellule d'imports scindée en 3 (imports / détection+DATA_DIR adaptatif / chargement). `DATA_DIR` se fixe maintenant automatiquement selon l'environnement. Commit `06b7c77`, poussé.
- **Prochaine session** : rédiger `memoire/02-baseline/EDA_RAPPORT.md`, puis lancer `tsf_transf.py`.

## 2026-05-02 — Premier run EDA complet + analyse résultats (session 15)

- **Durée** : ~30 min
- **Fait** :
  - Run Colab `EDA_clusters.ipynb` exécuté avec succès (08h31). JSON + HTML archivés dans `code/experiments/eda/`.
  - Résultats clés : 19/19 fonctions stationnaires (ADF p≈0), période dominante 24h universelle pour tous les clusters, moyennes C0=121 937 / C4=97 / C6=2 / C8=5 inv/min.
  - `REGISTER.md` mis à jour manuellement avec les chiffres du run.
- **Prochaine session** : lancer `tsf_transf.py` sur les 4 clusters pour reproduire les métriques FAYAM.

## 2026-04-29 — Correction bug accents + exécution Colab (session 14 — suite)

- **Durée** : ~15 min
- **Fait** :
  - Bug `KeyError: 'Zeros (%)'` corrigé dans les cellules de capture du notebook (accents manquants sur `Zéros (%)` et `Zéros consécutifs max (h)`) — fix commité et poussé (commit `a113c93`).
  - Chemin des données adapté pour Colab : `DATA_DIR = Path('/content/drive/MyDrive/Recherche/Datasets')`.
  - Notebook en cours d'exécution sur Colab au moment de la clôture.
- **Prochaine session** : récupérer JSON + HTML depuis Colab, les archiver dans `code/experiments/eda/`, puis lancer `tsf_transf.py`.

## 2026-04-27 — Complément DEBRIEF présentation #4

- Article *Foundation Models for Time Series: A Survey* (recommandé par Dr DJOUMESSI Kerol) ajouté dans le DEBRIEF de la présentation #4.
- Action item créé : lire + ficher via skill `fiche-lecture` dans `memoire/01-litterature/`.

## 2026-05-02 — Audit documentation + guide EDA cellule par cellule (session 15 — suite 3)

- **Durée** : ~20 min
- **Fait** :
  - `memoire/02-baseline/EDA_RAPPORT.md` réécrit intégralement : remplacé la synthèse scientifique (7 sections) par un **guide cellule par cellule** des 49 cellules du notebook — chaque cellule explique ce qu'elle fait, pourquoi elle est là, et ce qu'on y observe. Lisible comme narration lors d'une présentation aux encadreurs.
  - `code/MEMOIRE.md` complété avec entrée datée pour cette session d'audit.
  - Audit de l'ensemble des fichiers de suivi confirmé complet : ROADMAP, JOURNAL, DECISIONS, QUESTIONS-OUVERTES, memory/ (`project_phase1_eda.md`, `MEMORY.md`), STEPS.md, code/MEMOIRE.md.
- **Prochaine session** : lancer `src/baseline/fayam/tsf_transf.py` sur les 4 clusters (dataset HuggingFace `FaalSa/dataME`) et reproduire les métriques FAYAM.

## 2026-05-02 — Experiment tracker + notebook Colab baseline (session 15 — suite 4)

- **Durée** : ~30 min
- **Fait** :
  - Skill `experiment-tracker` exécuté : dossier de run `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/` créé avec `run.md`, `command.sh`, `diff.patch` (7 fichiers non commités capturés).
  - Notebook Colab généré : `code/notebooks/baseline-fayam-transformer.ipynb` (37 cellules, JSON valide). Fidèle au `tsf_transf.py` FAYAM avec ajouts : seeding complet, gradient clipping, checkpoints Drive toutes les 10 époques, tqdm, métriques RMSE/R²/Spearman, extraction `output_attentions=True` post-training (cross_attn + enc_attn couche 4 → `.npy` Drive).
  - `code/STEPS.md`, `code/MEMOIRE.md`, `memoire/03-contribution/MEMOIRE.md` mis à jour.
- **Prochaine étape** : uploader `baseline-fayam-transformer.ipynb` sur Google Colab (T4 GPU) et lancer Run All.

## 2026-05-04 — Debug notebook + plan d'étude architecture (session 16)

- **Durée** : ~1h
- **Fait** :
  - Origine du dataset `FaalSa/dataME` clarifiée : pipeline FAYAM complet (Azure Trace → HDBSCAN → `dataToHub.py` → HuggingFace Hub).
  - Plan d'étude architecture 5 jours créé (`memoire/00-meta/PLAN-ETUDE-ARCHITECTURE.md`) : pipeline données GluonTS, encodeur, décodeur, inférence + production de visuels PNG/.excalidraw.
  - Bug `TypeError: 'Axes' object is not iterable` corrigé dans `baseline-fayam-transformer.ipynb` (cellule 31) : `squeeze=False` + `.flatten()`.
- **Prochaine étape** : lancer le notebook sur Colab T4 + démarrer J1 du plan d'étude (lecture Rasul & Rogge + esquisse architecture).

## 2026-05-04 — Run baseline Colab exécuté + archivage (session 16 — suite)

- **Durée** : ~20 min
- **Fait** :
  - Run `baseline-fayam-transformer.ipynb` exécuté avec succès sur Colab T4 GPU.
  - HTML du notebook + dossier `_files` copiés dans `code/experiments/runs/2026-05-02_11-15_baseline-fayam-transformer/`.
- **Bloquant** : métriques finales (MASE, sMAPE, RMSE, R², Spearman) pas encore reportées dans `run.md` — en attente de l'utilisateur.
- **Prochaine étape** : compléter `run.md` avec les métriques, puis démarrer J1 du plan d'étude architecture.

## 2026-05-04 — Métriques baseline extraites + run.md finalisé (session 16 — clôture)

- **Durée** : ~10 min
- **Fait** :
  - Métriques extraites depuis le HTML Colab : MASE=0.8169, sMAPE=0.2903, RMSE=4.0750, R²=0.5845, Spearman=0.8342.
  - `run.md` complété et status passé à `done`. Ces valeurs sont désormais la **baseline de référence** pour H1/H2.
- **Prochaine étape** : J1 du plan d'étude architecture — lecture Rasul & Rogge + esquisse encoder-decoder Excalidraw.

## 2026-05-05 — Adaptation notebook pour les 4 clusters locaux (session 17)

- **Durée** : ~45 min
- **Contexte** : `FaalSa/dataME` ne contient qu'1 série dans le test split (constaté à partir du `metrics.csv` à 1 ligne). Comparaison FAYAM Table VII impossible.
- **Fait** :
  - Notebook `baseline-fayam-transformer.ipynb` adapté en 6 cellules ciblées : nouveau `RUN_NAME=2026-05-05_baseline-fayam-local-clusters`, chargement direct des 4 CSV depuis `Drive/Recherche/Datasets/cluster_{0,4,6,8}.csv` (19 séries × 20 160 pas), métriques enrichies de `cluster` + `function_id`, nouvelle cellule de synthèse par cluster + sauvegarde `metrics_by_cluster.csv` et `cluster_mapping.csv`.
  - Pipeline GluonTS et architecture du modèle inchangés. `cardinality=[len(train_dataset)]` s'auto-ajuste à 19.
- **Prochaine étape** : re-upload sur Colab + Run All. Comparer les métriques par cluster avec FAYAM Table VII (notamment cluster 0 = profil très populaire vs Dataset 12 FAYAM).

## 2026-05-05 — Traçage run local-clusters + diagnostic NameError (session 18)

- **Durée** : ~15 min
- **Fait** :
  - `NameError: name 'cross_attn_arr' is not defined` (cellule 34) diagnostiqué : problème d'ordre d'exécution Colab — cellule 34 lancée avant cellule 33. Solution : Runtime → Run All.
  - Dossier de traçage `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/` créé avec `run.md` pré-rempli : configuration, tableau métriques vide (global + par cluster C0/C4/C6/C8), chemins Drive des sorties attendues.
  - `code/MEMOIRE.md` et section « État courant » de `ROADMAP.md` mis à jour (session 18).
- **Prochaine étape** : re-upload notebook sur Colab → Runtime → Run All → remplir `run.md` avec les métriques → comparer par cluster avec FAYAM Table VII.

## 2026-05-05 — Correction cellule attention manquante (session 19)

- **Durée** : ~10 min
- **Fait** :
  - Diagnostic approfondi du `NameError` : la cellule de code d'extraction des attention weights était **absente** du notebook — seul le titre markdown "## 11 — Extraction attention" était présent, sans le code correspondant.
  - Cellule insérée après `cell-33` dans `baseline-fayam-transformer.ipynb` : forward pass teacher-forcé avec `output_attentions=True`, extraction `cross_attentions[-1]` (dernière couche décodeur) + `encoder_attentions[-1]`, empilement en `cross_attn_arr` shape `(n, heads, pred_len, ctx_len)` et `enc_attn_arr`, sauvegarde `.npy` sur Drive.
- **Prochaine étape** : re-upload notebook sur Colab → Runtime → Run All → remplir `run.md` avec métriques → comparer avec FAYAM Table VII.

## 2026-05-05 — Archivage HTML run local-clusters (session 20)

- **Durée** : ~5 min
- **Fait** :
  - Run Colab `2026-05-05_baseline-fayam-local-clusters` exécuté avec succès par l'utilisateur (19 séries, 4 clusters locaux, cellule attention incluse).
  - HTML `baseline-fayam-transformer.ipynb - Colab2.html` + `_files/` copiés depuis `Downloads/` et renommés dans `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/` : `baseline-fayam-local-clusters - Colab.html` + `_files/`.
- **Prochaine étape** : récupérer `metrics.csv`, `metrics_by_cluster.csv`, `cluster_mapping.csv`, `metrics.json` depuis Drive → compléter `run.md` → comparer par cluster avec FAYAM Table VII.

## 2026-05-05 — Analyse métriques + clôture Phase 1 (session 21)

- **Durée** : ~20 min
- **Fait** :
  - 4 fichiers résultats copiés depuis `Downloads/` dans `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/results/`.
  - `run.md` complété avec métriques réelles (global : MASE=1.38, sMAPE=1.45, R²=-1.26) et analyse par cluster : C0 Spearman=0.74 (tendance captée, amplitude ratée), C4 Spearman=-0.90 (prédit à l'envers), C6 prédit 0 constant (zero-inflated), C8 MASE=0.44 (seul cluster < 1). Écart avec FAYAM (R²=0.958) expliqué par hétérogénéité des 19 séries dans un seul modèle + dataset trop petit.
  - Status run → `done`. Phase 1 **terminée et archivée**.
- **Décision** : C6 à exclure de H1 (trop zero-inflated). Priorité : C0 + C8 pour l'analyse d'attention et SoftCAM-Transformer.
- **Prochaine étape** : démarrer **Phase 2 — H1** : lire `PLAN-ETUDE-ARCHITECTURE.md` (J1), puis implémenter SoftCAM-Transformer sur C0/C8.

## 2026-05-05 — Archivage figures PNG (session 22)

- **Durée** : ~5 min
- **Fait** :
  - `scatter_metrics.png`, `loss_curve.png`, `forecast_samples.png` téléchargés depuis Drive et copiés dans `code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/results/`.
  - Dossier de run entièrement archivé localement : HTML + CSV/JSON + PNG. `cross_attention_heatmap.png` reste sur Drive (à récupérer si besoin pour la rédaction).
- **Prochaine étape** : Phase 2 — H1 (SoftCAM-Transformer sur C0/C8).

## 2026-05-05 — Runs par cluster : création notebooks + archivage C6 (session 23)

- **Durée** : ~45 min
- **Contexte** : métriques du run 19-séries (session 21) très mauvaises pour C4 et C6. Analyse : FAYAM entraîne un modèle par série/dataset (pas un modèle global sur tous les clusters). Décision : passer aux runs dédiés par cluster pour isoler les profils.
- **Fait** :
  - **Notebook `baseline-cluster0.ipynb` créé** (34 cellules) : chargement `cluster_0.csv` uniquement (3 fonctions — 942, 943, 944), plots par fonction (zoom 6 h + vue 24 h, 2 colonnes), vue comparative 3 fonctions (subplot vertical), heatmap cross-attention par fonction. Générique : `CLUSTER_ID = 0` suffit à tout adapter.
  - **Run cluster 6 archivé** : dossier `code/experiments/runs/2026-05-05_baseline-cluster6/` créé. Résultats extraits du zip Drive (`drive-download-20260505T212415Z-3-001.zip`) + HTML + `_files/` copiés. `run.md` rédigé avec métriques (RMSE≈0, MASE≈0, sMAPE≈2, R²=0, Spearman=NaN pour les 5 fonctions) et analyse : modèle prédit 0 constant, C6 zero-inflated → **exclu de H1**.
  - **Notebook `baseline-cluster8.ipynb` créé** (34 cellules) : identique à cluster 0, `CLUSTER_ID = 8` — 6 fonctions (964, 965, 967, 968, 969, 977), ~5 inv/min, ~20-28 % de zéros, profil légèrement zero-inflated mais modélisable. Tableau stats EDA (CV, burstiness B) intégré dans le titre.
- **Décisions** :
  - C6 définitivement exclu de H1/H3 (zero-inflated, aucune information dans l'attention).
  - Clusters prioritaires pour H1 : **C0** (profil populaire, Spearman=0.74 même avec modèle hétérogène) et **C8** (MASE=0.44, le plus propre).
- **Prochaine étape** : lancer `baseline-cluster8.ipynb` sur Colab T4, puis `baseline-cluster0.ipynb`. Archiver les résultats. Comparer métriques dédiées vs run mélangé.
- `code/MEMOIRE.md` retouché en fin de session pour refléter la complétion de `baseline-cluster8.ipynb` (ligne "Suite → créer" remplacée par confirmation de création).

## 2026-05-05 — Archivage run cluster 8 (session 24)

- **Durée** : ~10 min
- **Fait** :
  - Run `baseline-cluster8.ipynb` exécuté par l'utilisateur sur Colab T4. Zip Drive + HTML téléchargés et archivés dans `code/experiments/runs/2026-05-05_baseline-cluster8/`.
  - `run.md` rédigé avec métriques par fonction et comparaison modèle dédié vs mixte : résultats identiques (MASE=0.44, sMAPE≈2.0, R²≈-0.79) → isolation ne résout pas le problème zero-inflated de C8.
- **Décision** : C0 confirmé comme seule cible viable pour H1 (signal riche, profil populaire).
- **Prochaine étape** : lancer `baseline-cluster0.ipynb` sur Colab T4.

## 2026-05-05 — Synthèse 4 baselines + bascule cible H1 vers C4 (session 25)

- **Durée** : ~30 min
- **Contexte** : avec les 4 runs dédiés archivés (C0/C4/C6/C8), comparaison croisée entre les prédictions EDA et les résultats baseline pour valider/infirmer le choix de cible H1.
- **Fait** :
  - **Synthèse des 4 baselines** : C4 est le **seul cluster où le modèle apprend** (R²=0.37, Spearman=0.92). C0 R²≈0 (échec inattendu), C6/C8 = trivial predictors (RMSE≈0, sMAPE=2).
  - **Confrontation EDA vs résultats** : 3/4 résultats étaient prévisibles (C6 explicitement annoncé par EDA cellule 17, C4 par FFT 75–80 % variance, C8 plus mitigé). C0 = surprise — l'EDA n'avait pas alerté sur la magnitude absolue (~120 000) interagissant avec le scaler interne du `TimeSeriesTransformer`.
  - **Bascule de cible H1** : la conclusion de session 24 (*"C0 = seule cible viable"*) est invalidée par le run C0 dédié. **Nouvelle cible primaire : C4**.
  - Hypothèses opératoires H1.A à H1.E définies et expliquées en langage simple à l'utilisateur (avant validation puis acte dans `DECISIONS.md`).
- **Décisions** : voir [DECISIONS.md](DECISIONS.md) entrée 2026-05-05. C4 retenu, C0 archivé comme cas-limite pour le chapitre Discussion (diagnostic scaler à creuser ultérieurement).
- **Fichiers de suivi mis à jour** : `DECISIONS.md` (entrée 2026-05-05), `ROADMAP.md` (État courant session 25), `code/MEMOIRE.md`, `memoire/02-baseline/MEMOIRE.md` (tableau comparatif 4 baselines), `memoire/03-contribution/MEMOIRE.md` (cible H1=C4 + hypothèses H1.A–H1.E), mémoire persistante (`project_phase1_baseline_results.md` + `MEMORY.md`).
- **Prochaine étape** : démarrer **Phase 2** — étude architecture `TimeSeriesTransformer` HuggingFace (cf. `PLAN-ETUDE-ARCHITECTURE.md`, J1) en vue de l'implémentation SoftCAM-Transformer sur C4.
