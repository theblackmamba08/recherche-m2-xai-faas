# Mémoire — presentations

> Liste chronologique des présentations.

## 2026-04-26 — Setup initial

- Dossier `_template/` créé avec `BRIEF.md` + `slides.tex` (Beamer thème Madrid/seahorse) + `DEBRIEF.md`.
- Convention de nommage `YYYY-MM-DD-mot-cle/` documentée dans [README.md](README.md).
- Suite → cf. [STEPS.md](STEPS.md) : adapter le template aux couleurs Dschang.

| Date | Titre | Audience | Lien |
|------|-------|----------|------|
| 2025-10-23 | Analyse et Modélisation des Séries Chronologiques | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2025-10-23-series-chronologiques/) |
| 2025-10-25 | Séries Temporelles : De la Prédiction à la Confiance par l'Explicabilité | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2025-10-25-series-temporelles-explicabilite/) |
| 2025-11-11 | Les Séries Temporelles : de la Prédiction à l'Explicabilité | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2025-11-11-series-temporelles-explicabilite-v3/) |
| 2025-11-25 | Architecture LSTM : du Neurone Biologique à la Mémoire Longue Terme | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2025-11-25-lstm-architecture/) |
| 2025-12-13 | Foundation Models for Time Series: A Survey | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2025-12-13-foundation-models-survey/) |
| 2026-01-17 | Implémentation Transformer Encodeur-Décodeur (Colab) | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2026-01-17-transformer-encodeur-decodeur-colab/) |
| 2026-01-31 | Implémentation Transformer : Approche Alternative (Colab) | Dr LACMOU ZEUTOUO, Dr DJOUMESSI, Pr KENGNE TCHENDJI, M. Yvan | [dossier](2026-01-31-transformer-approche-2-colab/) |
| 2026-04-17 | Explicabilité et Interprétabilité en IA : Fondations, Méthodes et Enjeux Éthiques | Pr KENGNE TCHENDJI, Dr LACMOU ZEUTOUO | [dossier](2026-04-17-explicabilite-interpretabilite/) |
| 2026-04-25 | Soft-CAM : Rendre les Modèles Boîtes Noires Auto-Explicables *(incomplète — à recompléter)* | Dr LACMOU ZEUTOUO | [dossier](2026-04-25-softcam-presentation/) |
| 2026-04-28 | Explicabilité : Panorama des méthodes XAI (LIME → SHAP → CAM → SoftCAM) | Dr LACMOU ZEUTOUO | [dossier](2026-04-28-explicabilite-panorama-methodes/) |
| 2026-05-09 | Panorama XAI v2 — canevas Dr LACMOU + zooms SHAPformer/SoftCAM + architecture SoftCAM-Transformer | Dr LACMOU ZEUTOUO | [dossier](2026-05-09-panorama-xai-v2/) |
| 2026-05-20 | Résultats H1 SoftCAM-Transformer — validation complète H1.A→H1.G + dissociation entraînement/inférence *(à présenter)* | Encadreurs | [dossier](2026-05-20-resultats-h1-softcam/) |

## 2026-04-27 — Ajout présentation #1

- Dossier `2025-10-23-series-chronologiques/` créé avec `BRIEF.md`, `DEBRIEF.md`, et `slides.pdf` (31 diapositives Beamer).
- Retours encadreurs documentés : cold-start, XAI/IML, références, schémas, edge computing.
- Suite → préparer la présentation #2 en intégrant les action items du DEBRIEF.

## 2026-04-27 — Ajout article recommandé dans DEBRIEF présentation #4

- Article *Foundation Models for Time Series: A Survey* (recommandé par Dr DJOUMESSI Kerol) ajouté dans le DEBRIEF de `2025-11-25-lstm-architecture/`.

## 2026-04-27 — Ajout présentation #8

- Dossier `2026-04-17-explicabilite-interpretabilite/` créé avec `BRIEF.md`, `DEBRIEF.md` et `slides.pdf` (24 diapositives).
- Séance présentielle (Pr KENGNE + Dr LACMOU uniquement). Présentation XAI approfondie : LIME, SHAP, Grad-CAM, NIST, mythes de Lipton.
- **Décision clé** : Dr LACMOU a demandé d'adapter SoftCAM (Djoumessi 2025) au Transformer de FAYAM — convergence vers H1.
- Action item ajouté : lire + ficher via skill `fiche-lecture`.

## 2026-04-28 — Script de présentation (SPEECH.md)

- `SPEECH.md` créé dans `2026-04-28-explicabilite-panorama-methodes/` : speech complet slide par slide, ~30 s par slide, total ~20 min.
- Inclut temps cibles par section, conseils de débit, et stratégie d'abréviation si retard en section 4 (SHAP).

## 2026-04-28 — Références + finalisation (47 slides)

- Frame `Références` ajoutée à la fin de `slides.tex` : 10 entrées bibliographiques, 5 slides auto via `allowframebreaks`.
- État final : **47 pages, 760 Ko, 0 erreur LaTeX**. Présentation prête pour Dr LACMOU.
- Suppression des 9 dossiers `figures/` vides dans tous les dossiers de présentations.

## 2026-04-28 — Uniformisation du style tcolorbox (suite)

- Style `tcolorbox` étendu aux slides 14→42 (sections 4c à 7) : bleu/vert/rouge/orange/gris selon le rôle du contenu.
- Recompilation confirmée : 42 pages, 0 erreur LaTeX.
- Stratégies de gestion des overflows documentées (shrink, footnotesize, boxsep, découpage).

## 2026-04-28 — Présentation panorama XAI (réponse Dr LACMOU)

- Dossier `2026-04-28-explicabilite-panorama-methodes/` créé avec `BRIEF.md` et `slides.tex` compilé (**42 diapositives**, PDF 974 Ko).
- Présentation construite suite au retour « incomplète » sur la séance du 25/04 : inclut toute la famille SHAP (LIME, KernelSHAP, TimeSHAP, WindowSHAP, TsSHAP, SHAPformer) + famille CAM (CAM, GradCAM, SoftCAM) + tableau de synthèse + grille de décision H1/H2.
- Slides de transition automatiques ajoutés via `\AtBeginSection` (1 slide par section, TOC avec section courante en clair).
- Métriques FAYAM (slide 2.3) = placeholders à compléter après phase 1.
- Ajustements visuels LaTeX pris en charge directement par l'utilisateur.

## 2026-04-27 — Ajout présentation #2

- Dossier `2025-10-25-series-temporelles-explicabilite/` créé avec `BRIEF.md`, `DEBRIEF.md`, et `slides.pdf` (18 diapositives Beamer).
- Amélioration notée par les encadreurs ; explicabilité introduite mais non opérationnalisée.
- Retours : sorties d'un modèle à approfondir, dataset concret à choisir, étude comparative XAI, références et figures obligatoires.

## 2026-05-08 — DEBRIEF présentation panorama XAI (28/04/2026)

- `DEBRIEF.md` créé pour `2026-04-28-explicabilite-panorama-methodes/` à partir du retour oral de l'étudiant.
- Retour Dr LACMOU : présentation **trop technique**, méthodes exposées comme blocs isolés sans fil conducteur. Étudiant en peine à l'oral.
- **Cadre prescrit pour les futures présentations XAI** : 4 points par méthode (contexte / problèmes résolus / transposabilité / limites) + **plus-value de transition** vers la méthode suivante. Acté dans [`DECISIONS.md`](../memoire/00-meta/DECISIONS.md) (entrée 2026-05-08) et mémoire persistante.
- Action items : reprendre les slides selon ce cadre, réduire le formalisme, construire le fil de plus-values jusqu'à SoftCAM-Transformer.

## 2026-05-20 — Brief présentation résultats H1 (session 68)

- Dossier `2026-05-20-resultats-h1-softcam/` créé : `BRIEF.md` pré-rempli + `slides.tex` squelette (8 sections).
- Brief couvre : architecture Evidence Layer, R²=0.7563, dissociation entraînement/inférence, grille H1.A→H1.G avec nuances, 5 questions anticipées, 3 demandes encadreurs.
- Suite → remplir les slides `.tex`, fixer date avec encadreurs, compiler PDF.

## 2026-05-09 — Présentation Panorama XAI v2 (canevas Dr LACMOU appliqué)

- Dossier `2026-05-09-panorama-xai-v2/` créé : `BRIEF.md`, `slides.tex`, `figures/softcam-transformer-architecture.svg`, `slides.pdf` compilé (**64 pages, 908 Ko, 0 erreur LaTeX**).
- Préambule hérité de la V1 (Madrid + seahorse), ajout `amssymb` + lib TikZ `calc` ; environnements `ctxbox / probbox / transposbox / limitbox / plusbox` créés pour matérialiser le canevas Dr LACMOU.
- Structure : 12 sections — pourquoi expliquer / contexte FAYAM (avec métriques Phase 1) / cadre canevas / SHAP panorama (LIME → KernelSHAP → TimeSHAP → WindowSHAP → TsSHAP) / **★ZOOM SHAPformer (7 slides)** / transition / CAM (CAM → GradCAM) / **★ZOOM SoftCAM (6 slides)** / synthèse comparative / **★Architecture SoftCAM-Transformer (10 slides : SVG embarqué TikZ, entrées/sorties par bloc, evidence layer Python, équation centrale, ElasticNet, variantes A/B/C, exemple carte C4, hypothèses H1.A--H1.E)** / conclusion / questions / références.
- Slide-discussion ajoutée : *« Peut-on rendre SHAPformer pleinement self-explainable ? »* — discussion FastSHAP, conclusion non sans dénaturer, transition vers SoftCAM.
- SVG d'architecture (`softcam-transformer-architecture.svg`) copié depuis `memoire/03-contribution/figures/`. Schéma TikZ équivalent embarqué dans le slides pour robustesse compilation.
- 9 passes de débogage compilation : virgules dans titres tcolorbox (encadrer en `{...}`), Unicode (★, →, ↔ → équivalents math LaTeX), `\verbatim` → frames marqués `[fragile]`, lib `calc` ajoutée.
