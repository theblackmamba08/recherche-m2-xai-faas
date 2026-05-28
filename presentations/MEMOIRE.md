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
| 2026-05-20 | Résultats H1 SoftCAM-Transformer — validation complète H1.A→H1.G + dissociation entraînement/inférence | Encadreurs | [dossier](2026-05-20-resultats-h1-softcam/) |

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

## 2026-05-23 — Q&A explication Run B collapse en français simple (session 86)

- Explication vulgarisée du collapse Run B (mix=0.3 dès le début) via analogie moniteur de conduite/rétroviseur mal réglé.
- Message clé : M pas encore structurée à l'init → mélanger dès le départ perturbe la rétropropagation → effondrement sur position arbitraire.
- Suite → reformuler H1.A ; relire SPEECH.md.

## 2026-05-23 — Q&A méthodes post-hoc par perturbation + TFT (session 85)

- Tour d'horizon LIME / SHAP / TimeSHAP / TsSHAP / WindowSHAP — principe perturbation, coût 2^n, lien avec H2 de repli.
- TFT (Temporal Fusion Transformer, Lim 2019) défini : concurrent direct sur l'interprétabilité, mais attention non régularisée → argument Jain & Wallace pour défendre M face au jury.
- Suite → reformuler H1.A ; relire SPEECH.md.

## 2026-05-23 — Q&A méthodes post-hoc par gradient (session 84)

- Tour d'horizon des méthodes gradient : Vanilla Gradients, SmoothGrad, Integrated Gradients, GradCAM — principe commun (∂ŷ/∂x), limites (sensibilité locale, pas de fidélité garantie).
- Lien établi : IG via Captum = H2 abandonné ; GradCAM = ancêtre de SoftCAM. Argument anti-gradient réaffirmé pour défendre M face aux encadreurs.
- Suite → reformuler H1.A ; relire SPEECH.md.

## 2026-05-23 — Q&A interprétabilité/explicabilité + correction slide 4 (session 83)

- Q&A : distinction interprétabilité (propriété intrinsèque du modèle) vs explicabilité (artefact post-hoc) ; lien avec Rudin 2019 (explication d'un modèle interprétable = redondant et risqué).
- Slide 4 : "Débogage" remplacé par "Diagnostic" (terme plus précis — diagnostic du raisonnement modèle, pas correction de code).
- Suite → reformuler H1.A (test diagonal) ; relire le SPEECH.md slide par slide.

## 2026-05-23 — Q&A Rudin 2019 : propriétés d'une vraie explication (session 82)

- Explication des 3 propriétés Rudin à partir de la fiche `2019_Rudin_StopExplaining.md` : simulable / features signifiantes / fidèle par construction.
- Lien établi avec H1 : l'Evidence Layer M est dans le modèle (fidélité par construction), pas une approximation externe comme SHAP/LIME.
- Suite → reformuler H1.A ; débriefer après la présentation réelle.

## 2026-05-25 — Débrief présentation encadreurs (session 82)

- Présentation tenue aux encadreurs (Dr LACMOU, Dr DJOUMESSI, Pr KENGNE TCHENDJI).
- `DEBRIEF.md` créé : 7 retours majeurs documentés — théorie avant application, dot product, renommage, évaluation explicabilité, schémas visuels, seuil R², confiance en M.
- Suite → révision architecturale justifiée par la littérature ; noms alternatifs ; compléter évaluation explicabilité.

## 2026-05-22 — SPEECH.md complet 35 slides (session 81)

- Création de `2026-05-20-resultats-h1-softcam/SPEECH.md` : script de présentation intégral, un paragraphe par slide, première personne, français naturel.
- Couvre les 35 slides : titre → contexte FaaS → panorama XAI → architecture → runs B → résultats H1.A–H1.G → positions/limites → demandes encadreurs.
- Suite → relire le speech par sections pour l'affiner ; débriefer après la présentation réelle.

## 2026-05-21 — Slide 35 restructuré + Q&A conceptuel (session 80)

- Slide 35 : "Ma priorité immédiate" reformulée en deux blocs de questions — "Ce que je propose — vos avis ?" (B5+mix=0.25, H1.A incomplet, calibrated self-explainability) et "Ce qu'il me reste à décider avec vous" (post-hoc, calendrier draft).
- Q&A : définition "idiosyncratique" dans contexte M, explication "calibrated self-explainability" (entre native et post-hoc — honnête sur la dissociation entraînement/inférence).
- Suite → reformuler H1.A ; slides 32+ bilan ; confirmer chiffre FAYAM.

## 2026-05-21 — Diagramme TikZ architecture + slide loss simplifié + Q&A (session 79)

- Slide 7 (architecture) : figure PDF remplacée par diagramme TikZ fait main — 5 blocs verticaux (Entrées→Encodeur→Décodeur→Evidence Layer→Prédiction), flèche latérale enc_hidden en tirets softcam, badge "nouveauté H1".
- Slide 8 (loss régularisée) : reformulé en français simple — problème d'abord ("carte plate vs collapse"), tableau 3 lignes avec analogies (surligneur/stabilisateur/filtre), formule reléguée dans l'interpbox.
- Slide 4 (stakeholders) : 4 stakeholders complets avec descriptions concrètes.
- Q&A : explications Run A vs Run B, origine du gap R²=0.37→0.53 (cluster seul + HPO).
- Suite → reformuler H1.A (test diagonal) ; slides 32+ bilan ; confirmer chiffre FAYAM.

## 2026-05-21 — Finitions slides + architecture + réécriture "positions et limites" (session 78)

- Slide 30 (H1.F) : verdictbox rendue explicite (formule 5.37%/25%≈21%) puis simplifiée en français courant ("M ne pèse que 25% dans le calcul final").
- Slides 33/34/35 (section "Position, limites et perspectives") : réécrits intégralement en français à la 1ère personne — titres reformulés, méta-instructions supprimées, ton humain direct.
- Slide 7 (architecture) : placeholder remplacé par `\includegraphics` sur `architecture-globale-revisee.pdf`.
- Slide 9 (Run B) : figures TikZ (loss stagnation + collapse M) ajoutées puis retirées sur décision de l'étudiant.
- Suite → questions ouvertes de l'étudiant (session Q&A amorcée).

## 2026-05-21 — Figures TikZ slides 30/31 H1.F/H1.G (session 77)

- Slide 30 (H1.F comprehensiveness) : courbe ΔMAE vs k (axe x non-linéaire : 0,1,10,100,max) avec zone softcam!15 sous la courbe, plafond warning 25% pointillé, bracket bilatéral fayam "79% restant", annotation "+5.37%" sur le max. Message visuel : le modèle utilise M de façon causale, mais reste très en-deçà du plafond.
- Slide 31 (H1.G sufficiency) : bar chart k=1/5/10/50 avec axe y zoomé 90–105% (barre brisée au bas), ligne 100% success, valeurs au-dessus des barres. Message : une seule position de M suffit à 97% de préservation.
- Compilation : 43 pages, 0 erreur.
- Suite → slides de conclusion/bilan H1.A–H1.G ; reformuler H1.A test diagonal ; confirmer chiffre dataset.

## 2026-05-21 — Figures TikZ slides 27/28/29 + discussion H1.E (session 76)

- Slide 27 (H1.C) : bar chart TikZ R²/Spearman avec ligne de seuil blanche pointillée à l'intérieur de chaque barre et delta `✓ +0.456` / `✓ +0.067` au-dessus.
- Slide 28 (H1.D) : heatmaps schématiques supprimées sur décision — slide simplifié avec Pearson=0.992 en pleine largeur + interpbox + problembox (nuance confondant).
- Slide 29 (H1.E) : scatter plot TikZ R² vs entropy(M), 5 points construits pour ρ=-0.80, droite de régression pointillée grise.
- ⚠️ Limite identifiée : avec n=5 fonctions, ρ=-0.80 n'est pas statistiquement significatif — H1.E à présenter comme tendance indicative, appuyée par H1.F/G.
- Suite → figures slides 30 (H1.F) et 31 (H1.G) ; reformuler H1.A test diagonal.

## 2026-05-21 — Figures TikZ slides 22/24/25 + réforme slide 26 (session 75)

- Slide 22 (Généralisation B6/B7) : coordonnées des 3 courbes et marqueurs remappés à la nouvelle échelle x (scale=12.8/unité, x→9) ; scale TikZ réduit de 0.88 à 0.75.
- Slides 24/25 : waterfall chart 4 paliers (FAYAM→RunA→B5mix=0→B5mix=0.25, deltas +15.98/+13.46/+9.17 pp) et bar chart FAYAM vs B5+mix=0.25 avec ligne TFT pointillée (non mesuré) produits en TikZ.
- Slide 26 (Méthodologie) : 7 hypothèses reformulées en propositions déclaratives avec critère explicite dans un tableau booktabs ; 2 bugs LaTeX Beamer résolus (`\begin{footnotesize}` invalide, `p{}` colonnes → minipage+`l`).
- ⚠️ H1.A identifiée comme mal formulée : test argmax/17-19h trop partiel ; reformulation vers test diagonal `argmax(M[t]) ≈ heure(t)` à implémenter.
- Suite → reformuler H1.A dans slides.tex + H1-narration.md ; produire figures slides 27+.

## 2026-05-21 — Figures TikZ slides 12-13 + explications pédagogiques (session 74)

- Slide 12 (Run B2) : figure TikZ schedules mix/γ corrigée selon le vrai notebook — plateau à 0 jusqu'aux époques 15/25, puis rampes, puis plateaux cibles ; époque 40 ajoutée sur l'axe des abscisses.
- Slide 13 (Run B3 LayerNorm) : figure remplacée — deux histogrammes côte à côte (Avant LN : spike à 0.97 / Après LN : distribution uniforme max=0.06) illustrant le collapse vs la distribution de M.
- Explication pédagogique du schéma LayerNorm fournie : avant = artifact numérique, après = vraie carte d'attention.
- Suite → continuer les figures placeholders restantes.

## 2026-05-21 — Affinages slides 5-6 + explications pédagogiques (session 73)

- Slide 5 (Panorama XAI) : exemples ajoutés famille post-hoc gradient (Vanilla Gradients, SmoothGrad, Grad-CAM) et famille intrinsèque (CAM, Concept Bottleneck Models, SoftCAM) ; espaces compressés pour éviter le débordement.
- Slide 6 (attention ≠ explication) : tentative de restructuration en tableau puis 3 colonnes rejetée (erreurs LaTeX Beamer) → retour à la version bullet list d'origine sur demande utilisateur.
- Explications pédagogiques fournies : sens de "distribution adversariale" (Jain & Wallace 2019) en français simple.
- Suite → continuer la lecture des slides suivants ; produire les figures placeholders.

## 2026-05-21 — Affinages slides 4-5 + explications pédagogiques (session 72)

- Slide 4 : bullet Débogage enrichi (vrai pattern vs artifact, ressources gaspillées) ; Stakeholders revenus à la version simple sans Auditeur (2 items : opérateur cloud + développeur).
- Slide 5 : explication pédagogique des 3 familles XAI temporelles (post-hoc gradient, post-hoc perturbation, intrinsèque) sans modification du .tex — positionnement H1 dans le panorama.
- Suite → continuer la lecture et l'explication des slides suivants ; produire les figures placeholders.

## 2026-05-21 — Correction slide 3 + explication slide 4 (session 71)

- Chiffre "~1900 fonctions Lambda" retiré de la slide 3 (non vérifiable) — remplacé par "33 clusters DBSCAN" seul, à compléter après confirmation encadreurs.
- Explication pédagogique du slide 4 ("Pourquoi expliquer ?") fournie : 3 motivations (Lipton/Rudin/EU AI Act) + stakeholders FaaS.

## 2026-05-20 — Contenu complet slides H1 + ajustements footer (session 70)

- `slides.tex` entièrement meublé : les 35 frames passent de squelette à contenu réel (textes, tableaux, colonnes, boîtes tcolorbox, TikZ, citations). Compilation : 44 pages, 967 Ko.
- Footer ajusté : section centrale ("Validation H1 : SoftCAM-Transformer") affichée en **noir sur fond blanc** via `\setbeamercolor{title in head/foot}{fg=black,bg=white}`.
- ⚠️ **À vérifier** : chiffre "~1900 fonctions Lambda" slide 3 non validé — à confirmer dans le mémoire FAYAM avant présentation.
- Suite → produire les ~10 figures placeholders + vérifier les chiffres du dataset FAYAM + fixer date encadreurs.

## 2026-05-20 — Squelette présentation H1 35 slides (session 69)

- `slides.tex` reconstruit en squelette complet : 7 actes / 35 frames + transitions auto + clôture, alignés sur le plan détaillé.
- Préambule hérité du panorama explicabilité (Madrid+seahorse, couleurs shap/cam/softcam/fayam, tcolorbox, tikz) ; ajout de `pifont`, boîtes `constatbox/problembox/verdictbox/interpbox`, commandes `\ok/\fail/\warn`, couleurs `success/warning/failure`.
- Chaque frame contient son titre + `\source{...}` avec les citations prévues + un `[TODO]` ou `[FIGURE]` indiquant le contenu attendu.
- Compilation OK : 44 pages, 440 Ko, 0 erreur (warning MiKTeX non bloquant).
- Suite → meubler Acte I (slides 2-8) en priorité pour valider le style avant de propager.

## 2026-05-09 — Présentation Panorama XAI v2 (canevas Dr LACMOU appliqué)

- Dossier `2026-05-09-panorama-xai-v2/` créé : `BRIEF.md`, `slides.tex`, `figures/softcam-transformer-architecture.svg`, `slides.pdf` compilé (**64 pages, 908 Ko, 0 erreur LaTeX**).
- Préambule hérité de la V1 (Madrid + seahorse), ajout `amssymb` + lib TikZ `calc` ; environnements `ctxbox / probbox / transposbox / limitbox / plusbox` créés pour matérialiser le canevas Dr LACMOU.
- Structure : 12 sections — pourquoi expliquer / contexte FAYAM (avec métriques Phase 1) / cadre canevas / SHAP panorama (LIME → KernelSHAP → TimeSHAP → WindowSHAP → TsSHAP) / **★ZOOM SHAPformer (7 slides)** / transition / CAM (CAM → GradCAM) / **★ZOOM SoftCAM (6 slides)** / synthèse comparative / **★Architecture SoftCAM-Transformer (10 slides : SVG embarqué TikZ, entrées/sorties par bloc, evidence layer Python, équation centrale, ElasticNet, variantes A/B/C, exemple carte C4, hypothèses H1.A--H1.E)** / conclusion / questions / références.
- Slide-discussion ajoutée : *« Peut-on rendre SHAPformer pleinement self-explainable ? »* — discussion FastSHAP, conclusion non sans dénaturer, transition vers SoftCAM.
- SVG d'architecture (`softcam-transformer-architecture.svg`) copié depuis `memoire/03-contribution/figures/`. Schéma TikZ équivalent embarqué dans le slides pour robustesse compilation.
- 9 passes de débogage compilation : virgules dans titres tcolorbox (encadrer en `{...}`), Unicode (★, →, ↔ → équivalents math LaTeX), `\verbatim` → frames marqués `[fragile]`, lib `calc` ajoutée.
