# BRIEF — Panorama XAI v2 (canevas Dr LACMOU + zooms SHAPformer/SoftCAM + architecture SoftCAM-Transformer)

- **Date** : 2026-05-09 (à confirmer avec Dr LACMOU)
- **Audience** : Dr LACMOU ZEUTOUO Jerry (Pr KENGNE TCHENDJI si présent)
- **Durée estimée** : 50-60 min + questions
- **Format** : Beamer (~50-55 diapositives) — thème Madrid, couleurs seahorse
- **Contexte** : suite directe de la présentation du 28/04/2026 (panorama XAI complet, mais jugée **trop technique** par Dr LACMOU). Cette nouvelle version applique le **canevas en 4 points prescrit** (cf. [DEBRIEF 28/04](../2026-04-28-explicabilite-panorama-methodes/DEBRIEF.md) et [DECISIONS.md — 2026-05-08](../../memoire/00-meta/DECISIONS.md)).

## Objectif

Produire un récit pédagogique en chaîne où chaque méthode XAI est présentée selon le **même canevas** (contexte → problèmes résolus → transposabilité → limites), avec une **plus-value de transition** explicite vers la méthode suivante. Le récit aboutit naturellement à **SoftCAM-Transformer (H1)** comme conclusion logique du parcours.

## Ce qui change vs présentation 28/04/2026

| Aspect | 28/04 (V1) | 09/05 (V2) |
|--------|------------|------------|
| Structure des méthodes | Bloc isolé par méthode (problème, méthode, limites) | **Canevas 4 points uniforme** + plus-value de transition |
| Niveau technique | Formules, équations, détails algorithmiques | Plus pédagogique, schémas et phrases courtes |
| SHAPformer | 1 slide | **Zoom 6-7 slides** : architecture détaillée (masking d'attention, training, inférence 2^N) |
| SoftCAM | 4 slides (architecture, ElasticNet, résultats, lien H1) | **Zoom 6-7 slides** : architecture, ElasticNet visualisé, résultats, positionnement intrinsèque |
| Architecture SoftCAM-Transformer (H1) | 1 slide schéma TikZ + 1 slide grille décision | **Section dédiée 8-10 slides** : architecture complète (SVG), entrées/sorties par bloc, evidence layer, variantes A/B/C, équation centrale |
| Question SHAPformer self-explainable | Non traitée | **Slide dédiée** : peut-on l'adapter ? Discussion (FastSHAP), conclusion non sans dénaturer |
| Premiers résultats Phase 1 (C4) | Non traités | À mentionner brièvement comme socle empirique |

## Plan de la présentation (12 sections)

1. **Pourquoi expliquer ?** (2 slides) — paradoxe perf/confiance
2. **Contexte FAYAM + faille XAI** (3 slides) — Transformer FAYAM, verrou XAI
3. **Cadre du panorama** (2 slides) — annonce du **canevas 4 points** + taxonomie XAI
4. **Famille SHAP — panorama avec canevas** (12-14 slides) — LIME → KernelSHAP → TimeSHAP → WindowSHAP → TsSHAP, chacune en canevas court + plus-value de transition
5. **★ ZOOM SHAPformer** (6-7 slides) — architecture détaillée, training masked inputs, inférence 2^N coalitions, résultats, positionnement *semi-intrinsèque*, slide-discussion "peut-on rendre SHAPformer 1-pass ?"
6. **Transition SHAP → CAM** (1 slide) — pourquoi explorer la famille intrinsèque
7. **Famille CAM** (4-5 slides) — CAM → GradCAM en canevas court + plus-value vers SoftCAM
8. **★ ZOOM SoftCAM** (6-7 slides) — architecture intrinsèque, evidence layer, ElasticNet, résultats CNN, positionnement *self-explainable pleinement*
9. **Synthèse SHAPformer vs SoftCAM** (2 slides) — tableau, axe semi-intrinsèque ↔ intrinsèque
10. **★ Architecture SoftCAM-Transformer (H1, notre contribution)** (8-10 slides) — motivation, architecture complète (SVG/TikZ), entrées-sorties par bloc, evidence layer en détail, variantes A/B/C, exemple d'explication sur C4, hypothèses opératoires H1.A–H1.E
11. **Conclusion + perspectives** (2-3 slides) — récit récapitulatif, perspective "vers un SHAPformer self-explainable ?" (chapitre Discussion)
12. **Questions** (1 slide)
13. **Références** (auto-générées)

## Points clés à faire passer

1. **Le canevas Dr LACMOU est appliqué systématiquement** : visible dès la slide d'annonce (Section 3) et à chaque méthode (4 boîtes contexte/problèmes/transposabilité/limites + une box plus-value).
2. **SHAPformer = climax du post-hoc** : SHAP exact en quasi-temps-constant, mais reste 2 phases (prédiction + explication). Position *semi-intrinsèque*.
3. **SoftCAM = climax intrinsèque** : explication produite *pendant* le calcul de prédiction, fidèle par construction. Position *self-explainable pleinement*.
4. **SoftCAM-Transformer (H1) = synthèse logique** : combine la rigueur architecturale de SoftCAM (intrinsèque, ElasticNet) avec la cible Transformer prévision FaaS.
5. **Critère de bascule H1 → H2** : R²≥0.30, Spearman≥0.85 conservés sur C4 ; sinon repli SHAPformer/TsSHAP.

## Méthodes par canevas à 4 points — checklist contenu

Pour chaque méthode, les 4 boîtes doivent contenir :

- **Contexte** : auteur, année, conférence, domaine d'application *original* (texte/image/tabulaire/séquentiel/forecast)
- **Problèmes résolus** : la lacune comblée à sa création
- **Transposabilité** : applicable à la prévision multivariée FaaS ? oui/partiel/non + pourquoi
- **Limites** : ce qui justifie la méthode suivante dans la chaîne

Et une boîte de transition : **"Plus-value de la méthode suivante"** — qu'apporte-t-elle de plus ?

## Statut

Présentation à compiler après finalisation. Cible : démarrage Phase 2 (J1 PLAN-ETUDE-ARCHITECTURE) en parallèle de la préparation finale des slides.

## Métriques FAYAM disponibles (différence vs 28/04)

Phase 1 terminée ⇒ slide 2.3 contient maintenant les **vraies métriques** :
- Baseline FAYAM-Transformer mixte 19 séries : MASE=1.38, sMAPE=1.45, R²=-1.26
- Baseline FAYAM-Transformer dédié C4 : R²=0.37, Spearman=0.92, sMAPE=0.22

(Cf. [run 2026-05-05_baseline-fayam-local-clusters](../../code/experiments/runs/2026-05-05_baseline-fayam-local-clusters/run.md) et runs par cluster.)
