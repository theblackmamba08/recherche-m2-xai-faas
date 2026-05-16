# Mémoire — 01-litterature

## 2026-04-26

- Dossier créé. Structure `articles/` + `fiches/` mise en place.
- Mémoire FAYAM ajouté (renommé `2024_FAYAM_FaaS-prediction.pdf`) et fiché.
- BibTeX-key à reporter dans `redaction/biblio/refs.bib` : `FAYAM2024Prediction`.

## Fiches de lecture

- 2026-04-26 — fiché : `2024_FAYAM_FaaS-prediction.pdf` → [`fiches/2024_FAYAM_FaaS-prediction.md`](fiches/2024_FAYAM_FaaS-prediction.md) (pertinence 5/5 — baseline architecturale)
- 2026-04-27 — fiché : `2025_Djoumessi_SoftCAM.pdf` → [`fiches/2025_Djoumessi_SoftCAM.md`](fiches/2025_Djoumessi_SoftCAM.md) (pertinence 5/5 — fondation conceptuelle de la nouvelle H1 ; à transposer du CNN au TimeSeriesTransformer)

- 2026-04-28 — fiché : `2021_Bento_TimeSHAP.pdf` → [`fiches/2021_Bento_TimeSHAP.md`](fiches/2021_Bento_TimeSHAP.md) (pertinence 4/5 — fondation de H2 repli ; baseline post-hoc directement applicable au Transformer FAYAM)
- 2026-04-28 — fiché : `2023_Nayebi_WindowSHAP.pdf` → [`fiches/2023_Nayebi_WindowSHAP.md`](fiches/2023_Nayebi_WindowSHAP.md) (pertinence 3/5 — alternative à TimeSHAP pour longues séquences ; Dynamic WindowSHAP évite l'hypothèse anciens événements peu importants)
- 2026-04-28 — fiché : `2023_Raykar_TsSHAP.pdf` → [`fiches/2023_Raykar_TsSHAP.md`](fiches/2023_Raykar_TsSHAP.md) (pertinence 5/5 — **seule méthode SHAP conçue pour la prévision** ; H2 repli prioritaire si H1 échoue ; surrogate XGBoost + TreeSHAP sur backtested forecasts)
- 2026-04-28 — fiché : `2025_Hertel_SHAPformer.pdf` → [`fiches/2025_Hertel_SHAPformer.md`](fiches/2025_Hertel_SHAPformer.md) (pertinence 5/5 — SHAP exact sampling-free pour Transformers de prévision ; 800× plus rapide que Permutation Explainer ; code GitHub disponible ; preprint déc. 2025)
- 2026-04-28 — fiché : `2016_Ribeiro_LIME.pdf` → [`fiches/2016_Ribeiro_LIME.md`](fiches/2016_Ribeiro_LIME.md) (pertinence 2/5 — référence fondatrice XAI post-hoc ; à citer dans l'état de l'art comme point de départ avant SHAP ; pas adapté séries temporelles ni prévision)

## 2026-04-27 — Pivot stratégique XAI

- Article SoftCAM (Djoumessi & Berens, mai 2025) intégré et fiché.
- Conséquence : refonte des hypothèses (cf. [`../00-meta/DECISIONS.md`](../00-meta/DECISIONS.md) et [`../00-meta/ROADMAP.md`](../00-meta/ROADMAP.md)).
- BibTeX-keys à reporter dans `redaction/biblio/refs.bib` : `FAYAM2024Prediction`, `Djoumessi2025SoftCAM`.

## 2026-05-16 — Fiche TFT (Lim et al. 2021)

- Fiche rédigée sans PDF (connaissances de formation) : [`fiches/2019_Lim_TFT.md`](fiches/2019_Lim_TFT.md) (pertinence 5/5)
- TFT = concurrent direct de H1 : Transformer TS interprétable (VSN + IMHA), déployé en production (Google), IJF 2021.
- Argument clé contre TFT : IMHA partage les valeurs → poids d'attention non fidèles par construction (critique Jain & Wallace 2019). H1 produit une décomposition linéaire exacte.
- BibTeX-key : `Lim2021TFT`.
