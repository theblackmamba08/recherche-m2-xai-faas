# Mémoire — redaction

## 2026-05-28 — SVG architecture v4 + notebook sanity check (session 87)

- `redaction/figures/architecture-evidence-layer-v4.svg` créé (720×920 px) : diagramme complet du forward pass Evidence Layer v4 (enc_hidden + dec_output → Linear → M=softmax → bmm → LayerNorm → 1+tanh gate → ⊙ → h_ev → output_projection → Prédiction). Skip connection dec_output → ⊙ visible. Badges NEW sur gate et ⊙.
- Justifications théoriques dans le SVG : Vaswani 2017, SE-Net Hu 2018, GLU Dauphin 2017, Ba 2016, Srivastava 2015.
- Suite → insérer le SVG dans §3.3 du mémoire (Architecture — couche d'évidence).

## 2026-05-28 — PLAN-MEMOIRE.md enrichi avec retours encadreurs + plus-value (session 85)

- `PLAN-MEMOIRE.md` mis à jour : section "Plus-value" (tableau 3 axes vs FAYAM/TimeSHAP/TFT), tableau comparatif XAI §2.7, pattern théorie→application→résultats rendu obligatoire (§3.1.4), §4.2.4 seuil R², §4.3.7 confiance en M (3 pistes), §4.4 évaluation qualitative renforcée (3 exemples typés), checklist 8 points retours encadreurs.
- Suite → soumettre le plan aux encadreurs ; trancher nom modèle + dot product ; commencer Chap4.

## 2026-05-28 — Template Dschang importé + plan de mémoire complet (session 84)

- Template Dschang ajouté dans `redaction/Thesis-Template/` (structure `My-Thesis/` + `Template-Style/`).
- `PLAN-MEMOIRE.md` créé : plan complet 4 chapitres + intro + conclusion, contenu attendu par section, références clés, volume estimé (85–100 pages), ordre de rédaction conseillé, décisions en attente (nom modèle, dot product).
- Suite → soumettre le plan aux encadreurs ; trancher nom modèle + dot product ; commencer Chap4 (résultats déjà disponibles).

## 2026-04-26 — Setup initial

- Squelette créé : `main.tex` (placeholder), `chapitres/`, `biblio/refs.bib` (entrée d'exemple), `figures/`.
- `main.tex` configuré pour `biblatex` + IEEE + français en attendant le template Dschang.
- Suite → cf. [STEPS.md](STEPS.md) : importer template Dschang, configurer LaTeX Workshop dans Antigravity.
