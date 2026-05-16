# DEBRIEF — Explicabilité : Panorama des méthodes XAI (LIME → SHAP → CAM → SoftCAM)

> Rempli a posteriori (2026-05-08) à partir du retour de l'étudiant.

## Retour global

Présentation **non formellement validée** — les méthodes (LIME, KernelSHAP, TimeSHAP, WindowSHAP, TsSHAP, SHAPformer, CAM, GradCAM, SoftCAM) étaient présentées de manière **trop technique**. L'étudiant a eu de la peine à les exposer à l'oral : trop de formalisme, pas assez de fil conducteur pédagogique reliant les méthodes entre elles.

Dr LACMOU a prescrit un **nouveau cadre de présentation** à appliquer dans les séances à venir.

## Nouveau cadre prescrit pour présenter une méthode XAI

À appliquer pour **chaque** méthode présentée dans les futures séances. Quatre points par méthode :

1. **Contexte de la méthode d'explicabilité** — d'où elle vient, dans quel domaine elle est née, à quoi elle a été appliquée à l'origine.
2. **Problèmes que la méthode essaye de résoudre** — la lacune ou le besoin qui a motivé sa création.
3. **Transposabilité** — la méthode est-elle applicable / explicable dans un autre domaine que celui où elle a été conçue ?
4. **Limites de la méthode** — ce qu'elle ne fait pas bien, ce qui motive de chercher mieux.

Et **pour chaque méthode qui suit la précédente** : indiquer explicitement la **plus-value** apportée (qu'est-ce que la nouvelle méthode résout que la précédente ne résolvait pas ?).

L'idée est de construire un récit en chaîne : chaque méthode comble un manque de la précédente, ce qui justifie naturellement l'arrivée de la suivante — et finalement le choix de SoftCAM.

## Critiques précises

| # | Remarque | Suite à donner |
|---|----------|----------------|
| 1 | Méthodes présentées de manière trop technique (formalismes, formules, détails algorithmiques) | Reprioriser le récit pédagogique sur la technique |
| 2 | Pas de fil conducteur reliant les méthodes — chaque méthode présentée comme un bloc isolé | Appliquer le cadre 4 points + plus-value de transition |
| 3 | Difficulté à présenter à l'oral car trop de contenu spécialisé par slide | Simplifier les slides et les ancrer sur le cadre prescrit |

## Action items (avant la prochaine présentation)

- [ ] Reprendre les slides du panorama avec le cadre prescrit (4 points par méthode + plus-value en transition).
- [ ] Pour chaque méthode XAI déjà fichée (LIME, KernelSHAP, TimeSHAP, WindowSHAP, TsSHAP, SHAPformer, CAM, GradCAM, SoftCAM) : extraire le **contexte d'origine**, le **problème résolu**, la **transposabilité**, les **limites** depuis la fiche de lecture.
- [ ] Construire le **fil de plus-values** : LIME → KernelSHAP → TimeSHAP → WindowSHAP → TsSHAP → SHAPformer ; CAM → GradCAM → SoftCAM ; transition SHAP→CAM justifiée par la complémentarité post-hoc/intrinsèque.
- [ ] Réduire le formalisme — préférer un schéma + une phrase à une équation par méthode.

## Pourquoi la prochaine présentation ?

Reprise du panorama XAI selon le cadre Dr LACMOU, mais aussi premiers résultats Phase 1 (4 baselines par cluster, cible H1=C4) à intégrer comme socle empirique. Démontrer que le choix de SoftCAM-Transformer (H1) est la conclusion logique du fil pédagogique des 9 méthodes.
