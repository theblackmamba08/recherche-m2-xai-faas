# SPEECH — Panorama des méthodes XAI (LIME → SHAP → CAM → SoftCAM)

> **Cible** : ~20 minutes — environ 30 secondes par slide de contenu, 10–15 secondes pour les slides de transition.
> **Rythme** : ~130 mots/minute. Lire posément, ne pas se presser.
> Les `[PAUSE]` indiquent des moments où regarder le public avant de continuer.

---

## Slide 1 — Page de titre
*(~20 s)*

Bonjour. Aujourd'hui je vous présente un panorama des méthodes d'explicabilité en intelligence artificielle, en progressant de LIME — la méthode fondatrice de 2016 — jusqu'à SoftCAM, un article de mai 2025. L'objectif final est de justifier notre hypothèse de travail pour la prédiction de charge FaaS.

---

## Slide 2 — Plan de la présentation
*(~30 s)*

La présentation est organisée en sept parties. On commence par poser la question de fond : pourquoi l'explicabilité est-elle devenue incontournable ? On revisite ensuite les travaux de FAYAM pour localiser le verrou précis. On fait un tour de la taxonomie des méthodes, on plonge dans la famille SHAP puis la famille CAM, on fait une comparaison synthétique, et on finit par notre choix de voie pour ce projet. [PAUSE]

---

## ▶ Transition — Section 1 : L'explicabilité : pourquoi maintenant ?
*(~10 s)*

Commençons par la première partie : pourquoi l'explicabilité, et pourquoi maintenant ?

---

## Slide 1.1 — Le paradoxe performances / confiance
*(~35 s)*

Les modèles d'apprentissage automatique modernes — ResNets, GPT, Transformers — atteignent des performances remarquables. Mais leur complexité les rend opaques : des millions de paramètres, des décisions non traçables, des comportements inattendus hors distribution. Les utilisateurs finaux ne font pas confiance à des prédictions qu'ils ne comprennent pas. [PAUSE] On a donc un dilemme fondamental : d'un côté la performance, de l'autre la confiance et la déployabilité. Un modèle précis mais inexplicable ne peut tout simplement pas être déployé dans un secteur critique.

---

## Slide 1.2 — Enjeux concrets : réglementation et secteurs critiques
*(~35 s)*

Ce n'est pas qu'un problème académique. L'EU AI Act de 2024 exige la traçabilité des décisions automatisées pour les systèmes à risque élevé. Le RGPD impose un droit à l'explication. Et dans les secteurs concrets : en santé, le médecin doit comprendre pourquoi le modèle dit « anomalie ». En finance, la décision de crédit est soumise à obligation légale. Dans notre cas — le cloud et le FaaS — les opérateurs ont besoin de comprendre les pics de charge pour pouvoir agir rapidement.

---

## Slide 1.3 — Qu'est-ce qu'une bonne explication ?
*(~35 s)*

Qu'est-ce qu'une bonne explication ? Trois dimensions. La **portée** : locale — expliquer une seule prédiction — ou globale, le comportement général du modèle. La **fidélité** : est-ce que l'explication reflète vraiment le raisonnement interne, ou est-ce une approximation ? Enfin l'**interprétabilité** : compréhensible par un non-expert ? La famille SHAP, qu'on va voir tout à l'heure, ajoute trois propriétés axiomatiques : précision locale, missingness, et cohérence. Ces propriétés vont structurer toute notre analyse.

---

## ▶ Transition — Section 2 : Contexte : les travaux de FAYAM
*(~10 s)*

Passons maintenant au contexte précis de notre projet : les travaux de FAYAM.

---

## Slide 2.1 — Prédiction de charge FaaS : le problème métier
*(~35 s)*

Le contexte c'est le FaaS — Function-as-a-Service — un paradigme cloud où les fonctions s'exécutent à la demande. Le défi : les invocations sont imprévisibles, avec des pics soudains et des saisonnalités complexes. L'enjeu opérationnel est de pré-allouer les ressources pour éviter la latence ou le gaspillage. FAYAM propose de prédire le nombre d'invocations futures à partir de l'historique. Leur dataset couvre 18 fonctions FaaS instrumentées, avec 33 clusters de profils de charge identifiés par DBSCAN.

---

## Slide 2.2 — Les deux architectures proposées par FAYAM
*(~30 s)*

FAYAM propose deux architectures. Le premier modèle est un CNN-LSTM hybride — on ne s'y attardera pas, c'est hors de notre périmètre. Le second, c'est notre focus : un `TimeSeriesTransformer` HuggingFace, encodeur-décodeur, avec mécanisme d'attention multi-têtes. [PAUSE] C'est ce modèle que nous allons chercher à expliquer.

---

## Slide 2.3 — Résultats du Transformer sur les datasets de test
*(~25 s)*

FAYAM a évalué les deux modèles sur leurs datasets. Les métriques — sMAPE, RMSE, R², Spearman — seront complétées lors de la phase 1 quand on reproduira les expériences. Ce qu'on sait déjà : le Transformer surpasse le CNN-LSTM sur la majorité des datasets. C'est un modèle compétitif — mais qui a une faille.

---

## Slide 2.4 — Le verrou : de bonnes performances, mais aucune explication
*(~35 s)*

Et c'est là qu'on arrive au cœur du problème. Le Transformer de FAYAM fait de bonnes prédictions. Mais il n'y a aucune explication associée. Quand il prédit un pic de charge dans 6 heures, on ne sait pas pourquoi. Quelles features ont compté ? Quel historique a été utilisé ? [PAUSE] Un opérateur FaaS ne peut pas valider, auditer ni faire confiance à un modèle qui ne justifie pas ses prédictions. La déployabilité dans un secteur critique devient impossible. C'est le verrou scientifique que ce mémoire veut lever.

---

## ▶ Transition — Section 3 : Taxonomie des méthodes d'explicabilité
*(~10 s)*

Avant de présenter les méthodes, faisons un rapide tour de la taxonomie pour se repérer.

---

## Slide 3.1 — Axes de classification
*(~35 s)*

On classe les méthodes selon quatre axes. **Quand ?** Post-hoc — on explique après entraînement —, ante-hoc — le modèle est intrinsèquement interprétable —, ou pendant l'entraînement, c'est le cas de SoftCAM. **Qui peut l'utiliser ?** Les méthodes agnostiques comme LIME et SHAP fonctionnent sur n'importe quel modèle. Les méthodes spécifiques comme CAM sont liées à une architecture. **Quelle portée ?** Locale, globale, ou semi-locale pour TsSHAP. **Quelle tâche ?** La plupart ciblent la classification — la prévision est très sous-représentée dans la littérature.

---

## Slide 3.2 — Positionnement de toutes les méthodes
*(~35 s)*

Ce tableau synthétise le positionnement de toutes les méthodes. La colonne clé, c'est « Prévision ? » : seuls TsSHAP, SHAPformer, et notre extension SoftCAM-Transformer y répondent par oui. La ligne en gras en bas, c'est SoftCAM : la seule méthode intrinsèque, fidélité maximale, compatible Transformer, adaptée à la prévision — mais c'est une extension que **nous** proposons, pas ce que l'article original réalise. [PAUSE] C'est précisément notre contribution scientifique.

---

## ▶ Transition — Section 4 : Famille SHAP
*(~10 s)*

Entrons dans le détail de la famille SHAP, en commençant par son ancêtre direct : LIME.

---

## Slide 4.1a — LIME — Ribeiro et al. (KDD 2016)
*(~35 s)*

LIME — Local Interpretable Model-agnostic Explanations — est publié à KDD 2016. La question posée est simple : comment expliquer la prédiction d'un modèle quelconque, de façon localement fidèle et interprétable ? La méthode apprend un modèle linéaire sparse autour de l'instance à expliquer. On minimise la perte de fidélité locale pondérée par un noyau de proximité, plus la complexité du modèle d'explication. Les features interprétables peuvent être des mots, des super-pixels d'image, ou des features tabulaires.

---

## Slide 4.1b — LIME — Résultats et SP-LIME
*(~30 s)*

Les résultats sont bons : recall supérieur à 90 % sur les features importantes, contre 64 % pour les méthodes greedy et 17 % au hasard. SP-LIME va plus loin en sélectionnant des instances représentatives et non-redondantes pour donner une confiance globale dans le modèle. La limite fondamentale : LIME est **instable** — deux runs donnent deux explications différentes — et surtout pas conçu pour les séries temporelles, où les features ne sont pas indépendantes entre elles.

---

## Slide 4.1c — LIME — Ce que cela nous enseigne
*(~25 s)*

Ce que LIME apporte : le premier cadre model-agnostique rigoureux pour l'XAI post-hoc, et la notion de représentation interprétable. Ce qui manque : des garanties théoriques solides et l'adaptation au domaine temporel. Un an après, Lundberg et Lee vont unifier LIME et d'autres méthodes sous un cadre basé sur la théorie des jeux coopératifs. C'est la naissance de SHAP.

---

## Slide 4.2a — KernelSHAP — Lundberg & Lee (NeurIPS 2017)
*(~35 s)*

SHAP repose sur les valeurs de Shapley, issues de la théorie des jeux. L'idée : mesurer la contribution marginale de chaque feature, moyennée sur **toutes** les coalitions possibles. C'est la seule solution satisfaisant simultanément les trois axiomes : précision locale, missingness — une feature absente contribue zéro —, et cohérence. KernelSHAP reformule ce calcul comme une régression pondérée, ce qui en fait exactement LIME avec le bon noyau — mais cette fois avec des garanties théoriques solides.

---

## Slide 4.2b — KernelSHAP face aux séquences temporelles
*(~30 s)*

Le problème de KernelSHAP face aux séquences : il attribue de l'importance aux features du pas de temps **courant** uniquement. Tout l'historique passé de la séquence est ignoré. Pour notre Transformer, qui lit des séquences entières de contexte, c'est rédhibitoire. [PAUSE] Il faut adapter SHAP au domaine séquentiel — c'est l'objet de TimeSHAP.

---

## Slide — Pause : Pourquoi SHAP s'impose face à LIME
*(~25 s)*

Petite pause pour consolider. LIME satisfait seulement la précision locale. SHAP, lui, satisfait les trois axiomes, et propose en plus TreeSHAP — exact en temps polynomial pour les arbres. Conclusion : la famille SHAP est le **standard théorique** de l'XAI post-hoc. Toutes les méthodes suivantes sont des extensions SHAP adaptées aux séries temporelles.

---

## Slide 4.4a — TimeSHAP — Bento et al. (KDD 2021)
*(~35 s)*

TimeSHAP, publié à KDD 2021, étend KernelSHAP au domaine séquentiel avec des perturbations bidimensionnelles. Pour une matrice séquence de dimension d × l, ils définissent trois types de perturbations : par features, par événements temporels, ou par cellules. L'innovation majeure est l'**élagage temporel** : au lieu d'explorer les 2^l coalitions, on parcourt la séquence de la fin vers le début et on groupe les événements passés dont l'importance est sous un seuil eta. La médiane passe de 138 événements à seulement 14.

---

## Slide — TimeSHAP : Résultats et limites
*(~30 s)*

Sur la fraude bancaire, avec un GRU entraîné sur 20 millions d'instances, TimeSHAP identifie que l'événement le plus récent compte pour 41 % du score, et révèle un pattern enrollment-login-transaction. Il détecte même un biais : corrélation entre âge et taux de faux positifs. La limite pour notre projet : TimeSHAP est conçu pour la **classification**, pas pour la prévision. Pas adapté à FAYAM tel quel.

---

## Slide 4.4b — TimeSHAP : Élagage temporel et explications cellules
*(~25 s)*

Pour le détail : l'algorithme d'élagage groupe les événements dont le poids absolu est inférieur à eta, ce qui réduit la complexité de façon significative. On obtient trois niveaux d'explication : événements — quels instants ont compté —, features — quelles variables —, et cellules — quel couple feature-instant. S'applique en boîte noire au Transformer FAYAM, mais requiert une adaptation pour la tâche de prévision.

---

## Slide 4.5 — WindowSHAP — Nayebi et al. (2023)
*(~30 s)*

WindowSHAP réduit le coût de KernelSHAP en calculant des valeurs Shapley par **fenêtre temporelle** plutôt que par pas de temps. Trois variantes : stationnaire, glissante, ou dynamique — la plus intéressante, qui ajuste automatiquement la taille des fenêtres selon leur importance et évite toute hypothèse sur les instants critiques. Résultat : moins 80 % de CPU par rapport à KernelSHAP. Limite : validé uniquement en classification, pas de librairie Python officielle.

---

## Slide 4.6a — TsSHAP — Raykar et al., IBM Research (2023)
*(~35 s)*

TsSHAP — IBM Research, 2023 — est la **seule méthode SHAP conçue nativement pour la prévision**. C'est une distinction fondamentale : toutes les méthodes vues jusqu'ici expliquent des scores de classification. TsSHAP explique des prévisions. Son pipeline en quatre étapes : backtesting pour générer des prévisions historiques, extraction de features interprétables — lags, saisonnalités, date —, entraînement d'un surrogate XGBoost qui imite le forecaster, puis TreeSHAP sur ce surrogate.

---

## Slide — TsSHAP : Features interprétables
*(~25 s)*

TsSHAP supporte trois portées d'explication : locale pour une prédiction, semi-locale pour un horizon, et globale pour toute la série. Les features interprétables couvrent les lags, les lags saisonniers, les rolling windows — max, min, moyenne —, les features de date et les jours fériés. C'est la référence en prévision, mais la fidélité est approximative puisqu'on passe par un surrogate XGBoost qui peut mal imiter un Transformer.

---

## Slide 4.6b — TsSHAP : Résultats et limites
*(~30 s)*

Sur les datasets de test, les explications semi-locales sont toujours plus fidèles que les locales. L'exemple jeans-sales-weekly est parlant : SeasonalNaive dépend du lag à 52 semaines, Prophet est dominé par la feature externe discount, XGBoost par le lag immédiat. Les limites pour nous : TsSHAP est **univarié**, donc on devra l'appliquer série par série sur les 18 fonctions FaaS. C'est notre hypothèse **H2 prioritaire** si H1 échoue.

---

## Slide 4.7 — SHAPformer — Hertel et al., KIT (déc. 2025)
*(~30 s)*

SHAPformer — KIT Karlsruhe, décembre 2025, donc très récent — calcule des valeurs SHAP **exactes** pour les Transformers de prévision, sans Monte Carlo, en moins d'une seconde. Le mécanisme : on groupe les features en N groupes, on entraîne avec des entrées masquées, et à l'inférence on évalue les 2^N sous-ensembles via attention — zéro tirage aléatoire. Sur le benchmark TransnetBW : 0,60 secondes contre 484 secondes pour le Permutation Explainer. 800 fois plus rapide, RMSE comparable.

---

## ▶ Transition — Section 5 : Famille CAM : de l'activation à l'évidence
*(~10 s)*

On quitte la famille SHAP pour entrer dans la famille CAM, qui va nous amener directement à SoftCAM.

---

## Slide 5.1 — CAM — Zhou et al. (CVPR 2016)
*(~30 s)*

CAM — Class Activation Maps, CVPR 2016. Pour un CNN classifieur, on calcule une carte de chaleur en combinant les feature maps de la dernière couche convolutive, pondérées par les poids de la couche fully-connected. La carte montre quelles régions activent la décision. La contrainte : il faut un Global Average Pooling avant la couche FC — architecture imposée. C'est la **première méthode** à localiser ce que le modèle regarde, sans back-propagation ni perturbation.

---

## Slide 5.2 — GradCAM — Selvaraju et al. (ICCV 2017)
*(~30 s)*

GradCAM — ICCV 2017 — généralise CAM à toute architecture CNN, sans contrainte GAP. L'astuce : on utilise les gradients du score de classe par rapport aux activations d'une couche pour pondérer les feature maps. Ça lève la contrainte architecturale. Mais GradCAM reste post-hoc : un back-pass par classe, résolution limitée à la feature map cible, et méthode sensible aux gradients saturés. La famille CAM a besoin d'une refonte.

---

## Slide 5.3a — SoftCAM : Architecture — Djoumessi & Berens (arXiv, mai 2025)
*(~35 s)*

SoftCAM — mai 2025 — part d'un constat différent. Les méthodes post-hoc approximent le raisonnement **après coup**. Elles sont sensibles aux perturbations et infidèles par construction. SoftCAM propose une modification **minimale** de l'architecture : on remplace la couche FCL par une convolution 1×1 — la class-evidence layer — suivie d'un average pool et d'un softmax. La sortie est une carte d'évidence A dans R^(M × N × C). [PAUSE] Un seul forward pass pour toutes les classes. Aucun paramètre supplémentaire.

---

## Slide 5.3b — SoftCAM : Régularisation ElasticNet
*(~30 s)*

La régularisation ElasticNet est ce qui rend la carte d'évidence vraiment interprétable. La loss est augmentée de deux termes : L1-Lasso pour la parcimonie, L2-Ridge pour le lissage. SoftCAM sparse donne des cartes focales précises — utile pour des lésions localisées. SoftCAM dense donne des cartes lissées — utile quand on veut capturer de grandes régions. ElasticNet est le compromis adaptatif : on contrôle le trade-off via les hyperparamètres lambda.

---

## Slide 5.3c — SoftCAM : Résultats expérimentaux
*(~30 s)*

Les résultats sur trois datasets médicaux sont convaincants. La modification architecturale n'altère pas la précision : VGG-16 passe de 0,938 à 0,942 AUC, ResNet-50 de 0,999 à 1,000. Côté explicabilité : Sparse SoftCAM ResNet surpasse **toutes** les méthodes post-hoc en top-k precision — GradCAM, Integrated Gradients, Guided Backprop, ScoreCAM, LayerCAM. Et côté coût : un seul forward pass contre un back-pass ou N forward passes pour les concurrents.

---

## Slide 5.3d — SoftCAM : Perspective et lien avec notre projet
*(~35 s)*

Les limites identifiées par les auteurs : résolution coarse des feature maps, évaluation uniquement sur CNN 2D en imagerie médicale, et le trade-off lambda qui est task-specific. Mais dans leurs perspectives, page 10, ils écrivent explicitement : *« In the future, we could explore the integration of SoftCAM with other standard architectures like ViT. »* [PAUSE] C'est exactement notre H1. Notre transposition : remplacer la projection finale du décodeur du TimeSeriesTransformer par une couche d'évidence temporelle, produisant un tenseur predictionLength × contextLength, avec ElasticNet pour la sparsité temporelle.

---

## ▶ Transition — Section 6 : Comparaison synthétique
*(~10 s)*

On a maintenant tous les éléments. Faisons la comparaison synthétique.

---

## Slide 6.1 — Tableau comparatif
*(~25 s)*

Ce tableau résume toutes les méthodes sur sept critères. La lecture clé c'est la colonne « Prévision ? » : seuls TsSHAP, SHAPformer, et notre extension SoftCAM-Transformer y répondent oui. SoftCAM est la seule méthode **intrinsèque** — fidélité maximale, un forward pass — mais son extension au Transformer de prévision, c'est nous qui la proposons. L'astérisque le rappelle.

---

## Slide 6.2 — Lecture du tableau : aucune méthode n'est parfaite pour notre cas
*(~30 s)*

La lecture révèle une lacune réelle dans la littérature. LIME et KernelSHAP ne couvrent pas les séries temporelles. TimeSHAP et WindowSHAP ne font pas de prévision. TsSHAP fait de la prévision mais uniquement de façon univariée avec un surrogate. SHAPformer requiert un réentraînement. SoftCAM original ne vise que les CNN. [PAUSE] Ce qui manque : une méthode adaptée aux Transformers, pour la prévision, à fidélité intrinsèque, sur séries multivariées, sans surcoût à l'inférence. C'est précisément ce vide que notre H1 cherche à combler.

---

## ▶ Transition — Section 7 : Retour sur FAYAM : quelle voie choisir ?
*(~10 s)*

Pour finir, retournons sur FAYAM et voyons concrètement comment on se positionne.

---

## Slide 7.1 — Architecture TimeSeriesTransformer : la couche cible
*(~35 s)*

L'architecture du TimeSeriesTransformer HuggingFace est un encodeur-décodeur. L'encodeur lit le contexte passé. Le décodeur prédit l'horizon futur. La **couche cible** de H1, c'est la projection finale du décodeur — celle qui projette de l'espace d_model vers la distribution de prédiction. C'est là qu'on insère la couche d'évidence temporelle. Le résultat espéré est un tenseur predictionLength × contextLength. Le risque identifié : dans un CNN, la convolution 1×1 préserve les positions spatiales. Dans un Transformer, le self-attention les mélange. L'extension n'est pas triviale.

---

## Slide 7.2 — Notre grille de décision
*(~35 s)*

Notre grille de décision. **H1 en prioritaire** : modifier la projection finale du décodeur, ajouter la loss ElasticNet sur la carte d'évidence, réentraîner le Transformer. Résultat espéré : carte intrinsèque, un forward pass, fidélité maximale. Risque : incompatibilité du principe CNN avec les Transformers. **H2 en repli** : TsSHAP — surrogate XGBoost sur les backtests FAYAM, applicable sans modifier l'architecture — ou SHAPformer si on accepte un réentraînement avec masked attention. H2 est moins ambitieux mais robustement justifié par la littérature.

---

## Slide 7.3 — Notre hypothèse prioritaire : H1 — SoftCAM-Transformer
*(~30 s)*

La contribution scientifique visée par H1 : **première transposition documentée d'un schéma SoftCAM-like à un Transformer de séries temporelles**, avec étude différentielle par cluster DBSCAN sur les 33 profils de charge de FAYAM. Le critère de bascule vers H2 est clair : si l'adaptation ne converge pas ou dégrade significativement la performance prédictive, on bascule sans attendre. H3 — les attention weights — sert à valider la cohérence des explications obtenues, quel que soit le chemin pris.

---

## Slide 7.4 — Questions ouvertes et prochaines étapes
*(~30 s)*

Trois questions méthodologiques restent ouvertes. Comment définir l'équivalent d'une class evidence map pour la **régression** multi-horizons ? Quel est l'équivalent temporel des super-pixels CNN — un patch temporel, un lag, un cluster DBSCAN ? Comment évaluer la fidélité sur une tâche de prévision — comprehensiveness et sufficiency devront être adaptés. Les prochaines étapes concrètes : récupérer le code FAYAM, agréger le dataset, entraîner le Transformer avec `output_attentions=True`, cartographier la projection du décodeur, et chiffrer les métriques.

---

## Slide — Merci / Questions
*(~15 s)*

Voilà pour ce panorama. On est passés de LIME en 2016 jusqu'à SoftCAM en 2025, en identifiant à chaque étape ce qui manque pour notre cas. Je suis disponible pour vos questions.

---

## Slides Références
*(~10 s — ne pas lire, juste mentionner)*

Les références sont disponibles. Je peux les transmettre si vous souhaitez approfondir un article en particulier.

---

## Notes pratiques

| Slide | Temps cible | Alerte si dépassé |
|-------|-------------|-------------------|
| Titre + TOC | ~50 s | > 1 min |
| Section 1 (3 slides + transition) | ~1 min 45 s | > 2 min |
| Section 2 (4 slides + transition) | ~2 min 15 s | > 2 min 30 s |
| Section 3 (2 slides + transition) | ~1 min 30 s | > 2 min |
| Section 4 (14 slides + transition) | ~7 min | > 8 min |
| Section 5 (6 slides + transition) | ~3 min | > 3 min 30 s |
| Section 6 (2 slides + transition) | ~1 min 15 s | > 1 min 30 s |
| Section 7 (5 slides + transition) | ~2 min 30 s | > 3 min |
| Références | ~15 s | — |
| **Total estimé** | **~20 min** | |

**Conseils** :
- Sur les slides de transition automatiques, ne pas rester silencieux : utiliser la phrase de transition prévue.
- Les slides SHAP avec beaucoup de formules (4.2a, 4.4a, 5.3b) : ne pas lire les formules, les pointer et expliquer l'intuition.
- Le tableau 6.1 : pointer les colonnes, pas les lignes.
- Si vous sentez que vous avez du retard à la section 4 : les slides 4.4b (élagage) et TsSHAP-features peuvent être abrégés à 15 s chacun.
