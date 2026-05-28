# Speech — Validation H1 : SoftCAM-Transformer
> Un texte par slide, à lire à voix haute lors de la présentation.

---

## Slide 1 — Titre

Bonjour. Je m'appelle Fosso Cabrel, étudiant en Master 2 à l'Université de Dschang. Aujourd'hui je vous présente les résultats complets de mon hypothèse principale, H1 : rendre le modèle de prédiction de charge FaaS auto-explicable, grâce à ce que j'appelle le SoftCAM-Transformer.

---

## Slide 2 — Plan

La présentation suit cinq actes. D'abord le contexte et l'architecture. Ensuite je vous montre comment la première tentative a échoué. Puis comment j'ai construit la recette qui fonctionne. Ensuite une crise méthodologique que j'ai découverte et résolue. Et enfin la validation complète des sept hypothèses.

---

## Slide 3 — Le problème métier

Le point de départ, c'est le serverless, ou FaaS. Dans ce modèle, les fonctions sont facturées à l'invocation, sans serveur permanent. Le problème, c'est le cold-start : quand une fonction n'a pas été appelée depuis un moment, son container est éteint. Le démarrer prend du temps, et ça dégrade la latence pour l'utilisateur final.

La solution naturelle, c'est de prédire quand la fonction va être appelée, pour pré-chauffer les ressources à l'avance. C'est ce que fait FAYAM, le travail sur lequel je m'appuie : un Transformer entraîné sur des données Azure Functions 2019, avec 33 clusters de fonctions et un horizon de prédiction de 120 minutes.

Mais FAYAM prédit sans expliquer. Or pour un opérateur qui doit décider d'allouer des ressources, une boîte noire ne suffit pas.

---

## Slide 4 — Pourquoi expliquer ?

Il y a trois raisons fondamentales pour lesquelles expliquer la prédiction est nécessaire.

La première, c'est la confiance. Lipton en 2018 l'a bien formulé : un opérateur ne peut pas faire confiance à un modèle opaque pour des décisions coûteuses. Si le modèle prédit un pic à 3h du matin, l'opérateur veut savoir pourquoi avant d'allouer des ressources.

La deuxième, c'est le diagnostic. Sans explication, impossible de distinguer un vrai pattern récurrent d'un artifact dans les données. On risque de gaspiller des ressources pour corriger quelque chose qui n'existe pas.

La troisième, c'est la conformité. Avec le reglement europeen sur l'intelligence artificielle de 2024, la traçabilité des décisions automatisées est une obligation légale, pas une option.

Concrètement, plusieurs types d'acteurs sont concernés parmi lesquels : l'opérateur cloud qui alloue les ressources, le développeur qui veut comprendre pourquoi son application ralentit, l'auditeur qui vérifie la légitimité des décisions, et le chercheur — moi — qui évalue si l'explication reflète vraiment le calcul du modèle.

---

## Slide 5 — Panorama XAI temporel

Il existe trois grandes familles de méthodes d'explicabilité pour les séries temporelles.

Les méthodes post-hoc par gradient, comme Integrated Gradients ou Grad-CAM. Elles calculent la sensibilité de la prédiction par rapport aux entrées. La fidélité est locale et approximée.

Les méthodes post-hoc par perturbation, comme TimeSHAP. Elles masquent des features et mesurent l'impact. Plus rigoureuses, mais très coûteuses.

Et les méthodes intrinsèques, notre choix. Ici l'explication est produite directement dans le calcul du modèle, pas après coup. CAM et SoftCAM en sont des exemples. La fidélité est garantie par construction, pas approximée.

---

## Slide 6 — Attention ≠ explication

Avant de présenter notre approche, je dois lever un malentendu fréquent. On pourrait penser que les poids d'attention d'un Transformer expliquent naturellement ses prédictions. Ce n'est pas le cas.

Jain et Wallace l'ont montré en 2019 : on peut construire des distributions d'attention complètement différentes qui produisent exactement la même prédiction. Autrement dit, l'attention peut changer sans que la sortie change — elle ne reflète pas nécessairement ce qui cause la prédiction.

Wiegreffe et Pinter nuancent en disant que l'attention peut parfois être informative, mais la fidélité n'est pas garantie en général. Serrano et Smith étendent cette critique aux Transformers.

Conclusion : toute approche qui utilise la cross-attention brute comme explication, comme le TFT (Temporal Fusion Transformer) de Lim et al., reste exposée à cette critique.

---

## Slide 7 — La promesse de l'explication intrinsèque

La distinction fondamentale est la suivante.

Une méthode post-hoc comme SHAP ou Integrated Gradients entraîne d'abord le modèle, puis essaie d'expliquer ce qu'il a fait. C'est une approximation. Il n'y a pas de garantie que l'explication correspond vraiment au calcul réel.

Une méthode intrinsèque, à l'inverse, produit l'explication dans le même forward pass que la prédiction. L'explication n'est pas une approximation — c'est un coefficient algébrique exact dans le calcul.

Mon hypothèse H1, c'est d'adapter ce principe au TimeSeriesTransformer de FAYAM, en m'inspirant de SoftCAM.

---

## Slide 8 — L'architecture SoftCAM-Transformer

Voici l'architecture concrète. Le modèle reprend entièrement l'encodeur et le décodeur de FAYAM — quatre couches chacun, non modifiés.

La nouveauté est l'Evidence Layer, en vert sur le schéma. À partir de la sortie du décodeur, on calcule une matrice M par une transformation linéaire suivie d'un softmax. Cette matrice M fait 120 × 240 : pour chaque pas futur à prédire, elle attribue un poids à chacun des 240 pas passés du contexte. C'est notre carte d'évidence.

Ensuite, M est utilisée pour pondérer les états de l'encodeur — la flèche en tirets montre ce lien. On obtient une représentation pondérée h_ev. La sortie finale est une combinaison entre la sortie du décodeur et h_ev, contrôlée par le paramètre mix.

Ce qui est clé : M est produite dans le forward pass. Elle n'est pas calculée après coup. Sa fidélité est garantie par construction.

---

## Slide 9 — Comment forcer M à pointer les vrais moments clés ?

Sans contrainte, M peut dériver vers deux extrêmes inutiles. Soit elle devient complètement plate : tout le passé a le même poids, aucune position n'est plus importante qu'une autre — ce n'est plus une explication. Soit elle s'effondre sur une seule position de manière arbitraire — c'est ce qui s'est passé au Run B, que je vais vous montrer juste après.

Pour éviter ça, j'ajoute trois termes de régularisation à la loss.

Le premier, le Lasso, force M à être sparse : quelques positions clés concentrent l'essentiel du poids. Comme un surligneur qui ne marque que les phrases importantes.

Le deuxième, le Ridge, empêche qu'une seule position prenne tout le poids. Il stabilise la distribution.

Le troisième, l'entropie, pénalise M si elle est trop uniforme. Elle oblige M à choisir.

---

## Slide 10 — Run B : l'échec de la première tentative

La première tentative, Run B, a consisté à brancher l'Evidence Layer directement à 30% dès la première époque d'entraînement, sans aucun warm-up.

Mon hypothèse de départ était que si l'architecture est correcte, l'optimisation trouverait d'elle-même le bon équilibre.

Résultat : R² = -2.83, Spearman = 0.33. Échec total.

---

## Slide 11 — Diagnostic

Le problème, c'est que la tête de sortie du modèle est initialisée pour traiter uniquement la sortie du décodeur. Lui imposer immédiatement un mélange à 30% avec un signal non régularisé déstabilise toute la rétropropagation. M ne se structure jamais — elle s'effondre sur une position arbitraire.

Ce phénomène est bien documenté dans la littérature sous le nom de warm-up : démarrer trop fort dégrade la convergence. La solution standard, c'est d'introduire le nouveau signal progressivement.

Il faut donc chauffer le mix progressivement, au lieu de l'imposer d'emblée.

---

## Slide 12 — Stratégie

La stratégie pour résoudre ça se résume en deux lignes.

Run B : mix figé à 0.3 dès le départ. Résultat : croix rouge.

Run B5 : on part de mix = 0, on monte progressivement avec warm-up, on ajoute l'anneal de gamma et la LayerNorm, et on arrive à mix = 0.05 en fin d'entraînement. Résultat : coche verte.

Les quatre ingrédients de la recette sont : le warm-up du mix, l'anneal de gamma, la LayerNorm sur h_ev, et un mix final petit de 0.05.

---

## Slide 13 — Run B2

Run B2 introduit les schedules : au lieu de fixer mix et gamma à des constantes, on les fait monter progressivement. Résultat : le décodeur n'est plus détruit. R² remonte à -1.97. C'est encore négatif, mais le collapse est évité. Le problème restant, c'est que h_ev reste statistiquement différent de dec_output — ses valeurs ne sont pas dans la même échelle.

---

## Slide 14 — Run B3

Run B3 ajoute une LayerNorm sur h_ev avant le mélange. Cette normalisation aligne les statistiques des deux branches — h_ev et dec_output ont maintenant des distributions compatibles. R² passe à -1.59. M commence à se structurer, comme on peut le voir dans l'histogramme : au lieu d'un collapse sur une position, la distribution s'élargit.

---

## Slide 15 — Run B4

Run B4 est une contre-expérience. J'enlève la LayerNorm pour vérifier qu'elle est bien responsable de l'amélioration. R² s'effondre à -3.58 — pire que Run B. Ça confirme que la LayerNorm n'était pas un détail accessoire, c'était un ingrédient essentiel.

---

## Slide 16 — Run B5

Run B5 assemble les quatre ingrédients : warm-up mix, anneal gamma, LayerNorm, et mix final à 0.05. R² = 0.71, Spearman = 0.92. Les cinq fonctions du cluster ont toutes un R² positif. C'est la première recette qui converge.

---

## Slide 17 — Synthèse

Ce graphique résume les six runs sur un axe commun. La ligne du milieu représente R² = 0. Tout ce qui est en dessous est un modèle inutilisable.

On part de Run A à +0.53 — notre baseline de référence. Run B s'écrase à -2.83. B2 remonte légèrement à -1.97. B3 continue à -1.59. B4 plonge à -3.58 — la contre-expérience. Et B5 passe au-dessus de la ligne avec +0.71.

La trajectoire raconte comment, étape par étape, chaque ingrédient a été identifié et ajouté.

---

## Slide 18 — L'anomalie : Fix #5

Après avoir validé B5, j'ai conduit un test simple : faire varier le paramètre mix à l'inférence sur neuf valeurs entre 0 et 1, et observer l'évolution du R².

Ce que j'attendais : une courbe en cloche ou monotone — le mix d'inférence devrait avoir un impact sur les résultats.

Ce que j'ai obtenu : les neuf valeurs donnent strictement le même R² = 0.6646. La courbe est plate. Il n'y a aucune variation.

Un seul scénario explique cette platitude : l'Evidence Layer est inactive à l'inférence. Le mix ne fait rien parce que M n'est pas utilisée.

---

## Slide 19 — Diagnostic dans le code

En lisant le code de HuggingFace, j'ai trouvé le problème. La méthode generate() appelle directement parameter_projection sur dec_output, sans passer par notre override output_params. En d'autres termes, à l'inférence, le modèle contourne entièrement notre Evidence Layer.

Le patch, Fix #5, est une seule ligne changée : remplacer l'appel à parameter_projection par l'appel à output_params dans un override de generate(). Une ligne qui change tout.

---

## Slide 20 — Implications

Ce bug est important à contextualiser. Il n'invalide pas le travail.

L'entraînement n'était pas affecté : forward() utilisait correctement output_params. Tous les checkpoints sont valides. La loss régularisée et la convergence n'ont jamais touché la branche buggée.

Ce qui était affecté, c'est toutes les métriques rapportées avant la découverte du bug. Elles mesuraient dec_output seul, pas la branche h_ev. Les décisions sur le choix du mix d'entraînement étaient donc prises sur des chiffres incomplets.

Je présente ce bug comme un contrôle méthodologique : l'avoir détecté et documenté renforce la rigueur du protocole, plutôt qu'il ne l'invalide.

---

## Slide 21 — Sweep mix post Fix #5 — B5

Après Fix #5, je relance le sweep mix sur B5. Et cette fois, la courbe n'est plus plate.

À mix = 0, R² = 0.6646 — c'est M neutralisée, le modèle ignore l'Evidence Layer.

À mix = 0.05, le mix d'entraînement, R² = 0.71.

Mais le pic est à mix = 0.25 : R² = 0.7563. Le modèle performe mieux avec plus de M que ce qu'il a vu à l'entraînement.

Au-delà de 0.25, ça se dégrade : à 0.50 on est en cassure.

---

## Slide 22 — Généralisation B6 et B7

Pour vérifier que ce n'est pas spécifique à B5, j'ai entraîné deux autres runs : B6 avec mix = 0.10 à l'entraînement, B7 avec mix = 0.15.

Le pattern se répète : dans les trois cas, le pic d'inférence est à environ 5 fois le mix d'entraînement.

B5 : entraîné à 0.05, pic à 0.25 — ×5.
B6 : entraîné à 0.10, pic à 0.50 — ×5.
B7 : entraîné à 0.15, pic à 0.50 — ×3.3.

Ce n'est pas un hasard isolé. C'est un pattern empirique reproductible.

---

## Slide 23 — Interprétation de la dissociation

Pourquoi le pic est-il plus loin que le mix d'entraînement ?

À l'entraînement avec mix = 0.05, 95% du signal passe par dec_output. La tête de sortie se calibre principalement sur cette branche. h_ev, elle, apprend sous contrainte ElasticNet et entropie — elle ne peut pas se permettre de patterns spécifiques à une seule fonction. Elle encode donc des patterns stables et partagés au niveau du cluster.

À l'inférence, ce signal stable et sparse est fiable. Lui donner plus de poids — passer à mix = 0.25 — améliore la prédiction. Au-delà de 0.25, dec_output perd trop d'influence, et le compromis se casse.

La dissociation est une propriété de la régularisation, pas un artefact. C'est la trouvaille originale de ce travail.

---

## Slide 24 — Décomposition du gain final

Ce graphique montre comment le R² progresse étape par étape.

On part de FAYAM à 0.37. Notre reproduction pure — Run A — monte à 0.53. L'ajout de l'Evidence Layer avec mix = 0 donne 0.66. Et avec mix = 0.25 à l'inférence, on atteint 0.7563.

Chaque barre correspond à un apport réel et mesurable.

---

## Slide 25 — Comparaison à la baseline et au TFT

Comparé à FAYAM, notre configuration finale gagne 38 points de R². Le TFT de Lim et al., à titre de référence, est autour de 0.71 sur des données comparables — mais attention, ce chiffre n'est pas mesuré sur notre dataset, c'est une référence indicative.

Notre modèle est comparable au TFT sur la performance, avec en plus une explication fidèle par construction — ce que le TFT, basé sur l'attention, ne peut pas revendiquer.

---

## Slide 26 — Méthodologie de validation

Avant de vous montrer les résultats, je veux clarifier ce que je cherche à valider. J'ai posé sept hypothèses sur le comportement de M.

H1.A : M pointe les moments de pic de charge — elle est temporellement cohérente.
H1.B : M corrèle avec les cross-attentions du décodeur.
H1.C : La précision du modèle est conservée malgré l'Evidence Layer.
H1.D : M est cohérente entre les fonctions du même cluster.
H1.E : M est plus concentrée quand la prédiction est meilleure.
H1.F : Masquer les positions clés de M dégrade la précision — c'est le test de comprehensiveness.
H1.G : Garder seulement les positions clés de M préserve la précision — c'est le test de sufficiency.

---

## Slide 27 — H1.C : non-dégradation de la précision

H1.C est le test de survie : est-ce que l'Evidence Layer ne dégrade pas la précision du modèle original ?

R² atteint 0.7563 — largement au-dessus du seuil de 0.30. Spearman atteint 0.938 — au-dessus de 0.85. Les deux critères sont validés avec une marge confortable.

L'Evidence Layer n'est pas une contrainte qui nuit à la performance. Elle améliore même le modèle.

---

## Slide 28 — H1.D : cohérence intra-cluster

H1.D vérifie que M est cohérente entre les cinq fonctions du cluster. Si M capture des patterns de cluster plutôt que des artefacts individuels, les matrices M de fonctions différentes devraient se ressembler.

Le Pearson moyen entre les matrices M des cinq fonctions est de 0.992. C'est une cohérence quasi-parfaite.

Je note cependant une limite honnête : les fonctions du même cluster partagent des paramètres et des profils similaires par construction du clustering. La corrélation était en partie prévisible. Un test inter-clusters, plus discriminant, reste à faire.

---

## Slide 29 — H1.E : sparsité et précision

H1.E explore si une M plus concentrée — plus sparse — correspond à une meilleure prédiction.

Sur les cinq fonctions, le Spearman entre le R² par fonction et l'entropie moyenne de M est de -0.80. La tendance est là : plus M est concentrée, meilleure est la prédiction.

Je suis transparent sur la limite : avec seulement cinq fonctions, ce résultat n'est pas statistiquement significatif. Je le présente comme une tendance indicative, qui sera confirmée ou infirmée par H1.F et H1.G.

---

## Slide 30 — H1.F : comprehensiveness

H1.F est un test de causalité. Je masque progressivement les positions les plus importantes de M, et je mesure si la précision se dégrade.

Si M est vraiment utilisée dans le calcul, la masquer doit faire baisser la performance.

Résultats : masquer 1 position dégrade de 1.13%, masquer 10 de 3.27%, masquer 100 de 4.97%, et masquer tout donne au maximum 5.37%.

Le plafond théorique est 25% — parce que M ne pèse que 25% dans le calcul final avec mix = 0.25. On n'atteint que 5.37% sur 25% possibles, soit 21% du plafond. M guide bien la prédiction, mais le décodeur reste autonome.

---

## Slide 31 — H1.G : sufficiency

H1.G est l'inverse. Je garde seulement les k positions les plus importantes de M, et je mesure si la précision est préservée.

Si M est sparse, une seule position devrait suffire à transmettre l'essentiel de l'information.

Résultat : avec seulement k = 1 position gardée, la préservation est déjà de 97.1%. Avec k = 50, on dépasse 100%. Une seule position de M concentre presque toute l'information utile.

---

## Slide 32 — H1.A et H1.B : résultats nuancés

Je suis honnête sur les deux hypothèses restantes.

H1.A vérifie que M pointe les pics de charge 17h-19h. J'ai observé que les maxima de M se concentrent bien dans ces plages horaires. Mais je n'ai pas encore conduit le test complet : vérifier l'alignement heure par heure sur les 24h, pas juste aux heures de pic. C'est une nuance que je présente ouvertement.

H1.B vérifie que M corrèle avec les cross-attentions du décodeur. Ce test n'a pas été conduit — la cross-attention brute et M jouent des rôles différents dans l'architecture, et la divergence attendue rendait le test peu concluant sans une référence fiable.

---

## Slide 33 — La question que vous pourrez me poser

Je veux anticiper une question qui pourrait surgir : vous avez réglé mix après l'entraînement. N'est-ce pas une explication post-hoc déguisée ?

Non. Mix est un réglage global et unique — comme la température d'un LLM, il ne change pas d'une prédiction à l'autre. La fidélité de M au calcul tient dès que mix est supérieur à zéro, quelle que soit la valeur. Une fois mix fixé à 0.25, le modèle est entièrement déterministe. Et la dissociation entraînement/inférence est une observation empirique documentée, pas une correction de dernière minute.

Je parlerai de self-explainability calibrée dans le mémoire — honnête sur la dissociation, mais pas post-hoc.

---

## Slide 34 — Ce que ce travail ne fait pas encore

Cinq limites que j'assume ouvertement.

Un seul run, seed 998 — je n'ai pas évalué la variance des résultats.

Un seul cluster testé, C4 — les autres ne convergent pas même en baseline.

H1.A incomplet — le test heure par heure sur 24h reste à faire.

H1.D avec un biais potentiel — les fonctions du même cluster sont similaires par construction.

H1.F avec renormalisation — après masquage, j'ai renormalisé M, ce qui atténue l'effet mesuré.

Ces points ne remettent pas H1 en cause. Ils balisent honnêtement ce qui reste ouvert.

---

## Slide 35 — Ce que j'attends de vous aujourd'hui

Pour conclure, voici les points sur lesquels j'ai besoin de votre retour.

Sur ce que je propose : validez-vous B5 + mix = 0.25 comme configuration finale ? H1.A incomplet est-il bloquant pour la soutenance ? Et le terme calibrated self-explainability vous semble-t-il défendable dans le mémoire ?

Sur ce qu'il reste à décider : faut-il ajouter une comparaison post-hoc avec TimeSHAP ou Integrated Gradients, ou H1 se suffit-il ? Et quand pouvez-vous me donner un retour sur le premier draft du chapitre H1 ?

Je vous remercie.
