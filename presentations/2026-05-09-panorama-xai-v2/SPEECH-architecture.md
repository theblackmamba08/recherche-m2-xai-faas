# SPEECH — Section 10 : Architecture SoftCAM-Transformer (notre contribution)

> **Pour qui** : Cabrel, lors de la présentation du 09/05/2026 à Dr LACMOU.
> **Couvre** : les **10 slides** de la section *« SoftCAM-Transformer (notre contribution H1) »* de [slides.pdf](slides.pdf), section 10.
> **Durée cible totale** : 12–15 min (~1 min 30 par slide en moyenne).
> **Objectif** : expliquer de façon très explicite **l'architecture proposée**, comment elle fonctionne, pourquoi elle apporte de l'explicabilité, et comment on va la valider.

---

## Avant la section — Phrase de transition depuis la synthèse SHAPformer vs SoftCAM

> *« Maintenant que nous avons vu d'un côté SHAPformer — un Transformer rendu compatible avec SHAP exact mais qui reste à deux étapes — et de l'autre côté SoftCAM — un CNN où l'explication est constitutive du modèle mais qui n'a jamais été appliqué à un Transformer de prévision — la question naturelle qui se pose est : peut-on combiner les deux ? Peut-on appliquer le principe SoftCAM, qui est intrinsèque, à un Transformer comme celui de FAYAM, qui fait de la prévision multivariée ? C'est exactement ce que je propose avec SoftCAM-Transformer. Je vais maintenant détailler son architecture. »*

> 📌 **Note Cabrel** : ralentir ici. Cette transition est le point pivot de la présentation. Marquer une pause de 2-3 secondes avant de passer au slide suivant pour laisser le jury "attendre" votre proposition.

---

## Slide 10.1 — Motivation : la synthèse logique du parcours

**Durée cible : 60 s**

### Texte à dire

> *« Reprenons d'abord ce que les méthodes que nous venons de voir ne couvrent pas. Aucune méthode XAI existante ne combine simultanément ces cinq propriétés : architecture Transformer, tâche de prévision, fidélité intrinsèque (c'est-à-dire en un seul forward pass), séries multivariées, et coût d'inférence nul pour l'explication.*
>
> *Chaque méthode existante en couvre certaines mais jamais toutes. TsSHAP fait de la prévision mais avec un surrogate XGBoost — donc fidélité approximative. SHAPformer fait du Transformer prévision exact mais en deux phases. SoftCAM est intrinsèque mais sur CNN classification.*
>
> *Notre proposition, c'est SoftCAM-Transformer : adapter le principe SoftCAM, c'est-à-dire la couche d'évidence avec régularisation ElasticNet, au TimeSeriesTransformerForPrediction de HuggingFace que FAYAM utilise. Et on cible spécifiquement le cluster 4 de FAYAM, parce que c'est le seul cluster où le baseline Transformer apprend correctement — avec un R² de 0.37 et un Spearman de 0.92 — et qu'il a un signal périodique journalier très structuré, ce qui devrait produire une carte d'évidence visuellement convaincante. »*

> 📌 **Note Cabrel** : insister sur "intrinsèque" en pointant vers la slide. Le mot "synthèse logique" est important — montrez que c'est la conclusion *naturelle* de votre revue de littérature, pas une idée tirée du chapeau.

**Transition vers slide suivante** : *« Voyons maintenant à quoi ressemble cette architecture concrètement. »*

---

## Slide 10.2 — Vue d'ensemble : le flux complet (le grand schéma)

**Durée cible : 2 min** (slide la plus dense — prendre le temps)

### Texte à dire

> *« Voici le schéma complet de l'architecture proposée. Je vais le parcourir avec vous de haut en bas.*
>
> *Tout commence par le contexte passé : 240 valeurs, c'est-à-dire 240 minutes d'observation de la charge FaaS, qui constituent l'entrée du modèle. Cette entrée passe par l'encodeur du Transformer — qui est exactement celui de FAYAM, je n'y touche pas — et qui produit ce qu'on appelle des embeddings encodeur : un tenseur de 240 vecteurs de dimension 32 qui résument le contexte.*
>
> *Notez bien la flèche jaune en pointillés sur la droite : ces embeddings encodeur ne servent pas qu'au décodeur. Ils sont aussi réutilisés plus bas, et ce sera essentiel.*
>
> *Les embeddings encodeur passent ensuite dans le décodeur — toujours hérité de FAYAM — qui produit les embeddings décodeur : 120 vecteurs de dimension 32, un par pas futur à prédire.*
>
> *C'est ici que commence ma contribution : à la place de la couche `parameter_projection` qui chez FAYAM transforme directement ces embeddings en paramètres de distribution, j'introduis le bloc encadré en rouge : l'**Evidence Layer**. C'est le seul composant véritablement nouveau de mon architecture.*
>
> *Cette couche prend les embeddings décodeur et produit ce que j'appelle la carte d'évidence : une matrice de taille 120 par 240. Concrètement, pour chaque pas futur t que je veux prédire, j'apprends un vecteur de 240 poids — un poids par position du contexte passé.*
>
> *Cette carte d'évidence a deux destinations. D'abord, elle est combinée linéairement avec les embeddings encodeur — c'est ici qu'intervient la flèche jaune dont je parlais — pour produire les vecteurs latents prédits. Ensuite, ces vecteurs passent par une projection linéaire finale qui produit les paramètres Student-t et donc la prédiction.*
>
> *Mais — et c'est la beauté du design — la carte d'évidence est aussi **directement** la sortie d'explication, exposée à l'utilisateur. C'est la flèche orange à droite. L'explication est produite **gratuitement**, dans le même forward pass que la prédiction. »*

> 📌 **Note Cabrel** : ce slide est dense. Utiliser un pointeur (souris ou laser) pour suivre le flux. Si le jury suit visuellement, vous suivrez plus naturellement. Marquer une pause après avoir terminé pour laisser absorber.

**Transition vers slide suivante** : *« Pour rendre cette vue d'ensemble plus concrète, regardons les dimensions exactes des tenseurs qui circulent dans chaque bloc. »*

---

## Slide 10.3 — Entrées et sorties par bloc

**Durée cible : 1 min 30**

### Texte à dire

> *« Ce tableau récapitule les entrées et sorties de chaque bloc, avec les dimensions exactes pour le cas FAYAM : context_length à 240, prediction_length à 120, et d_model à 32.*
>
> *La colonne de droite indique le statut de chaque bloc. Vous voyez que la grande majorité — encodeur, décodeur, projection finale, prédiction — sont marqués "héritage FAYAM". Je ne réécris pas le Transformer : je le réutilise tel quel.*
>
> *Seules deux lignes sont en rouge gras, marquées "NOUVEAU" : l'Evidence Layer et la Carte d'évidence qui en sort. C'est précisément l'objet de ma contribution.*
>
> *Le bloc Combinaison linéaire est marqué "calcul" — ce n'est pas un module entraîné, c'est juste une multiplication matricielle entre la carte d'évidence et les embeddings encodeur. Pas de paramètres apprenables ajoutés ici.*
>
> *Ce que je veux qu'on retienne : 95% du modèle FAYAM est inchangé. La modification est minimaliste — c'est exactement l'esprit de SoftCAM original sur CNN. »*

> 📌 **Note Cabrel** : si quelqu'un demande "combien de paramètres ajoutés ?", la réponse est : **environ 7 700 paramètres** (Linear 32→240 = 7 680 + Linear 32→3 = 96), soit **moins de 5%** du Transformer FAYAM (~150 000 paramètres). Mention rapide possible si question.

**Transition vers slide suivante** : *« Concentrons-nous maintenant sur le seul vrai bloc nouveau : l'Evidence Layer. À l'intérieur, qu'est-ce qu'il y a ? »*

---

## Slide 10.4 — Zoom : composition interne de l'Evidence Layer

**Durée cible : 1 min 30**

### Texte à dire

> *« L'Evidence Layer est en réalité très simple : trois ingrédients seulement.*
>
> *Premier ingrédient : une **projection linéaire**, c'est-à-dire une couche dense `Linear(32 → 240)`. Elle prend chaque embedding décodeur de dimension 32 et produit 240 scores bruts, un par position du contexte. C'est le cœur appris du module.*
>
> *Deuxième ingrédient : un **softmax** appliqué sur les 240 positions. Le softmax garantit deux choses : que les poids sont positifs, et qu'ils somment à 1. Donc chaque ligne de la carte d'évidence devient une **distribution de probabilité** sur les 240 pas du passé. Ça se lit très naturellement : "70% sur le lag 24h, 20% sur les 5 dernières minutes, 10% ailleurs".*
>
> *Troisième ingrédient : la **régularisation ElasticNet**. Mais attention — elle n'est pas dans le module lui-même, elle est dans la **fonction de perte**. C'est un terme qu'on ajoute au loss pour forcer la carte à être parcimonieuse et lisse. On y reviendra dans deux slides.*
>
> *À droite, vous voyez le code PyTorch correspondant. Vous remarquerez que c'est très court — une dizaine de lignes seulement. Le forward fait trois choses : projection des embeddings décodeur, softmax pour obtenir la carte, multiplication matricielle de la carte par les embeddings encodeur, puis projection finale en paramètres de distribution.*
>
> *Et la fonction renvoie deux choses : les paramètres pour la prédiction, et la carte elle-même pour l'explication. »*

> 📌 **Note Cabrel** : si on demande "pourquoi softmax et pas softplus ?", la réponse est : softmax donne une **distribution de probabilité** qui se lit directement comme "X% sur tel pas". Softplus donne des poids positifs mais sans contrainte de somme — moins lisible. C'est un choix d'**interprétabilité**.

**Transition vers slide suivante** : *« Maintenant, l'équation qui résume tout. »*

---

## Slide 10.5 — L'équation centrale

**Durée cible : 1 min 15**

### Texte à dire

> *« Voici la formule mathématique au cœur de SoftCAM-Transformer.*
>
> *La première ligne dit : pour chaque pas futur t, le **vecteur latent prédit** est la somme, sur les 240 positions du passé, du produit du poids de la carte par l'embedding encodeur correspondant. C'est une **combinaison linéaire pondérée**.*
>
> *La deuxième ligne dit : la prédiction finale est obtenue en projetant ce vecteur latent par la couche linéaire finale qui produit les paramètres Student-t.*
>
> *Lecture sémantique — c'est le point clé de la diapositive : `carte[t, i]` est, **par construction**, le **poids** que le modèle attribue au pas passé i pour fabriquer la prédiction du pas t. Ce n'est pas une approximation, ce n'est pas une heuristique — c'est la formule **mathématique exacte** qui produit la prédiction.*
>
> *Et c'est précisément ce qui en fait une explication intrinsèque, comme indiqué à droite : la carte n'est pas calculée à part après la prédiction, elle **est** la formule de la prédiction. La fidélité de l'explication est donc **parfaite, par construction**. Il n'y a aucun écart possible entre ce que la carte raconte et ce que le modèle fait. »*

> 📌 **Note Cabrel** : si on demande "mais alors la projection finale linéaire ne casse-t-elle pas l'interprétabilité ?", la réponse est : **non**, parce qu'elle est *linéaire et globale* — c'est-à-dire que la même projection s'applique pour tous les pas futurs t, et qu'elle préserve la décomposition. Si la carte est interprétable, la prédiction l'est aussi par composition. C'est exactement le même principe que dans SoftCAM CNN, où le softmax après l'evidence layer ne casse pas l'interprétabilité de la carte.

**Transition vers slide suivante** : *« Cette équation seule ne suffit pas — il faut aussi la régularisation ElasticNet pour que les cartes restent lisibles. »*

---

## Slide 10.6 — Loss étendue : ElasticNet sur la carte

**Durée cible : 1 min 30**

### Texte à dire

> *« La fonction de perte totale a trois termes.*
>
> *Le premier terme est la log-vraisemblance négative de la distribution Student-t — c'est exactement le loss de FAYAM. C'est ce qui pousse le modèle à bien prédire.*
>
> *Le deuxième terme, c'est la norme L1 de la carte d'évidence, multipliée par un coefficient lambda1. C'est la pénalité **Lasso** — elle force la majorité des poids de la carte à être zéro. Conséquence : la carte ne pointe qu'un **petit nombre** de positions du contexte. Au lieu d'avoir 240 poids éparpillés, on a peut-être 5 ou 10 pics — donc une explication directement lisible.*
>
> *Le troisième terme, c'est la norme L2 de la carte au carré, multipliée par lambda2. C'est la pénalité **Ridge** — elle pénalise les pics isolés et favorise un lissage entre poids voisins. Donc les poids voisins varient progressivement, ce qui donne une cohérence temporelle visible sur la carte.*
>
> *Les deux ensemble, c'est ce qu'on appelle **ElasticNet** — c'est le compromis adaptatif. La parcimonie pour la lisibilité, le lissage pour la cohérence.*
>
> *Pour les hyperparamètres de départ, je propose lambda1 autour de 10 puissance moins 3, lambda2 autour de 10 puissance moins 4 — ce sont les valeurs transposées de l'article SoftCAM CNN. Évidemment, il faudra les régler par grid search sur le cluster 4. »*

> 📌 **Note Cabrel** : c'est la slide la plus "mathématique". Si vous sentez le jury décrocher, dites simplement : *"La régularisation ElasticNet, c'est ce qui transforme une carte 'mathématiquement valide mais illisible' en une carte 'visuellement interprétable'. Sans elle, on aurait des poids éparpillés sans pic clair."*

**Transition vers slide suivante** : *« Avant de finir l'architecture, je dois être honnête sur un choix de design qui reste ouvert. »*

---

## Slide 10.7 — Trois variantes de design

**Durée cible : 1 min 30**

### Texte à dire

> *« J'ai présenté l'architecture en montrant la combinaison linéaire entre la carte et les **embeddings encodeur**. Mais c'est en réalité un choix de design parmi trois possibles.*
>
> *Variante A : combiner la carte avec le **contexte brut**, c'est-à-dire avec les valeurs scalaires originales. Lisibilité maximale — on peut dire "70% de la prédiction vient de la valeur d'il y a 24h". Mais expressivité limitée : le modèle ne peut prédire que des combinaisons linéaires des valeurs déjà observées. Il ne pourra jamais prédire un pic plus haut que le maximum vu.*
>
> *Variante B — celle que je recommande, et qui est dans le schéma — : combiner la carte avec les **embeddings encodeur**. Lisibilité élevée — la position est préservée, on lit la carte comme un graphique 1D sur le contexte. Mais expressivité élevée aussi, parce que les embeddings ont déjà capturé les patterns non-linéaires comme la saisonnalité.*
>
> *Variante C : hybride — combinaison linéaire des valeurs brutes plus un terme de biais résiduel appris par un MLP. Compromis intermédiaire mais qui sacrifie un peu la lisibilité.*
>
> *Pourquoi je propose la variante B ? Trois raisons : elle est la plus fidèle à l'esprit de SoftCAM original — qui combine la carte avec des feature maps, pas avec les pixels bruts. Elle est plus expressive, donc moins susceptible de dégrader la performance prédictive. Et le cluster 4 a une certaine hétérogénéité — la fonction 953 a un R² de 0.60, la 949 a un R² de 0.15 — la variante A serait probablement trop contrainte pour cette diversité. »*

> 📌 **Note Cabrel** : c'est le bon endroit pour montrer une posture critique et mature. Le jury appréciera que vous reconnaissiez les choix non triviaux et que vous les justifiiez explicitement.

**Transition vers slide suivante** : *« Pour rendre tout ça concret, voyons à quoi ressemble une carte d'évidence sur un cas réel. »*

---

## Slide 10.8 — À quoi ressemble une explication — exemple sur C4

**Durée cible : 1 min 30**

### Texte à dire

> *« Cette diapositive vous montre, de manière schématique, à quoi ressemblera une carte d'évidence pour la fonction 953 du cluster 4, quand le modèle prédit la charge à lundi 17h00.*
>
> *Sur l'axe horizontal : les 240 minutes du contexte, du plus récent à droite jusqu'à 4 heures en arrière à gauche. Sur l'axe vertical : le poids de chaque pas dans la formule de prédiction.*
>
> *Que voit-on ? Trois pics :*
>
> *Le pic le plus haut, autour de 0.7, est à environ 24 heures dans le passé — c'est-à-dire dimanche 17h00. Le modèle dit donc : "pour prédire lundi 17h, je regarde principalement dimanche 17h" — ce qui est totalement cohérent avec le profil journalier que l'EDA a montré sur le cluster 4.*
>
> *Le deuxième pic, autour de 0.4, est sur les 5 dernières minutes — la valeur la plus récente, qui sert d'ajustement local.*
>
> *Le troisième pic, plus petit, est il y a environ 4 heures — peut-être lundi 13h, un signal partiel.*
>
> *Tout le reste est à zéro grâce à la parcimonie ElasticNet.*
>
> *Cette carte n'est pas une simulation — c'est ce que le modèle **devrait** produire si nos hypothèses sont correctes. Et c'est exactement la **validation visuelle** de l'hypothèse H1.A que j'ai actée dans DECISIONS.md : la carte d'évidence pointe les heures du profil journalier. Si la carte ressemble vraiment à ça après entraînement, l'hypothèse est validée. »*

> 📌 **Note Cabrel** : insister sur "**ce que le modèle devrait produire**" — c'est important de ne pas faire croire que c'est un résultat acquis. C'est un objectif qu'on va vérifier empiriquement.

**Transition vers slide suivante** : *« Comment va-t-on valider tout cela ? Voici les hypothèses que j'ai déjà actées. »*

---

## Slide 10.9 — Hypothèses opératoires H1.A à H1.E

**Durée cible : 1 min 30**

### Texte à dire

> *« Pour ne pas tester "à l'intuition", j'ai formulé cinq hypothèses opérationnelles qui sont déjà actées dans le fichier DECISIONS.md du projet.*
>
> *H1.A — la carte d'évidence pointe les heures du profil journalier. C'est ce que je viens de montrer dans l'exemple : pic à 17h, creux à 2h-6h. Validation : visuelle et quantitative sur les 5 fonctions du cluster.*
>
> *H1.B — les têtes d'attention du décodeur se polarisent autour des lags 1440 et 2880 minutes, c'est-à-dire 24h et 48h. C'est mon contrôle croisé : si la carte d'évidence et les attention weights pointent les mêmes choses, c'est un signe fort de cohérence.*
>
> *H1.C — c'est la garde-fou crucial : SoftCAM ne dégrade pas la précision baseline. Concrètement, le R² doit rester supérieur ou égal à 0.30 et le Spearman supérieur ou égal à 0.85 sur le cluster 4. Si je casse ces seuils, ça veut dire que la contrainte d'evidence layer est trop forte pour la tâche, et je dois basculer vers H2.*
>
> *H1.D — cohérence des cartes entre les 5 fonctions du cluster. Le Pearson intra-cluster est supérieur à 0.95 d'après l'EDA, donc les cartes apprises sur les différentes fonctions devraient être très similaires. Si elles divergent fortement, c'est suspect.*
>
> *H1.E — test sur les deux extrêmes : la fonction 953 qui est mon best case avec R² de 0.60, et la fonction 949 qui est le stress test à R² de 0.15. Cela permet d'évaluer la robustesse à la qualité de prédiction.*
>
> *Et en bas, le critère de bascule vers H2 : si après environ 3 semaines de prototypage, le modèle ne converge pas ou si H1.C est violé, je bascule sur SHAPformer ou TsSHAP comme repli — c'est rigoureusement justifié dans le ROADMAP. »*

> 📌 **Note Cabrel** : c'est le moment de montrer votre **rigueur scientifique**. Vous avez non seulement une idée — vous avez un protocole de validation et un plan de repli. Cela rassure énormément un jury et un encadreur.

**Transition vers slide suivante** : *« Voilà comment SoftCAM-Transformer combine ce qui était dispersé dans la littérature en une seule contribution intrinsèque pour la prévision FaaS. »*

---

## Conclusion de la section — Quoi dire pour fermer la boucle

**Durée cible : 30 s**

### Texte à dire

> *« Pour résumer cette section : SoftCAM-Transformer, c'est une modification architecturale **légère** — une dizaine de lignes de code et environ 7 000 paramètres ajoutés sur les 150 000 du Transformer FAYAM. C'est une **contribution scientifique distincte** parce que personne, à ma connaissance, n'a publié l'application de SoftCAM à un Transformer de prévision multivariée — la perspective ViT/Transformer est explicitement annoncée dans l'article SoftCAM original comme une piste future. Et c'est une approche **falsifiable** parce que j'ai des hypothèses opératoires précises et un critère de bascule en cas d'échec.*
>
> *La prochaine étape concrète, dès cette semaine, c'est le jour 1 du plan d'étude architecture : cartographier précisément la classe `parameter_projection` de HuggingFace pour localiser la cible exacte de mon evidence layer. »*

---

## Anticipation des questions probables

### Q1 : *« Pourquoi pas appliquer SHAPformer directement ? »*

> *« Excellente question. SHAPformer est mon repli H2 — il est directement applicable. Mais il a deux inconvénients pour notre cas. D'abord, il calcule SHAP en 2^N forward passes — c'est rapide en absolu, mais pas zéro. Pour FAYAM avec 18 fonctions, on aura potentiellement beaucoup de groupes de features, et le coût peut grimper. Ensuite, et c'est le point conceptuel, SHAPformer reste post-hoc dans son principe : prédiction et explication sont deux étapes, même si l'entraînement les rend cohérentes. SoftCAM-Transformer pousse plus loin : la prédiction ET l'explication sont produites en un seul calcul, indissociables. C'est un saut paradigmatique — qui peut échouer, d'où H2 en repli. »*

### Q2 : *« Vous risquez de dégrader la performance, non ? »*

> *« Oui, c'est le risque que je nomme explicitement dans H1.C. C'est pour ça que j'ai mis un seuil — R² ≥ 0.30 et Spearman ≥ 0.85. Le seuil correspond au baseline FAYAM dédié C4. Si je le casse de plus de 10-15%, je bascule. Mais empiriquement, l'article SoftCAM CNN montre une perte d'AUC inférieure à 1% sur trois datasets médicaux. Et SHAPformer, qui modifie aussi le modèle, perd seulement 1% de RMSE. Les modifications architecturales bien conçues ne dégradent pas autant qu'on le craint — surtout sur un signal aussi structuré que C4 (FFT 24h à 75-80% de variance). »*

### Q3 : *« Comment la carte est-elle évaluée objectivement ? »*

> *« Il n'y a pas d'annotation ground-truth — on n'a pas un humain qui dit "voici les heures qui devraient compter". J'utilise donc deux proxies. D'abord, les métriques de **faithfulness** classiques : comprehensiveness (si je masque les pas que la carte met en avant, la prédiction se dégrade-t-elle ?) et sufficiency (si je ne garde que ces pas, la prédiction tient-elle ?). Ensuite, le **contrôle croisé avec les attention weights** du décodeur — c'est l'hypothèse H1.B. Si la carte d'évidence et l'attention pointent les mêmes lags, c'est un signe convergent. Pas une preuve, mais un faisceau d'indices. »*

### Q4 : *« Pourquoi spécifiquement le cluster 4 et pas tous les clusters ? »*

> *« Pour deux raisons. Première raison empirique : c'est le seul cluster où le baseline Transformer apprend correctement. Sur C0, R² ≈ 0 — le modèle prédit la moyenne. Sur C6 et C8, R² négatif — modèles triviaux. Sans modèle qui apprend, on ne peut pas avoir de carte d'évidence informative. Deuxième raison didactique : C4 a un signal périodique 24h très structuré, donc la carte d'évidence devrait converger sur des heures précises et être visuellement convaincante. C'est le terrain idéal pour démontrer la faisabilité du concept. Si SoftCAM-Transformer marche sur C4, on a notre contribution. Étendre à d'autres clusters serait du travail futur, pas une condition de réussite du mémoire. »*

### Q5 : *« Et si la carte d'évidence apprise n'est pas interprétable ? »*

> *« C'est un risque réel. Trois cas possibles. Cas 1 : la carte est trop dense, les poids partout — solution : augmenter lambda1 (parcimonie). Cas 2 : la carte est instable entre les fonctions du cluster — c'est H1.D qui détecte ça, et indique probablement un sur-apprentissage spécifique aux fonctions. Cas 3 : la carte ne pointe pas les heures attendues — c'est plus profond, ça peut indiquer que le modèle n'a pas appris les bons patterns ou que mes hypothèses étaient fausses. Dans ce cas, c'est une contribution **honnête** : on peut publier "voilà ce que le modèle apprend vraiment" et ça reste un résultat scientifiquement valide. Pas un échec — un résultat différent. »*

---

## Notes finales pour Cabrel

1. **Rythme global** : la section dure environ **12-15 minutes**. Si vous êtes en retard sur l'horloge, l'élément le plus compressible est la slide 10.7 (variantes A/B/C) — vous pouvez réduire à 30 secondes en disant juste "j'ai trois variantes possibles, je prends la B parce qu'elle est la plus fidèle à SoftCAM original, j'expliquerai si question".

2. **Élément non négociable** : la slide 10.5 (équation centrale). C'est le **moment intellectuel** de la présentation. Si vous bâclez celui-là, le jury n'aura pas compris pourquoi votre méthode est intrinsèque.

3. **Pause stratégique** : après la slide 10.2 (vue d'ensemble), faire une pause d'**au moins 5 secondes** avant de continuer. C'est le moment où le jury absorbe le schéma — donnez-lui le temps.

4. **Si vous bloquez** : revenez toujours à la phrase clé : *« la carte d'évidence est, par construction, la formule de prédiction du modèle ».* C'est votre boussole conceptuelle.

5. **Confiance** : vous savez votre architecture. Vous l'avez écrite, schématisée, formulée mathématiquement, codée en pseudo-PyTorch, et formulé des hypothèses falsifiables. Vous êtes objectivement plus loin que beaucoup de mémoires M2 à ce stade. **Allez-y avec calme.**
