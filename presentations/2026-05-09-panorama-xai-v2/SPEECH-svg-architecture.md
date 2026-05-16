# SPEECH — Le SVG d'architecture SoftCAM-Transformer

> **Fichier référencé** : [`figures/softcam-transformer-architecture.svg`](figures/softcam-transformer-architecture.svg)
> **Pour qui** : Cabrel, à utiliser à chaque fois qu'il faut **expliquer le schéma seul** — devant un écran, au tableau (en le redessinant), ou en échange informel.
> **Différence avec [SPEECH-architecture.md](SPEECH-architecture.md)** : le speech architecture couvre **10 slides** ; ce fichier-ci se concentre **uniquement sur le grand schéma** et donne plusieurs niveaux de profondeur (60 s, 3 min, et version technique complète).

---

## Niveau 1 — Version courte (60 s)

À utiliser quand on dispose de peu de temps, ou en réponse à *« montrez-moi rapidement votre architecture »*.

> *« Le schéma se lit de haut en bas. En haut, j'ai les données d'entrée — 240 minutes de charge passée. Elles passent par l'encodeur puis le décodeur du Transformer FAYAM, que je laisse tels quels.*
>
> *La nouveauté est ici, encadrée en rouge : l'**Evidence Layer**. Elle remplace la dernière étape opaque du Transformer original — la `parameter_projection` — par un mécanisme transparent. À sa sortie, j'obtiens une **carte d'évidence**, c'est-à-dire une matrice qui dit, pour chaque pas futur à prédire, quelle pondération je donne à chaque pas du passé.*
>
> *Cette carte a deux destinations : à gauche, une combinaison linéaire qui produit la prédiction finale en bas. À droite — la flèche orange — c'est directement l'explication, exposée à l'utilisateur **dans le même forward pass**.*
>
> *Donc : un seul calcul produit prédiction et explication. C'est ça, l'intrinsèque pur. »*

> 📌 **Notes Cabrel** :
> - Pointer physiquement le bloc rouge en disant "encadrée en rouge"
> - Pointer la flèche orange en disant "même forward pass"
> - Si on vous pose une question avant la fin, c'est bon signe — répondez puis revenez au schéma

---

## Niveau 2 — Version standard (3 min)

À utiliser comme version **par défaut** quand vous présentez le schéma — c'est le niveau qui correspond à la slide 10.2 mais légèrement enrichi.

### Phase 1 : poser le contexte (15 s)

> *« Avant de plonger dans le schéma, deux mots de contexte. Ce que vous voyez, c'est l'architecture proposée pour SoftCAM-Transformer — ma contribution H1. Le code couleur est important : tout ce qui est en gris ou bleu pâle est **hérité tel quel** de FAYAM. Tout ce qui est en rouge est **ma contribution**. Et la branche orange à droite, c'est la sortie d'explication. »*

### Phase 2 : flux principal de haut en bas (90 s)

> *« On commence en haut.*
>
> *Le bloc bleu **Contexte passé** : 240 valeurs, c'est-à-dire 240 minutes d'observation de charge — pour FaaS, c'est le nombre d'invocations par minute sur la fenêtre récente.*
>
> *Ensuite, l'**encodeur** du Transformer. Je n'y touche pas. Sa sortie, c'est ce que j'appelle les **embeddings encodeur** — le rectangle jaune. Concrètement : 240 vecteurs de dimension 32, un par minute du contexte. Chaque vecteur est une représentation latente qui résume "ce que cette minute signifie dans son contexte".*
>
> *Cette information descend ensuite vers le **décodeur** — encore inchangé. Le décodeur produit, lui, les **embeddings décodeur** : 120 vecteurs de dimension 32, un par minute future à prédire.*
>
> *À ce stade-là, dans le Transformer FAYAM original, ces embeddings entreraient dans une couche `parameter_projection` qui les transforme directement en paramètres de la distribution Student-t. C'est une boîte noire — on ne sait pas ce qu'elle fait.*
>
> *Dans mon architecture, je remplace cette boîte noire par le bloc rouge en gras : l'**Evidence Layer**. Cette couche apprend, pour chaque pas futur, un **vecteur de 240 poids** sur les positions du contexte. Concrètement, si je veux prédire la minute t+50, l'evidence layer me dit : "voici comment je vais répartir mon attention sur les 240 minutes passées".*
>
> *Sa sortie, c'est la **carte d'évidence** — la matrice rouge en dessous, de taille 120 par 240. C'est l'objet central de mon architecture. »*

### Phase 3 : la double sortie de la carte (45 s)

> *« Et c'est ici que commence ce qui rend ma proposition élégante : la carte d'évidence a **deux destinations**.*
>
> *Première destination — la flèche du bas : elle entre dans la **combinaison linéaire**. Cette combinaison fait simplement le produit matriciel entre la carte et les embeddings encodeur — vous voyez la flèche jaune en pointillés qui revient les chercher tout en haut. Le résultat passe ensuite par une projection linéaire finale qui produit la prédiction.*
>
> *Deuxième destination — la flèche orange à droite : la carte d'évidence sort **directement** comme explication. Aucun calcul supplémentaire, aucun back-pass, aucun sampling. La carte qui a servi à fabriquer la prédiction est aussi celle qui l'explique. C'est la **même** matrice, indissociable. »*

### Phase 4 : conclusion (30 s)

> *« Donc en résumé : on entre par le haut avec un contexte, on sort par le bas avec une prédiction et — gratuitement, dans le même calcul — par la droite avec une explication.*
>
> *C'est ça, l'intrinsèque pur. La carte d'évidence n'est pas une explication post-hoc qu'on calcule à part en faisant des hypothèses sur le modèle. C'est la **formule de prédiction elle-même**. »*

> 📌 **Notes Cabrel** :
> - **Phrase clé à appuyer** : "C'est la même matrice, indissociable."
> - Pause de 2 secondes après "intrinsèque pur" — laisser absorber
> - Si on vous demande "et l'attention multi-tête alors ?" — répondre que la carte est une *attention sur les positions du contexte*, distincte de l'attention multi-tête interne du décodeur (qui opère entre embeddings, pas sur les positions brutes)

---

## Niveau 3 — Version technique complète (lecture exhaustive du schéma)

À utiliser si quelqu'un demande explicitement *« expliquez-moi tout ce qui est sur le schéma »*, ou pour vous-même comme référence pour ne rien oublier.

### Le titre et le sous-titre

> *« Le titre du schéma est "Architecture SoftCAM-Transformer". Le sous-titre précise : "Adaptation intrinsèque du TimeSeriesTransformer (HuggingFace) pour la prévision FaaS". Cela pose tout de suite trois choses : c'est intrinsèque, c'est basé sur HuggingFace, et c'est pour la prévision FaaS — pas pour de la classification d'images. »*

### Bloc 1 — Contexte passé (en haut, fond bleu pâle)

> *« Le rectangle bleu en haut. Il représente l'entrée du modèle : un tenseur de forme **(240, 1)** — 240 minutes, chacune avec une valeur scalaire de charge. Le "1" peut grandir si on ajoute des covariables (lags, calendrier), mais pour le cas de base sur le cluster 4, c'est juste la charge. L'exemple en italique : "ex. charge cluster C4". »*

### Bloc 2 — Encodeur (gris)

> *« En dessous, le rectangle gris **Encodeur (inchangé)**. La mention "inchangé" est explicite : je reprends exactement l'encodeur du `TimeSeriesTransformer` HuggingFace de FAYAM. 4 couches, attention multi-têtes, dimension cachée 32. »*

### Bloc 3 — Embeddings encodeur (jaune, avec branche dérivée)

> *« Encore en dessous, le rectangle **jaune** : embeddings encodeur, tenseur **(240, 32)**. Le code couleur jaune signale "représentation latente apprise" — c'est important visuellement.*
>
> *Notez la **double sortie** : une flèche descend vers le décodeur, et une autre — la **flèche jaune en pointillés à droite** — contourne tout le schéma pour redescendre directement vers la combinaison linéaire en bas. La mention "réutilisés par la combinaison" sur la droite est explicite. C'est un détail crucial : les embeddings encodeur servent **deux fois**, à deux endroits différents. »*

### Bloc 4 — Décodeur (gris)

> *« Le rectangle gris **Décodeur (inchangé)**. Encore une fois, hérité tel quel de FAYAM. Il fait la cross-attention sur les embeddings encodeur et produit ses propres embeddings. »*

### Bloc 5 — Embeddings décodeur (jaune)

> *« Le rectangle jaune **embeddings décodeur**, tenseur **(120, 32)**. 120 vecteurs latents, un par pas futur à prédire. C'est ce que la `parameter_projection` recevait dans FAYAM. »*

### Bloc 6 — ★ Evidence Layer (rouge, large, en gras)

> *« Le bloc rouge encadré en gras — c'est le pivot du schéma. La couleur rouge signale "composant nouveau, contribution H1". Le texte explicite trois choses :*
>
> *Premièrement, "apprend pour chaque pas futur t un vecteur de 240 poids sur les positions du contexte" — c'est la sémantique fonctionnelle.*
>
> *Deuxièmement, "+ régularisation ElasticNet (L1 → parcimonie, L2 → lissage)" — c'est l'astuce qui rend les poids interprétables. Sans cette régularisation, le modèle apprendrait des poids éparpillés sans pic clair.*
>
> *Concrètement, l'evidence layer est juste une couche `Linear(32 → 240)` suivie d'un softmax. Très peu de paramètres. »*

### Bloc 7 — Carte d'évidence (rouge, avec double sortie)

> *« Le rectangle rouge **carte d'évidence**, matrice **(120, 240)**. C'est la sortie de l'evidence layer après softmax — donc les poids sont positifs et somment à 1 par ligne.*
>
> *Comme les embeddings encodeur, la carte a une **double sortie** : une vers la combinaison linéaire en bas, et — flèche orange à droite — directement vers le bloc explication. »*

### Bloc 8 — Combinaison linéaire (vert)

> *« Le rectangle vert **Combinaison linéaire**. Le vert signale "calcul interprétable". Le texte précise la formule : pour chaque pas futur t, `pred_latent[t] = Σᵢ carte[t,i] · emb_enc[i]`. C'est juste un produit matriciel `bmm(carte, emb_encodeur)` qui produit un tenseur de forme (120, 32) — le vecteur latent par pas futur. »*

### Bloc 9 — Projection linéaire finale (gris)

> *« Le petit rectangle gris **projection linéaire finale**. Une simple `Linear(32 → 3)` qui transforme les 32 dimensions latentes en 3 paramètres : `(loc, scale, df)` pour la distribution Student-t. Hérité de FAYAM. »*

### Bloc 10 — Prédiction (bleu, en bas)

> *« Le rectangle bleu en bas **PRÉDICTION**, tenseur **(120,)** — 120 valeurs prédictives. À l'inférence, c'est généralement le `loc` de la distribution Student-t, ou des échantillons si on veut des intervalles de confiance. »*

### Bloc 11 — Explication intrinsèque (orange, à droite)

> *« Et enfin, le rectangle orange à droite **EXPLICATION INTRINSÈQUE**. Il reçoit la flèche orange depuis la carte d'évidence. Le texte précise : "gratuite, fidèle par construction".*
>
> *"Gratuite" parce qu'aucun calcul supplémentaire n'est fait — la carte est déjà calculée pour la prédiction.*
>
> *"Fidèle par construction" parce que la carte n'est pas une approximation : c'est mathématiquement la formule qui a produit la prédiction. »*

### Légende complète (bas du schéma)

> *« Le bloc gris en bas est la légende. À gauche, le code couleur :*
>
> *— **Gris** : composant standard du TimeSeriesTransformer (inchangé)*
> *— **Jaune** : représentation latente apprise (embeddings)*
> *— **Rouge** : composant nouveau, contribution H1 — remplace `parameter_projection`*
> *— **Vert** : calcul interprétable (la prédiction est la formule de la combinaison)*
> *— **Orange** : sortie d'explication produite gratuitement (1 forward pass = prédiction + explication)*
>
> *À droite, l'encart "Comparaison clé vs FAYAM" résume la différence :*
> *— FAYAM : `parameter_projection` = boîte noire*
> *— SoftCAM-Transformer : `evidence_layer` = lisible*
> *— Coût d'inférence identique (1 forward pass)*
> *— Coût d'explication : 0 (vs 2^N pour SHAPformer)*
> *— Cible H1 : C4 (R²≥0.30, Spearman≥0.85 conservés) »*

---

## FAQ — questions qu'un expert peut poser sur le schéma

### Q1 : *« Pourquoi la carte est-elle de taille (120, 240) ? D'où viennent ces dimensions ? »*

> *« 120 = `prediction_length`, c'est-à-dire le nombre de pas futurs à prédire — l'horizon, fixé à 120 minutes pour FAYAM.
> 240 = `context_length`, c'est-à-dire le nombre de pas passés observés — la fenêtre d'historique, fixée à 240 minutes.
> Donc une ligne de la carte = un vecteur de 240 poids sur le passé pour fabriquer **un** pas futur. Et il y a 120 lignes parce qu'il y a 120 pas futurs à fabriquer. »*

### Q2 : *« Pourquoi la flèche jaune contourne-t-elle tout le schéma ? Pourquoi ne pas juste recalculer ? »*

> *« Parce que les embeddings encodeur sont produits une seule fois — c'est coûteux de les recalculer. Et conceptuellement, ils représentent "ce que le contexte signifie" : on veut que la combinaison linéaire opère sur cette représentation riche, pas sur les valeurs brutes (qui sont moins expressives) ni sur des nouvelles activations (qui ne seraient pas alignées avec ce que le décodeur a vu). C'est la **variante B** dans mon document de design — la plus fidèle à SoftCAM CNN original. »*

### Q3 : *« Si la carte sort directement comme explication, est-ce que la combinaison linéaire est vraiment nécessaire ? »*

> *« Oui — parce que sans elle, on n'a pas de prédiction. La carte est juste une matrice de poids ; pour produire un nombre, il faut multiplier ces poids par quelque chose. C'est la combinaison linéaire qui fait le travail. La beauté du design, c'est que **les poids utilisés pour la prédiction sont exactement ceux qu'on expose comme explication**. Pas de divergence possible. »*

### Q4 : *« Quel est le rôle exact de la projection linéaire finale ? Ça ne casse pas l'interprétabilité ? »*

> *« Non, parce qu'elle est **linéaire et globale**. Globale veut dire qu'elle est la même pour tous les pas futurs t — donc elle ne réintroduit pas d'asymétrie temporelle. Linéaire veut dire qu'elle préserve les contributions additives : si la carte dit "70% de l'embedding du lag 24h", la projection finale dit "70% de la projection de l'embedding du lag 24h" — la décomposition se propage. C'est le même argument que dans SoftCAM CNN, où le softmax après l'evidence layer ne casse pas l'interprétabilité. »*

### Q5 : *« Qu'est-ce qui est entraîné, qu'est-ce qui ne l'est pas ? »*

> *« Tout est ré-entraîné de bout en bout. L'encodeur, le décodeur, l'evidence layer, la projection finale — tout est entraîné conjointement avec la nouvelle loss (Student-t NLL + ElasticNet sur la carte). Je ne fais pas de fine-tuning sur un modèle pré-entraîné. C'est un nouveau modèle qui apprend depuis zéro avec cette structure contrainte. »*

### Q6 : *« Combien de paramètres ajoutez-vous au total ? »*

> *« Très peu. L'evidence layer = `Linear(32, 240)` = 32 × 240 = 7 680 paramètres. La projection finale = `Linear(32, 3)` = 96 paramètres. Total ajouté : ~7 800 paramètres. Le Transformer FAYAM a environ 150 000 paramètres. Donc je rajoute moins de 5%. C'est une modification architecturale **minimaliste** — exactement l'esprit SoftCAM. »*

### Q7 : *« Et si on change `prediction_length` ou `context_length` ? »*

> *« L'evidence layer dépend des deux : sa matrice est de taille `d_model × context_length`, et elle est appliquée à chaque pas de `prediction_length`. Donc si on passe à un horizon de 60 minutes au lieu de 120, l'evidence layer reste la même mais on a 60 lignes de carte au lieu de 120. Si on étend le contexte à 480 minutes, il faut redéfinir `Linear(32, 480)`. C'est paramétrique mais structurellement adaptable. »*

### Q8 : *« Pourquoi un softmax et pas un sigmoid sur chaque poids ? »*

> *« Parce que le softmax garantit que la somme des poids fait 1 — donc la carte se lit comme une **distribution de probabilité** sur le contexte. C'est très naturel à interpréter : "le modèle distribue 100% de son attention de cette façon". Avec sigmoid, chaque poids serait dans [0, 1] indépendamment — on perdrait le sens de "100%". Mais c'est un choix discutable et je l'expérimenterai si nécessaire. »*

---

## Notes finales pour Cabrel

1. **Le schéma est ton allié** : si tu bloques sur une explication, **reviens au schéma** et pointe le bloc concerné. Le visuel guide ta parole.

2. **Trois éléments à toujours mettre en valeur** :
   - Le bloc rouge (la nouveauté)
   - La flèche jaune en pointillés (la double sortie des embeddings encodeur)
   - La flèche orange (la sortie d'explication gratuite)

3. **Si tu n'as que 30 secondes** : *« Le schéma montre comment je remplace une couche opaque du Transformer FAYAM par une evidence layer transparente, qui produit en un seul forward pass à la fois la prédiction et la carte d'explication. »* C'est tout ce qu'il faut.

4. **Si on te demande de dessiner le schéma au tableau** : commence par les **5 blocs verticaux** (contexte → encodeur → embeddings encodeur → ... → prédiction), puis **ajoute** les deux branches (jaune en pointillés, orange à droite). Ne dessine pas tout en parallèle, ça embrouille.

5. **Confiance** : ce schéma est **ton** schéma. Tu l'as réfléchi, validé, débattu. Tu le connais mieux que personne.
