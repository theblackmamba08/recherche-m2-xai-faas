# DEBRIEF — Résultats H1 SoftCAM-Transformer (2026-05-20)

> Présentation tenue le **2026-05-25**. Audience : encadreurs (Dr LACMOU, Dr DJOUMESSI, Pr KENGNE TCHENDJI).

---

## Ce qui a bien fonctionné

- Les résultats chiffrés (R²=0.7563, Spearman=0.987) ont été reçus positivement.
- La dissociation entraînement/inférence a été comprise et acceptée comme choix assumé.
- La structure en 7 hypothèses H1.A→H1.G a été appréciée comme cadre de validation.

---

## Retours critiques des encadreurs

### 1. Théorie d'abord, application ensuite

**Pattern requis pour la suite** : toute décision architecturale doit être précédée d'une justification théorique tirée de la littérature. L'ordre est : **théorie → application → résultats → interprétation**.

Exemple concret soulevé : la connexion résiduelle entre `dec_output` et `h_evidence` (le mélange additif `h = (1-mix)·dec_output + mix·LN(bmm(M, enc_hidden))`) — pourquoi ce design ? Quel papier le justifie ? (Réponse attendue : skip connections de He et al. 2016 / ResNet, ou analogie cross-attention Vaswani 2017.)

**Action** : avant tout retrain ou modification architecturale, rédiger d'abord la justification théorique dans `memoire/03-contribution/`.

---

### 2. Seuil R² dans la littérature

Question posée : existe-t-il dans la littérature un seuil au-dessus duquel un modèle est considéré "bon" en termes de R² pour les séries temporelles ?

**À creuser** : cf. `memoire/00-meta/QUESTIONS-OUVERTES.md`.

---

### 3. Mesure de confiance en M

Question posée : comment mesurer la confiance qu'on doit accorder à la carte d'évidence M ?

Pistes possibles : calibration (ECE/MCE), intervalles de confiance sur M par bootstrap, stabilité de M sur plusieurs runs (variance de M à input fixe).

**À creuser** : cf. `memoire/00-meta/QUESTIONS-OUVERTES.md`.

---

### 4. Schémas visuels obligatoires

Chaque étape de la présentation doit être accompagnée d'images réelles issues de l'exécution des notebooks — pas uniquement des figures TikZ schématiques. Cela inclut : courbes de loss, distributions de M, cartes de chaleur des attention weights, scatter plots effectifs.

**Action** : exporter systématiquement les figures des notebooks et les intégrer dans les slides suivantes.

---

### 5. Évaluation de l'explicabilité : quantitative ET qualitative

Les encadreurs ont noté l'absence d'une évaluation formelle de l'explicabilité. H1.F (comprehensiveness) et H1.G (sufficiency) sont des débuts, mais insuffisants.

**Quantitatif attendu** : métriques formelles (comprehensiveness/sufficiency complètes, activation precision, activation sensitivity adaptées au temporel).

**Qualitatif attendu** : visualisation de M sur des exemples concrets, interprétation par un expert métier ("est-ce que ces moments pointés par M ont du sens ?"), comparaison M vs attention weights.

---

### 6. Renommer "SoftCAM-Transformer"

Le nom crée de la confusion : "SoftCAM" évoque le papier Djoumessi & Berens (2025) sur les images, pas les séries temporelles. Les encadreurs demandent un nom qui référence exactement ce qui est fait.

**Options à explorer** :
- `TemporalEvidenceTransformer` (TET)
- `EvidenceLayerTransformer` (ELT)
- `FaaSEvidenceNet`
- Autre — à trancher avec les encadreurs.

**À décider** : cf. `memoire/00-meta/QUESTIONS-OUVERTES.md`.

---

### 7. Révision architecturale : dot product au lieu du mélange additif

**Retour majeur** : remplacer le mélange additif actuel par un **dot product** entre `dec_output` et M. Cette formulation permettrait de mesurer directement le poids de la présence de M dans la prédiction finale, et facilite l'interprétation.

Architecture actuelle :
```
h = (1 - mix) · dec_output + mix · LN(bmm(M, enc_hidden))
```

Architecture proposée (à explorer) :
```
h = dec_output ⊙ M   (dot product / element-wise)
```
ou une variante similaire à préciser selon la littérature.

**Condition sine qua non** : justifier ce choix dans la littérature avant d'implémenter.

---

### 8. Justification théorique de chaque choix architectural

Pour chaque brique de l'architecture actuelle, un papier de référence doit être cité :

| Brique | Justification attendue |
|--------|----------------------|
| Connexion résiduelle (mélange additif) | He et al. 2016 (ResNet) / Vaswani 2017 |
| softmax pour produire M | Vaswani 2017 (attention mechanism) |
| bmm(M, enc_hidden) | Cross-attention style |
| LayerNorm sur h_evidence | Ba et al. 2016 |
| ElasticNet (L1+L2+entropie) | Zou & Hastie 2005 + motivation entropie |

---

## Actions immédiates

1. 🔴 **Revoir l'architecture** : explorer le dot product, justifier depuis la littérature.
2. 🔴 **Renommer le modèle** : proposer 2-3 options aux encadreurs.
3. 🔴 **Évaluation explicabilité** : compléter quantitatif + qualitatif.
4. 🔴 **Exporter figures notebooks** : intégrer dans les prochains slides.
5. 🟡 **Seuil R²** : chercher dans la littérature.
6. 🟡 **Mesure de confiance en M** : explorer les pistes (calibration, variance, bootstrap).
7. 🟡 **Si résultats dot product décevants** : préparer présentation des résultats actuels (R²=0.7563) avec cadre théorique renforcé.
