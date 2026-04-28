# Transformer-Based Model for Cold Start Mitigation in FaaS Architecture
Ce dépôt contient des modèles pour la prévision de séries temporelles utilisant des architectures Transformers et LSTM. 

## Notes

* **requirements.txt**

  * Contient toutes les dépendances nécessaires à la bonne exécution des différents scripts.
  * Installation rapide :

    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Time Series Transformer

Ce dossier contient deux scripts principaux pour préparer et entraîner un modèle Transformer sur des séries temporelles :

### 1. `dataToHub.py`

Ce script permet de préparer et de publier de nouveaux jeux de données sur le Hub Hugging Face.

**Utilisation :**

1. Placez-vous dans un dossier ne contenant que le(s) fichier(s) de jeu de données à envoyer.
2. Les datasets doivent être au format CSV sans en-tête, avec une seule colonne nommée :

   ```csv
   nbrconc
   Val_1
   Val_2
   …
   Val_n
   ```
3. Lancez l'exécution :

   ```bash
   python dataToHub.py
   ```

**Paramètres modifiables :**

* **Ligne 66** : `df['target'] = df['nbrconc']`

  * Adaptez `nbrconc` au nom de la colonne de votre dataset.
* **Ligne 144**: `data_directory = '.'`

  * Répertoire local où réside le dataset.
* **Ligne 145**: `prediction_window = 120`

  * Longueur de la fenêtre de prédiction souhaitée.
* **Ligne 146**: `dataset_name = 'FaalSa/nomDataset'`

  * Changez `nomDataset` par le nom souhaité pour le Hub.
* **Ligne 147**: `token`

  * Votre token d'authentification Hugging Face.
* **Ligne 148**: `freq = 'T'`

  * Fréquence d'enregistrement adaptée à votre série temporelle (ex. `'H'` pour horaire, `'D'` pour journalier).
* **Ligne 149**: `output_dir = '.'`

  * Répertoire de sortie pour les fichiers générés.

### 2. `tsf_transf.py`

Ce script entraîne un modèle Transformer sur un dataset déjà disponible sur le Hub.

**Utilisation :**

```bash
python tsf_transf.py
```

**Paramètres modifiables :**

* **Ligne 4**: `np.random.seed(998)`

  * Graine aléatoire pour la reproductibilité.
* **Ligne 8**: `dataset = load_dataset("FaalSa/dataME")`

  * Nom du dataset à charger depuis le Hub (ex. `dataME`).
* **Ligne 20**: `freq = "1T"`

  * Fréquence temporelle, doit correspondre à celle du dataset.
* **Ligne 21**: `prediction_length = 120`

  * Taille de la fenêtre de prédiction, en pas de temps.
* **Ligne 83, 87–89**:

  ```python
  embedding_dimension = [2]
  encoder_layers = 4
  decoder_layers = 4
  d_model = 32
  ```

  * Paramètres de l'architecture Transformer.
* **Ligne 382–383**:

  ```python
  batch_size = 256
  num_batches_per_epoch = 100
  ```

  * Taille des batchs et nombre de mini-batchs par époque pour l'entraînement.
* **Ligne 390**: `batch_size = 64`

  * Taille du batch pour le jeu de test.
* **Ligne 432**: `for epoch in range(51):`

  * Nombre d'époques (ici 51).

**Résultats générés :**

* Figure du dataset utilisé pour l'entraînement.
* Graphiques d'évolution des métriques (MASE, sMAPE).
* Courbe des prédictions vs. valeurs réelles.
* Résumé des résultats de métriques dans la console.
* Fichier Excel (`.xlsx`) contenant les valeurs réelles et prédites.

---

## 🧠 Modèle LSTM (tsf_lstm.py)

Le script `tsf_lstm.py` implémente un modèle LSTM sur les mêmes données que pour le Transformer.

### Format du dataset

* Le fichier CSV doit s'appeler `datasetLSTM.csv`.
* Il doit contenir une seule colonne nommée :

  ```csv
  nbrconc
  Val_1
  Val_2
  …
  Val_n
  ```

### Paramètres modifiables

* **Ligne 24**: `np.random.seed(998)`

  * Graine aléatoire.
* **Ligne 52**: `seq_length = 4`

  * Longueur de la séquence d'entrée.
* **Ligne 55–56**:

  ```python
  train_size = int(len(y) * 0.67)
  test_size = len(y) - train_size
  ```

  * Répartition train/test.
* **Ligne 120**: `num_epochs = 2000`

  * Nombre d'époques.
* **Ligne 121**: `learning_rate = 0.01`

  * Taux d'apprentissage.
* **Ligne 123–127**:

  ```python
  input_size = 1
  hidden_size = 2
  num_layers = 1
  num_classes = 1
  ```

  * Paramètres du réseau LSTM.
* **Ligne 149 & 227**:

  * Type de LSTM utilisé (choix de la variante).

### Résultats générés

* Figure du dataset et de la séquence d'entraînement.
* Courbe des prédictions vs. valeurs réelles.
* Résumé des métriques calculées.
* Fichier Excel (`.xlsx`) avec valeurs réelles et prédites.
