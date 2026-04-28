# Baseline FAYAM — Notes techniques

> Code original : `transformer_m2-main.zip` (reçu 2026-04-28).
> **Ne pas modifier ce dossier.** Toute adaptation va dans `src/models/`.

## Fichiers

| Fichier | Rôle |
|---------|------|
| `tsf_transf.py` | Modèle Transformer HuggingFace (contribution principale FAYAM) |
| `tsf_lstm.py` | Modèle LSTM (référence de comparaison) |
| `DataPreprocessing.py` | Pipeline complet : téléchargement Azure Trace → HDBSCAN → export |
| `dataToHub.py` | Découpage train/val/test + push HuggingFace Hub (`FaalSa/dataME`) |
| `main.py` | Point d'entrée : lance `DataPreprocessing` sur les 3 datasets |
| `test.py` | Évaluation du modèle |
| `requirements.txt` | Dépendances pip |

## Hyperparamètres Transformer (tsf_transf.py)

```python
freq             = "1T"          # 1 minute
prediction_length = 120          # prédit 120 minutes dans le futur
context_length   = 240           # utilise 240 minutes de contexte (2x prediction_length)
lags_sequence    = get_lags_for_frequency("1T")  # calculé par gluonts
encoder_layers   = 4
decoder_layers   = 4
d_model          = 32
embedding_dimension = [2]        # embedding de l'ID de série
num_static_categorical_features = 1
```

## Dataset HuggingFace

- Nom : `FaalSa/dataME`
- Colonne cible : `nbrconc` (nombre d'invocations concurrentes)
- Fréquence : minute (`freq='T'`)
- Date de départ synthétique : `2021-01-01`
- Split : train / validation / test découpés par fenêtres de `prediction_window=120`

## Discordance DBSCAN / HDBSCAN

Le mémoire FAYAM parle de **DBSCAN** (p. 73-74), mais le code `DataPreprocessing.py`
importe et utilise **`hdbscan`** (variante hiérarchique). Les paramètres exacts
(`epsilon`, `min_samples`) ne sont pas documentés dans le mémoire.
→ À clarifier avec l'encadreur avant de reproduire le clustering.

## Points d'attention pour la reproduction

1. Le dataset `FaalSa/dataME` est déjà sur HuggingFace — pas besoin de re-lancer
   `DataPreprocessing.py` pour les premières expériences.
2. `cardinality = [len(train_dataset)]` dépend du nombre de séries chargées —
   valeur variable selon les clusters utilisés.
3. `lags_sequence` est auto-calculé par gluonts pour `"1T"` — vérifier la valeur
   réelle avant de comparer avec nos runs.
4. Le token HuggingFace est hardcodé dans `dataToHub.py` — ne pas committer.
