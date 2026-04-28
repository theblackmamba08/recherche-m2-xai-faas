import os
import pandas as pd
from datetime import datetime, timedelta

from gluonts.dataset.pandas import PandasDataset
from gluonts.itertools import Map
from datasets import Dataset, Features, Value, Sequence
from huggingface_hub import login


class HFTimeSeriesDatasetProcessor:
    """
    Classe pour traiter des datasets et les pousser sur le Hugging Face Hub.

    Paramètres modifiables à l'initialisation :
        data_directory (str) : Chemin du répertoire contenant les CSV d'origine.
        prediction_window (int) : Taille de la fenêtre de prédiction (nombre de pas de temps).
        dataset_name (str) : Nom du dataset à créer sur le Hub.
        token (str) : Token d'accès à Hugging Face Hub.
        freq (str) : Fréquence temporelle de la série, au format pandas (ex. 'T' pour minute).
        output_dir (str) : Répertoire où seront créés train.csv, validation.csv et test.csv.
    """

    def __init__(
        self,
        data_directory: str,
        prediction_window: int,
        dataset_name: str,
        token: str,
        freq: str = 'T',
        output_dir: str = None,
    ):
        self.data_directory = data_directory
        self.prediction_window = prediction_window
        self.dataset_name = dataset_name
        self.token = token
        self.freq = freq  # pandas frequency string
        self.output_dir = output_dir or os.getcwd()

        self.train_file = os.path.join(self.output_dir, 'train.csv')
        self.validation_file = os.path.join(self.output_dir, 'validation.csv')
        self.test_file = os.path.join(self.output_dir, 'test.csv')

    def split_csv_files(self):
        """
        Lit tous les datasets du répertoire, ajoute les colonnes date, target et item_id,
        et les découpe en train, validation, test.
        """
        # Réinitialiser ou créer les fichiers de sortie
        for f in [self.train_file, self.validation_file, self.test_file]:
            if os.path.exists(f):
                os.remove(f)

        # Parcours de chaque CSV source
        for filename in os.listdir(self.data_directory):
            if not filename.lower().endswith('.csv'):
                continue

            filepath = os.path.join(self.data_directory, filename)
            raw = pd.read_csv(filepath)

            # Génération colonne date en index
            start_date = datetime(2021, 1, 1)
            df = raw.copy()
            df['date'] = pd.date_range(start=start_date, periods=len(df), freq=self.freq)
            df['target'] = df['nbrconc']
            df['item_id'] = os.path.splitext(filename)[0]

            # Découpage selon prediction_window
            n = len(df)
            idx_val = n - self.prediction_window
            idx_train = idx_val - self.prediction_window

            df.iloc[:idx_train].to_csv(self.train_file, index=False, mode='a')
            df.iloc[:idx_val].to_csv(self.validation_file, index=False, mode='a')
            df.to_csv(self.test_file, index=False, mode='a')

    def process_csv(self, file_path: str):
        """
        Charge un CSV et le transforme en liste d'exemples GluonTS.
        """
        df = pd.read_csv(
            file_path,
            parse_dates=['date'],
            date_parser=lambda x: pd.to_datetime(x, format=None, utc=False)
        )
        df = df.set_index('date')
        # Assure que l'index a bien une fréquence
        if df.index.freq is None:
            df.index = pd.DatetimeIndex(df.index.values, freq=self.freq)

        # Création du PandasDataset en précisant la fréquence
        ds = PandasDataset.from_long_dataframe(
            df,
            target='target',
            item_id='item_id',
            freq=self.freq
        )

        # Ajout du champ start et feat_static_cat
        class ProcessStartField:
            ts_id = 0

            def __call__(self, data):
                data['start'] = data['start'].to_timestamp()
                data['feat_static_cat'] = [self.ts_id]
                self.ts_id += 1
                return data

        processor = ProcessStartField()
        return list(Map(processor, ds))

    def push_to_hf(self):
        """
        Construit et pousse les datasets sur Hugging Face Hub.
        """
        # Connexion au Hugging Face Hub
        login(token=self.token, add_to_git_credential=True)

        # Définition des features HF
        features = Features({
            'start': Value('timestamp[s]'),
            'target': Sequence(Value('float32')),
            'feat_static_cat': Sequence(Value('uint64')),
            'item_id': Value('string'),
        })

        # Pour chaque split, charger, construire et pousser
        for split, file in [('train', self.train_file),
                            ('validation', self.validation_file),
                            ('test', self.test_file)]:
            list_ds = self.process_csv(file)
            dataset = Dataset.from_list(list_ds, features=features)
            dataset.push_to_hub(self.dataset_name, split=split)

    def run(self):
        """Exécute l'ensemble du pipeline et retourne 'Ok'."""
        self.split_csv_files()
        self.push_to_hf()
        return 'Ok'

if __name__ == '__main__':
    processor = HFTimeSeriesDatasetProcessor(
        data_directory='.',
        prediction_window=120,
        dataset_name='FaalSa/nomDataset',
        token='<HF_TOKEN_REDACTED>',  # token original supprimé — utiliser os.environ['HF_TOKEN']
        freq='T',  # fréquence minute
        output_dir='.'
    )
    result = processor.run()
    print(result)