import numpy as np

# Fixer la graine aléatoire pour numpy
np.random.seed(998)

from datasets import load_dataset

dataset = load_dataset("FaalSa/dataME")

train_example = dataset["train"][0]
validation_example = dataset["validation"][0]

# freq = "1Y"  -----> Année
# freq = "1M"  -----> Mois
# freq = "1D"  -----> Jour
# freq = "1H"  -----> Heure
# freq = "1T"  -----> Minute
# freq = "1S"  -----> Seconde

freq = "1T"
prediction_length = 120

assert len(train_example["target"]) + prediction_length == len(
    validation_example["target"]
)

import matplotlib.pyplot as plt

figure, axes = plt.subplots()
axes.plot(train_example["target"], color="blue")
axes.plot(validation_example["target"], color="red", alpha=0.5)

plt.show()

"""Let's split up the data:"""

train_dataset = dataset["train"]
test_dataset = dataset["test"]

from functools import lru_cache

import pandas as pd
import numpy as np


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

from functools import partial

train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

from gluonts.time_feature import get_lags_for_frequency

lags_sequence = get_lags_for_frequency(freq)

from gluonts.time_feature import time_features_from_frequency_str

time_features = time_features_from_frequency_str(freq)

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    # context length:
    context_length=prediction_length * 2,
    # lags coming from helper given the freq:
    lags_sequence=lags_sequence,
    # we'll add 2 time features ("month of year" and "age", see further):
    num_time_features=len(time_features) + 1,
    # we have a single static categorical feature, namely time series ID:
    num_static_categorical_features=1,
    # it has 366 possible values:
    cardinality=[len(train_dataset)],
    # the model will learn an embedding of size 2 for each of the 366 possible values:
    embedding_dimension=[2],

    # transformer params:
    # encoder = decoder = 4
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
)

model = TimeSeriesTransformerForPrediction(config)

model

model.config.distribution_output

from gluonts.time_feature import (
    time_features_from_frequency_str, #Fonction pour extraire les caractéristiques temporelles à partir de la fréquence des données temporelles.
    TimeFeature, #Classe qui représente une caractéristique temporelle.
    get_lags_for_frequency, #Fonction pour obtenir les retards (lags) appropriés pour une fréquence donnée.
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature, #Ajoute une caractéristique d'âge aux données temporelles
    AddObservedValuesIndicator, #Ajoute un indicateur pour indiquer si une valeur a été observée.
    AddTimeFeatures, #Ajoute diverses caractéristiques temporelles, telles que le jour de la semaine, le mois, etc.
    AsNumpyArray, #ransformation pour convertir les données en un tableau NumPy.
    Chain, #Permet de chaîner plusieurs transformations pour les appliquer séquentiellement.
    ExpectedNumInstanceSampler, #
    InstanceSplitter, #ivise les instances temporelles en sous-séquences.
    RemoveFields, #Supprime des champs spécifiques des données.
    SelectFields, #Sélectionne des champs spécifiques dans les données
    SetField, #
    TestSplitSampler, #
    Transformation, #
    ValidationSplitSampler, #
    VstackFeatures, #Transformation qui fusionne les caractéristiques de différentes instances en un seul tableau
    RenameFields, #Transformation pour renommer les champs dans les données
)

from transformers import PretrainedConfig


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: Ajout d'une autre caractéristique temporelle - âge (just a single number)
            # indiquant où se situe la valeur de la série temporelle dans sa durée de vie
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: Empilement vertical de toutes les caractéristiques temporelles dans le champ FEAT_TIME.
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: Renommage des champs pour correspondre aux noms Hugging Face
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

from gluonts.transform.sampler import InstanceSampler
from typing import Optional

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

from typing import Iterable

import torch
from gluonts.itertools import Cyclic, Cached
from gluonts.dataset.loader import as_stacked_batches


def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # We create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "test")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape, v.type())

# perform forward pass
outputs = model(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"]
    if config.num_static_categorical_features > 0
    else None,
    static_real_features=batch["static_real_features"]
    if config.num_static_real_features > 0
    else None,
    future_values=batch["future_values"],
    future_time_features=batch["future_time_features"],
    future_observed_mask=batch["future_observed_mask"],
    output_hidden_states=True,
)

print("Loss:", outputs.loss.item())

from accelerate import Accelerator
from torch.optim import AdamW

accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

model.train()
for epoch in range(51):
    print(epoch)
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()

        if idx % 100 == 0:
            print(loss.item())

# CODE SANS ARRONDIR LES VALEURS
model.eval()

forecasts = []

for batch in test_dataloader:
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    forecasts.append(outputs.sequences.cpu().numpy())

# # CODE - ARRONDIR LES VALEURS A LA VALEUR SUP
# model.eval()

# forecasts = []

# for batch in test_dataloader:
#    outputs = model.generate(
#        static_categorical_features=batch["static_categorical_features"].to(device)
#        if config.num_static_categorical_features > 0
#        else None,
#        static_real_features=batch["static_real_features"].to(device)
#        if config.num_static_real_features > 0
#        else None,
#        past_time_features=batch["past_time_features"].to(device),
#        past_values=batch["past_values"].to(device),
#        future_time_features=batch["future_time_features"].to(device),
#        past_observed_mask=batch["past_observed_mask"].to(device),
#    )
#    # Arrondir les valeurs prévues à la valeur supérieure
#    rounded_forecasts = np.ceil(outputs.sequences.cpu().numpy())
#    forecasts.append(rounded_forecasts)

# # CODE - CONSERVER PARTIE ENTIERE DE LA VALEUR
# model.eval()

# forecasts = []

# for batch in test_dataloader:
#     outputs = model.generate(
#         static_categorical_features=batch["static_categorical_features"].to(device)
#         if config.num_static_categorical_features > 0
#         else None,
#         static_real_features=batch["static_real_features"].to(device)
#         if config.num_static_real_features > 0
#         else None,
#         past_time_features=batch["past_time_features"].to(device),
#         past_values=batch["past_values"].to(device),
#         future_time_features=batch["future_time_features"].to(device),
#         past_observed_mask=batch["past_observed_mask"].to(device),
#     )
#     # Conserver juste la partie entière des valeurs prévues
#     rounded_forecasts = np.trunc(outputs.sequences.cpu().numpy())
#     forecasts.append(rounded_forecasts)

forecasts = np.vstack(forecasts)

from evaluate import load
from gluonts.time_feature import get_seasonality

mase_metric = load("evaluate-metric/mase")
smape_metric = load("evaluate-metric/smape")

forecast_median = np.median(forecasts, 1)

mase_metrics = []
smape_metrics = []
for item_id, ts in enumerate(test_dataset):
    training_data = ts["target"][:-prediction_length]
    ground_truth = ts["target"][-prediction_length:]
    mase = mase_metric.compute(
        predictions=forecast_median[item_id],
        references=np.array(ground_truth),
        training=np.array(training_data),
        periodicity=get_seasonality(freq),
    )
    mase_metrics.append(mase["mase"])

    smape = smape_metric.compute(
        predictions=forecast_median[item_id],
        references=np.array(ground_truth),
    )
    smape_metrics.append(smape["smape"])

print(f"MASE: {np.mean(mase_metrics)}")

print(f"sMAPE: {np.mean(smape_metrics)}")

plt.scatter(mase_metrics, smape_metrics, alpha=0.3)
plt.xlabel("MASE")
plt.ylabel("sMAPE")
plt.show()

import matplotlib.dates as mdates


def plot(ts_index):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_dataset[ts_index][FieldName.START],
        periods=len(test_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # print(np.median(forecasts[ts_index], axis=0))
    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(
        index[-2 * prediction_length :],
        test_dataset[ts_index]["target"][-2 * prediction_length :],
        label="actual",
    )

    plt.plot(
        index[-prediction_length:],
        np.median(forecasts[ts_index], axis=0),
        alpha=0.6,
        label="median",
    )

    plt.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0),
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0),
        alpha=0.3,
        interpolate=True,
        label="+/- 1-std",
    )
    plt.legend()
    plt.show()

ts_index = 0

index = pd.period_range(start=test_dataset[ts_index][FieldName.START], periods=len(test_dataset[ts_index][FieldName.TARGET]), freq=freq,).to_timestamp()

index = index[-prediction_length:]
val_median = np.median(forecasts[ts_index], axis=0)
list_val_median = list(val_median)

print(list_val_median)

# Initialisation de la liste pour stocker les valeurs prédites et réelles
predicted_values = []
actual_values = []

# Boucle pour parcourir les indices ts_index
for ts_index in range(len(test_dataset)):
    index = pd.period_range(start=test_dataset[ts_index][FieldName.START], periods=len(test_dataset[ts_index][FieldName.TARGET]), freq=freq,).to_timestamp()
    index = index[-prediction_length:]
    val_median = np.median(forecasts[ts_index], axis=0)
    list_val_median = list(val_median)
    print("ts_index:", ts_index)
    print("Predicted values:", list_val_median)
    print("Actual values:", np.array(test_dataset[ts_index]["target"][-prediction_length:]))
    predicted_values.append(list_val_median)
    actual_values.append(np.array(test_dataset[ts_index]["target"][-prediction_length:]))

# # Impression des valeurs prédites et réelles pour toutes les séries temporelles
# print("All Predicted values:", predicted_values)
# print("All Actual values:", actual_values)

for i in range(0,len(dataset["train"])):
  print("Time serie number : ", i)
  plot(i)

# Print predicted and actual values
print("Predicted values:", np.median(forecasts[ts_index], axis=0))
print("Actual values:", np.array(test_dataset[ts_index]["target"][-prediction_length:]))

"""**Exportation des données réelles et prédites pour faire plus de comparaisons sur les données**"""

# file_name = "fichier2.xlsx"

# # Create a DataFrame for predicted and actual values
# df = pd.DataFrame({
#     'Actual_values': test_dataset[ts_index]["target"][-prediction_length:],
#     'Predicted_values': np.median(forecasts[ts_index], axis=0)
# })

# # Save DataFrame to Excel
# df.to_excel(file_name, index=False)

# CODE POUR PLUSIEURS FICHIERS DISTINCS
import pandas as pd

# Boucle pour parcourir les indices ts_index
for ts_index in range(len(test_dataset)):
    # Création d'un DataFrame pour les valeurs prédites et réelles
    df = pd.DataFrame({
        'Actual_values': np.array(test_dataset[ts_index]["target"][-prediction_length:]),
        'Predicted_values': np.median(forecasts[ts_index], axis=0)
    })

    # Nom de fichier pour enregistrer les valeurs
    file_name = "fichier{}.xlsx".format(ts_index)

    # Enregistrement du DataFrame au format Excel
    df.to_excel(file_name, index=False)

"""**Calcul des métriques :** explained_variance_score, mean_absolute_percentage_error, mean_squared_error, r2_score"""

import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

def compute_metrics(before, actual):
    """
    Calcule plusieurs métriques d'évaluation pour comparer les prédictions et les valeurs réelles.

    Arguments :
    before -- Prédictions du modèle.
    actual -- Valeurs réelles.

    Returns :
    dict -- Un dictionnaire contenant les métriques d'évaluation.
    """
    explained_variance = explained_variance_score(actual, before)
    mape = mean_absolute_percentage_error(actual, before)
    rmse = np.sqrt(mean_squared_error(actual, before))
    normalized_rmse = rmse / (actual.max() - actual.min())
    r2 = r2_score(actual, before)
    spearman_corr, _ = spearmanr(actual, before)

    return {
        "Explained Variance": explained_variance,
        "MAPE": mape,
        "RMSE": rmse,
        "Normalized RMSE": normalized_rmse,
        "R2 Score": r2,
        "Spearman Correlation": spearman_corr
    }

# Boucle pour parcourir les indices ts_index
for ts_index in range(len(test_dataset)):
    # Calcul des métriques pour la série temporelle actuelle
    before = np.median(forecasts[ts_index], axis=0)
    actual = np.array(test_dataset[ts_index]["target"][-prediction_length:])
    metrics = compute_metrics(before, actual)

    # Affichage des métriques pour la série temporelle actuelle
    print("Metrics for time serie number:", ts_index)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
