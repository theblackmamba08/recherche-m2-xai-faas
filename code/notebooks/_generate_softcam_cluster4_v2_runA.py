"""Génère code/notebooks/softcam-cluster4-v2-runA.ipynb (Phase 2 — H1 v2 Run A).

Sanity check : v2 avec use_evidence_layer=False → comportement parent FAYAM strict.
Protocole d'entraînement aligné sur baseline-cluster4.ipynb (51 epochs, pas d'early stopping).

Objectif : confirmer que la pipeline (data, GluonTS, training loop) reproduit le R² FAYAM
(≈ 0.37 sur Cluster 4). Si oui, le bug v1 est isolé dans l'evidence layer.

Lance : python _generate_softcam_cluster4_v2_runA.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4-v2-runA.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


# ──────────────────────────────────────────────────────────────────────────
md(
    """# SoftCAM-Transformer v2 — Run A (sanity check pipeline)

Test de contrôle pour isoler la cause du NO-GO v1 (Test R²=-6.16, Spearman=-0.87 le 2026-05-17).

| Champ | Valeur |
|-------|--------|
| **Modèle** | `SoftCAMTransformerV2ForPrediction(use_evidence_layer=False)` |
| **Comportement attendu** | identique au Transformer FAYAM parent (evidence layer désactivé) |
| **Protocole** | **51 epochs full, pas d'early stopping** (aligné sur `baseline-cluster4.ipynb`) |
| **Cible** | Reproduire **R² ≈ 0.37, Spearman ≈ 0.92** (résultats baseline FAYAM Cluster 4) |

## Décisions à l'issue de ce run

- **Si R² ≈ 0.37 (PASS)** → la pipeline est saine. Le bug v1 est **forcément** dans l'evidence layer → on enchaîne avec un Run B (`use_evidence_layer=True, evidence_mix=0.3`) pour valider l'hypothèse "info bottleneck".
- **Si R² très différent** → bug pipeline (data, lags, transforms, training loop). Corriger ça avant tout — l'evidence layer ne pourra jamais marcher sur des bases pourries.

> Avant de lancer : Runtime → Change runtime type → **T4 GPU**.
"""
)

md("## 1 — Setup")

code(
    """import subprocess
gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu.stdout if gpu.returncode == 0 else 'Pas de GPU — vérifier le runtime.')"""
)

code(
    """%%capture
!pip install -q transformers datasets "gluonts[torch]" accelerate evaluate scipy scikit-learn tqdm openpyxl ujson"""
)

md(
    """## 2 — Récupération du code (modèle v2 défini dans `code/src/models/`)

On clone le dépôt pour disposer de `src/models/softcam_transformer_v2.py`."""
)

code(
    """import os, sys

REPO_URL = 'https://github.com/theblackmamba08/recherche-m2-xai-faas.git'
REPO_DIR = '/content/recherche-m2-xai-faas'

ipy = get_ipython()
if not os.path.isdir(REPO_DIR):
    ipy.system(f'git clone {REPO_URL} {REPO_DIR}')
else:
    ipy.system(f'git -C {REPO_DIR} pull')

if f'{REPO_DIR}/code' not in sys.path:
    sys.path.insert(0, f'{REPO_DIR}/code')

models_dir = f'{REPO_DIR}/code/src/models'
if not os.path.isdir(models_dir):
    raise FileNotFoundError(
        f'{models_dir} introuvable.\\n'
        'Essaie : Runtime → Disconnect and delete runtime, puis relance la cellule.'
    )
print('Repo prêt :', os.listdir(models_dir))"""
)

md("## 3 — Imports, seed, config")

code(
    """import random, json, gc
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache, partial
from pathlib import Path
from tqdm.notebook import tqdm

SEED = 998   # FAYAM seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch : {torch.__version__}')
print(f'Device  : {device}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')"""
)

code(
    """# ── Hyperparams aligned with baseline-cluster4.ipynb (FAYAM) ────
FREQ              = '1T'
PREDICTION_LENGTH = 120
CONTEXT_LENGTH    = 240
D_MODEL           = 32
N_HEADS           = 2
ENCODER_LAYERS    = 4
DECODER_LAYERS    = 4
EMBEDDING_DIM     = [2]
DROPOUT           = 0.1
BATCH_SIZE_TRAIN  = 256
BATCH_SIZE_TEST   = 64
NUM_BATCHES_EPOCH = 100
LR                = 6e-4
BETAS             = (0.9, 0.95)
WEIGHT_DECAY      = 1e-1
GRAD_CLIP_NORM    = 1.0
NUM_EPOCHS        = 51                # ← FAYAM protocol, pas d'early stopping

# ── Diagnostic flags (v2) ────────────────────────────────────
USE_EVIDENCE_LAYER = False            # ← sanity check : Evidence Layer DÉSACTIVÉE
EVIDENCE_MIX       = 0.0              # ignored when use_evidence_layer=False
ALPHA_L1           = 0.0
BETA_L2            = 1e-3
GAMMA_ENTROPY      = 1e-3

# ── Baseline FAYAM Cluster 4 à reproduire ────────────────────
FAYAM_BASELINE_R2    = 0.3701
FAYAM_BASELINE_SPEAR = 0.9201

CLUSTER_ID = 4
RUN_NAME   = f'softcam-cluster{CLUSTER_ID}-v2-runA-sanity'
print(f'Run : {RUN_NAME}')
print(f'use_evidence_layer = {USE_EVIDENCE_LAYER}  (False → comportement parent FAYAM strict)')
print(f'NUM_EPOCHS         = {NUM_EPOCHS}  (full, pas d\\'early stopping)')
print(f'Baseline à reproduire : R²={FAYAM_BASELINE_R2}, Spearman={FAYAM_BASELINE_SPEAR}')"""
)

md("## 4 — Google Drive (checkpoints + résultats)")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'
for subdir in ['checkpoints', 'results', 'logs']:
    os.makedirs(f'{DRIVE_BASE}/{subdir}', exist_ok=True)
print(f'Dossier run : {DRIVE_BASE}')"""
)

md(
    """## 5 — Chargement Cluster 4 (identique à baseline-cluster4)"""
)

code(
    """from datasets import Dataset

try:
    from google.colab import drive as _drive  # noqa: F401
    DATA_DIR = Path('/content/drive/MyDrive/Recherche/Datasets')
except ImportError:
    DATA_DIR = Path('../../memoire/06-datasets/raw').resolve()

START_DATE = '2021-01-01 00:00:00'

df = pd.read_csv(DATA_DIR / f'cluster_{CLUSTER_ID}.csv', index_col='Function')
all_series = []
for func_id, row in df.iterrows():
    all_series.append({
        'function_id': int(func_id),
        'cluster':     CLUSTER_ID,
        'target_full': row.values.astype(np.float32),
    })

print(f'Cluster {CLUSTER_ID} — {len(all_series)} fonctions :')
for s in all_series:
    print(f"  function_id={s['function_id']}  longueur={len(s['target_full'])}  "
          f"moy={s['target_full'].mean():.0f}  max={s['target_full'].max():.0f}")

train_rows, val_rows, test_rows = [], [], []
for ts_idx, s in enumerate(all_series):
    base = {'start': START_DATE, 'feat_static_cat': [ts_idx],
            'cluster': s['cluster'], 'function_id': s['function_id']}
    train_rows.append({**base, 'target': s['target_full'][:-2*PREDICTION_LENGTH].tolist()})
    val_rows.append(  {**base, 'target': s['target_full'][:-PREDICTION_LENGTH].tolist()})
    test_rows.append( {**base, 'target': s['target_full'].tolist()})

train_dataset = Dataset.from_list(train_rows)
val_dataset   = Dataset.from_list(val_rows)
test_dataset  = Dataset.from_list(test_rows)

print(f'\\nDatasets : train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}')"""
)

md("## 6 — Pipeline GluonTS (identique à FAYAM)")

code(
    """from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature, AddObservedValuesIndicator, AddTimeFeatures,
    AsNumpyArray, Chain, ExpectedNumInstanceSampler, InstanceSplitter,
    RemoveFields, TestSplitSampler, ValidationSplitSampler,
    VstackFeatures, RenameFields,
)
from gluonts.itertools import Cyclic, Cached
from gluonts.dataset.loader import as_stacked_batches

lags_sequence = get_lags_for_frequency(FREQ)
time_features = time_features_from_frequency_str(FREQ)

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch['start'] = [convert_to_pandas_period(d, freq) for d in batch['start']]
    return batch

for ds in (train_dataset, val_dataset, test_dataset):
    ds.set_transform(partial(transform_start_field, freq=FREQ))

print(f'Lags : {len(lags_sequence)} (max={max(lags_sequence)})')
print(f'Time features : {len(time_features)}')"""
)

code(
    """def create_transformation(freq, config):
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)
    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + ([AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=int)]
           if config.num_static_categorical_features > 0 else [])
        + [AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
           AddObservedValuesIndicator(target_field=FieldName.TARGET,
                                      output_field=FieldName.OBSERVED_VALUES),
           AddTimeFeatures(start_field=FieldName.START, target_field=FieldName.TARGET,
                           output_field=FieldName.FEAT_TIME,
                           time_features=time_features_from_frequency_str(freq),
                           pred_length=config.prediction_length),
           AddAgeFeature(target_field=FieldName.TARGET, output_field=FieldName.FEAT_AGE,
                         pred_length=config.prediction_length, log_scale=True),
           VstackFeatures(output_field=FieldName.FEAT_TIME,
                          input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]),
           RenameFields(mapping={
               FieldName.FEAT_STATIC_CAT: 'static_categorical_features',
               FieldName.FEAT_TIME:       'time_features',
               FieldName.TARGET:          'values',
               FieldName.OBSERVED_VALUES: 'observed_mask',
           })]
    )

def create_instance_splitter(config, mode):
    sampler = {
        'train':      ExpectedNumInstanceSampler(num_instances=1.0,
                                                  min_future=config.prediction_length),
        'validation': ValidationSplitSampler(min_future=config.prediction_length),
        'test':       TestSplitSampler(),
    }[mode]
    return InstanceSplitter(
        target_field='values', is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START, forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=['time_features', 'observed_mask'],
    )

def _pred_fields(config):
    f = ['past_time_features', 'past_values', 'past_observed_mask', 'future_time_features']
    if config.num_static_categorical_features > 0:
        f.append('static_categorical_features')
    return f

def create_train_dataloader(config, freq, data, batch_size, num_batches_per_epoch):
    fields = _pred_fields(config) + ['future_values', 'future_observed_mask']
    tr = create_transformation(freq, config)
    td = Cached(tr.apply(data, is_train=True))
    sp = create_instance_splitter(config, 'train')
    return as_stacked_batches(sp.apply(Cyclic(td).stream()),
                              batch_size=batch_size, field_names=fields,
                              output_type=torch.tensor,
                              num_batches_per_epoch=num_batches_per_epoch)

def create_backtest_dataloader(config, freq, data, batch_size):
    tr = create_transformation(freq, config)
    sp = create_instance_splitter(config, 'validation')
    return as_stacked_batches(sp.apply(tr.apply(data), is_train=True),
                              batch_size=batch_size, output_type=torch.tensor,
                              field_names=_pred_fields(config))

print('Pipeline GluonTS prête.')"""
)

md(
    """## 7 — Construction du modèle v2 (avec `use_evidence_layer=False`)

Important : on instancie bien `SoftCAMTransformerV2ForPrediction` (le sous-classement v2), pas le parent HuggingFace directement. C'est le **toggle interne** `use_evidence_layer=False` qui fait que le modèle se comporte comme FAYAM. Cela teste à la fois la pipeline ET la mécanique de subclassing (hook encodeur ne perturbe rien, override `output_params` ne fait rien quand toggle off).
"""
)

code(
    """from src.models.softcam_transformer_v2 import (
    SoftCAMTransformerV2Config,
    SoftCAMTransformerV2ForPrediction,
)

cfg = SoftCAMTransformerV2Config(
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    lags_sequence=lags_sequence,
    num_time_features=len(time_features) + 1,
    num_static_categorical_features=1,
    cardinality=[len(train_dataset)],
    embedding_dimension=EMBEDDING_DIM,
    encoder_layers=ENCODER_LAYERS,
    decoder_layers=DECODER_LAYERS,
    d_model=D_MODEL,
    encoder_attention_heads=N_HEADS,
    decoder_attention_heads=N_HEADS,
    encoder_ffn_dim=max(D_MODEL, 32),
    decoder_ffn_dim=max(D_MODEL, 32),
    dropout=DROPOUT,
    # v2 diagnostic flags
    use_evidence_layer=USE_EVIDENCE_LAYER,    # False → parent strict
    evidence_mix=EVIDENCE_MIX,                # ignored
    alpha_l1=ALPHA_L1,
    beta_l2=BETA_L2,
    gamma_entropy=GAMMA_ENTROPY,
)

model = SoftCAMTransformerV2ForPrediction(cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_evidence = sum(p.numel() for p in model.evidence_linear.parameters())
print(f'Paramètres totaux       : {n_params:,}')
print(f'  dont evidence_linear  : {n_evidence:,}  (présent mais NON utilisé)')
print(f'\\nuse_evidence_layer={USE_EVIDENCE_LAYER} → comportement attendu : parent FAYAM strict')"""
)

md("## 8 — Helper d'évaluation (R² + Spearman)")

code(
    """from sklearn.metrics import r2_score
from scipy.stats import spearmanr

@torch.no_grad()
def evaluate(model, dataloader, dataset, config, device, prefix=''):
    model.eval()
    forecasts = []
    for b in dataloader:
        out = model.generate(
            static_categorical_features=b['static_categorical_features'].to(device)
                if config.num_static_categorical_features > 0 else None,
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
        )
        forecasts.append(out.sequences.cpu().numpy())
    forecasts = np.vstack(forecasts)
    forecast_median = np.median(forecasts, axis=1)

    r2s, spears = [], []
    for ts_idx in range(len(dataset)):
        target = np.array(dataset[ts_idx]['target'])
        actual = target[-config.prediction_length:]
        pred   = forecast_median[ts_idx]
        r2s.append(r2_score(actual, pred))
        rho, _ = spearmanr(actual, pred)
        spears.append(rho)
    r2 = float(np.mean(r2s)); spear = float(np.mean(spears))
    if prefix:
        print(f'{prefix}  R²={r2:.4f}  Spearman={spear:.4f}')
        print(f'{prefix}  per-series R²       : {[f\"{x:.3f}\" for x in r2s]}')
        print(f'{prefix}  per-series Spearman : {[f\"{x:.3f}\" for x in spears]}')
    return r2, spear, r2s, spears

print('Helper évaluation prêt.')"""
)

md(
    """## 9 — Entraînement 51 epochs full (pas d'early stopping)

Protocole FAYAM exact : on enregistre uniquement le checkpoint final (epoch 50), comme le faisait `baseline-cluster4.ipynb`. Pas de patience, pas de best-checkpoint — on veut une comparaison apples-to-apples.
"""
)

code(
    """from torch.optim import AdamW

train_loader = create_train_dataloader(cfg, FREQ, train_dataset, BATCH_SIZE_TRAIN, NUM_BATCHES_EPOCH)
val_loader   = create_backtest_dataloader(cfg, FREQ, val_dataset, BATCH_SIZE_TEST)

optimizer = AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

history = {'train_loss': [], 'val_r2': [], 'val_spear': []}
ckpt_path = f'{DRIVE_BASE}/checkpoints/v2_runA_final.pt'

for epoch in range(NUM_EPOCHS):
    model.train()
    losses = []
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}/{NUM_EPOCHS-1}', leave=False)
    for b in pbar:
        optimizer.zero_grad()
        out = model(
            static_categorical_features=b['static_categorical_features'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            future_values=b['future_values'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_observed_mask=b['future_observed_mask'].to(device),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(out.loss.item())
        pbar.set_postfix({'loss': f'{out.loss.item():.3f}'})

    train_loss = float(np.mean(losses))
    val_r2, val_spear, _, _ = evaluate(model, val_loader, val_dataset, cfg, device)
    history['train_loss'].append(train_loss)
    history['val_r2'].append(val_r2)
    history['val_spear'].append(val_spear)
    print(f'Epoch {epoch:03d}  loss={train_loss:.4f}  val_r2={val_r2:.4f}  val_spear={val_spear:.4f}')

# Final checkpoint (pas de best — proto FAYAM)
torch.save({
    'epoch': NUM_EPOCHS - 1, 'model': model.state_dict(),
    'val_r2': history['val_r2'][-1], 'val_spear': history['val_spear'][-1],
    'history': history, 'hyperparameters': cfg.to_dict(),
}, ckpt_path)
print(f'\\nEntraînement terminé — final val R² = {history[\"val_r2\"][-1]:.4f}')"""
)

md("## 10 — Courbes de loss + val R²")

code(
    """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ep = list(range(len(history['train_loss'])))

axes[0].plot(ep, history['train_loss'], lw=1.5, color='steelblue')
axes[0].set_xlabel('Époque'); axes[0].set_ylabel('Train loss (NLL)')
axes[0].set_title('Train loss')
axes[0].grid(alpha=0.3)

axes[1].plot(ep, history['val_r2'], lw=1.5, color='tomato', label='v2 (use=False) val R²')
axes[1].axhline(FAYAM_BASELINE_R2, color='gray', lw=1, ls='--', label=f'FAYAM ({FAYAM_BASELINE_R2})')
axes[1].set_xlabel('Époque'); axes[1].set_ylabel('R²')
axes[1].set_title('Validation R²')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/training_curves.png', dpi=150)
plt.show()"""
)

md(
    """## 11 — Évaluation finale sur TEST + verdict sanity check"""
)

code(
    """test_loader = create_backtest_dataloader(cfg, FREQ, test_dataset, BATCH_SIZE_TEST)
test_r2, test_spear, test_r2s, test_spears = evaluate(
    model, test_loader, test_dataset, cfg, device, prefix='TEST'
)

results = {
    'run': RUN_NAME,
    'mode': 'sanity-check-parent-FAYAM',
    'use_evidence_layer': USE_EVIDENCE_LAYER,
    'num_epochs': NUM_EPOCHS,
    'test_r2_mean': test_r2,
    'test_spear_mean': test_spear,
    'test_r2_per_series': test_r2s,
    'test_spear_per_series': test_spears,
    'baseline_fayam_r2': FAYAM_BASELINE_R2,
    'baseline_fayam_spear': FAYAM_BASELINE_SPEAR,
    'delta_r2_vs_fayam_pp': (test_r2 - FAYAM_BASELINE_R2) * 100,
    'delta_spear_vs_fayam_pp': (test_spear - FAYAM_BASELINE_SPEAR) * 100,
    'hyperparameters': cfg.to_dict(),
}
with open(f'{DRIVE_BASE}/results/test_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Verdict sanity check : on tolère un écart ≤ 10 pp vs FAYAM
SANITY_TOL_PP = 10.0
delta_pp = (test_r2 - FAYAM_BASELINE_R2) * 100

print()
print('=' * 70)
print('  VERDICT SANITY CHECK — v2 avec use_evidence_layer=False')
print('=' * 70)
print(f'  Test R²       : {test_r2:.4f}  ({delta_pp:+.2f} pp vs FAYAM={FAYAM_BASELINE_R2})')
print(f'  Test Spearman : {test_spear:.4f}  ({(test_spear-FAYAM_BASELINE_SPEAR)*100:+.2f} pp vs FAYAM={FAYAM_BASELINE_SPEAR})')
print('=' * 70)
if abs(delta_pp) <= SANITY_TOL_PP:
    print(f'  PASS  →  pipeline saine (écart ≤ {SANITY_TOL_PP} pp). Le bug v1 est dans l\\'evidence layer.')
    print('             Prochaine étape : Run B avec use_evidence_layer=True, mix=0.3 (hybrid).')
else:
    print(f'  FAIL  →  écart > {SANITY_TOL_PP} pp. Bug dans la pipeline avant l\\'evidence layer.')
    print('             Comparer cellule par cellule avec baseline-cluster4.ipynb.')
print('=' * 70)"""
)

md("## 12 — Sauvegarde `run.md`")

code(
    """from datetime import datetime

run_md = f'''# Run {RUN_NAME}

- **Date**          : {datetime.utcnow().isoformat()}Z
- **Mode**          : sanity-check v2 avec `use_evidence_layer=False` → parent FAYAM strict
- **Cluster**       : {CLUSTER_ID}
- **Hardware**      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}
- **Epochs**        : {NUM_EPOCHS} (full, pas d'early stopping)

## Résultats TEST

- **R² mean**       : {test_r2:.4f}  ({delta_pp:+.2f} pp vs FAYAM={FAYAM_BASELINE_R2})
- **Spearman mean** : {test_spear:.4f}  ({(test_spear - FAYAM_BASELINE_SPEAR)*100:+.2f} pp vs FAYAM={FAYAM_BASELINE_SPEAR})
- **Verdict**       : {\"PASS — pipeline saine\" if abs(delta_pp) <= SANITY_TOL_PP else \"FAIL — bug pipeline\"}

## Per-series

| function_id | R²       | Spearman |
|-------------|----------|----------|
'''
for ts_idx in range(len(test_dataset)):
    func_id = test_dataset[ts_idx]['function_id']
    run_md += f'| {func_id} | {test_r2s[ts_idx]:.4f} | {test_spears[ts_idx]:.4f} |\\n'

run_md += f'''

## Conclusion

{\"Pipeline saine — bug v1 isolé dans l'evidence layer. Prochaine étape : Run B avec evidence_mix=0.3.\" if abs(delta_pp) <= SANITY_TOL_PP else \"Bug dans la pipeline. Comparer cellule par cellule avec baseline-cluster4.ipynb.\"}
'''

with open(f'{DRIVE_BASE}/run.md', 'w') as f:
    f.write(run_md)
print(run_md)"""
)

# ──────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook écrit : {NOTEBOOK_PATH}")
print(f"  Cellules : {len(cells)}")
