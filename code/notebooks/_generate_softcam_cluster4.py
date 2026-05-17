"""Génère code/notebooks/softcam-cluster4.ipynb (Phase 2 — H1 first run).

Lance : python _generate_softcam_cluster4.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4.ipynb"

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
    """# SoftCAM-Transformer Cluster 4 — Phase 2 H1 (first run)

Première run de **H1 SoftCAM-Transformer** sur le Cluster 4.

| Champ | Valeur |
|-------|--------|
| **Phase** | 2 — H1 SoftCAM-Transformer |
| **Run** | `softcam-cluster4-h1-v1` |
| **Backbone** | `TimeSeriesTransformerForPrediction` (FAYAM, inchangé) |
| **Nouveauté** | `evidence_linear : Linear(d_model → context_length) + Softmax + bmm` |
| **Baseline à battre** | FAYAM Cluster 4 — R²=0.37, Spearman=0.92 |
| **Gate H1.C (non-dégradation)** | R²≥0.30 et Spearman≥0.85 (sinon → pivot vers H2) |

> Avant de lancer : Runtime → Change runtime type → **T4 GPU**
> Et n'oublie pas que **le critère go/no-go est écrit dans ce notebook** — pas négociable après coup.
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
    """## 2 — Récupération du code (modèle H1 défini dans `code/src/models/`)

On clone le dépôt pour disposer de `src/models/softcam_transformer.py`. Si tu n'as pas encore push tes changements, exécute la cellule suivante **après avoir fait** :

```bash
git add code/src/models/softcam_transformer.py code/src/models/__init__.py
git commit -m "feat(models): SoftCAM-Transformer H1 — Evidence Layer + ElasticNet/entropy"
git push
```
"""
)

code(
    """import os, sys

REPO_URL = 'https://github.com/theblackmamba08/recherche-m2-xai-faas.git'
REPO_DIR = '/content/recherche-m2-xai-faas'

# get_ipython().system() = equivalent du ! Colab — accès complet au shell,
# pas de problème de TTY/credentials comme avec subprocess
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

SEED = 2026
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
    """# ── Hyperparams reconduits de FAYAM (Cluster 4 baseline) ────
FREQ              = '1T'
PREDICTION_LENGTH = 120
CONTEXT_LENGTH    = 240
D_MODEL           = 32
N_HEADS           = 2
ENCODER_LAYERS    = 2
DECODER_LAYERS    = 2
EMBEDDING_DIM     = [2]
DROPOUT           = 0.1
BATCH_SIZE_TRAIN  = 256
BATCH_SIZE_TEST   = 64
NUM_BATCHES_EPOCH = 100
LR                = 6e-4
BETAS             = (0.9, 0.95)
WEIGHT_DECAY      = 1e-1
GRAD_CLIP_NORM    = 1.0
MAX_EPOCHS        = 60
PATIENCE          = 10

# ── Hyperparams spécifiques H1 (Evidence Layer regularization) ──
ALPHA_L1      = 0.0      # constant under softmax (gradient = 0) — kept for parity
BETA_L2       = 1e-3     # mean(M²) — promotes smoothness (uniform rows)
GAMMA_ENTROPY = 1e-3     # row entropy — promotes sparsity (peaked rows)

# ── Critère go/no-go H1.C ───────────────────────────────────
GATE_R2_MIN          = 0.30
GATE_SPEARMAN_MIN    = 0.85
FAYAM_BASELINE_R2    = 0.3701
FAYAM_BASELINE_SPEAR = 0.9201

CLUSTER_ID = 4
RUN_NAME   = f'softcam-cluster{CLUSTER_ID}-h1-v1'
print(f'Run : {RUN_NAME}')
print(f'Gate : R² ≥ {GATE_R2_MIN}, Spearman ≥ {GATE_SPEARMAN_MIN}')
print(f'Baseline FAYAM : R²={FAYAM_BASELINE_R2}, Spearman={FAYAM_BASELINE_SPEAR}')"""
)

md("## 4 — Google Drive (checkpoints + résultats)")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'
for subdir in ['checkpoints', 'results', 'evidence', 'logs']:
    os.makedirs(f'{DRIVE_BASE}/{subdir}', exist_ok=True)
print(f'Dossier run : {DRIVE_BASE}')"""
)

md(
    """## 5 — Chargement Cluster 4

Datasets construits comme pour FAYAM (train = `target[:-120]`, test = `target[:]`)."""
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

def create_evidence_dataloader(config, freq, data, batch_size):
    \"\"\"Like backtest but exposes future_values for teacher-forced explanation.\"\"\"
    fields = _pred_fields(config) + ['future_values', 'future_observed_mask']
    tr = create_transformation(freq, config)
    sp = create_instance_splitter(config, 'validation')
    return as_stacked_batches(sp.apply(tr.apply(data), is_train=True),
                              batch_size=batch_size, output_type=torch.tensor,
                              field_names=fields)

print('Pipeline GluonTS prêt.')"""
)

md("## 7 — Construction du modèle H1 (SoftCAM-Transformer)")

code(
    """from src.models.softcam_transformer import (
    SoftCAMTransformerConfig,
    SoftCAMTransformerForPrediction,
)

cfg = SoftCAMTransformerConfig(
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
    # H1 regularization on M
    alpha_l1=ALPHA_L1,
    beta_l2=BETA_L2,
    gamma_entropy=GAMMA_ENTROPY,
)

model = SoftCAMTransformerForPrediction(cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_params_evidence = sum(p.numel() for p in model.evidence_linear.parameters())
print(f'Paramètres totaux       : {n_params:,}')
print(f'  dont evidence_linear  : {n_params_evidence:,}  ({100*n_params_evidence/n_params:.1f}%)')
print(f'\\nEvidence Layer : Linear({D_MODEL} → {CONTEXT_LENGTH})  → Softmax(dim=-1)')
print(f'Régularisation : α‖M‖₁={ALPHA_L1}  β‖M‖₂²={BETA_L2}  γ·H(M)={GAMMA_ENTROPY}')"""
)

md("## 8 — Helper d'évaluation (R² + Spearman)")

code(
    """from sklearn.metrics import r2_score
from scipy.stats import spearmanr

@torch.no_grad()
def evaluate(model, dataloader, dataset, config, device, prefix=''):
    \"\"\"Mean R² and mean Spearman across all series of the dataset.\"\"\"
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
    """## 9 — Entraînement avec early stopping

Particularité H1 : on **trace séparément** les 3 composantes de la loss pour vérifier que :
- `forecast_loss` baisse (apprentissage)
- `entropy_loss` baisse (M devient peaky, donc explicatif)
- `elastic_loss` reste contrôlée
"""
)

code(
    """from torch.optim import AdamW

train_loader = create_train_dataloader(cfg, FREQ, train_dataset, BATCH_SIZE_TRAIN, NUM_BATCHES_EPOCH)
val_loader   = create_backtest_dataloader(cfg, FREQ, val_dataset, BATCH_SIZE_TEST)

optimizer = AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

history = {'total': [], 'forecast': [], 'elastic': [], 'entropy': [], 'val_r2': [], 'val_spear': []}
best_val_r2 = -float('inf'); patience_ctr = 0; best_state = None
ckpt_path = f'{DRIVE_BASE}/checkpoints/softcam_h1_best.pt'

for epoch in range(MAX_EPOCHS):
    model.train()
    losses_t, losses_f, losses_el, losses_en = [], [], [], []
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}', leave=False)
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
        losses_t.append(out.loss.item())
        losses_f.append(out.forecast_loss.item())
        losses_el.append(out.elastic_loss.item())
        losses_en.append(out.entropy_loss.item())
        pbar.set_postfix({
            'tot':  f'{out.loss.item():.3f}',
            'fcst': f'{out.forecast_loss.item():.3f}',
            'ent':  f'{out.entropy_loss.item():.4f}',
        })

    mean_t = float(np.mean(losses_t))
    mean_f = float(np.mean(losses_f))
    mean_el = float(np.mean(losses_el))
    mean_en = float(np.mean(losses_en))
    val_r2, val_spear, _, _ = evaluate(model, val_loader, val_dataset, cfg, device)

    history['total'].append(mean_t)
    history['forecast'].append(mean_f)
    history['elastic'].append(mean_el)
    history['entropy'].append(mean_en)
    history['val_r2'].append(val_r2)
    history['val_spear'].append(val_spear)

    line = (f'Epoch {epoch:03d}  tot={mean_t:.4f}  fcst={mean_f:.4f}  '
            f'el={mean_el:.4f}  ent={mean_en:.4f}  val_r2={val_r2:.4f}  val_spear={val_spear:.4f}')
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2; patience_ctr = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({'epoch': epoch, 'model': best_state, 'val_r2': val_r2,
                    'val_spear': val_spear, 'history': history,
                    'hyperparameters': cfg.to_dict()}, ckpt_path)
        print(line + '  ← best (saved)')
    else:
        patience_ctr += 1
        print(line + f'  (patience {patience_ctr}/{PATIENCE})')
        if patience_ctr >= PATIENCE:
            print(f'\\nEarly stopping @ epoch {epoch} (best val_r2={best_val_r2:.4f})')
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f'\\nEntraînement terminé — best val R² = {best_val_r2:.4f}')"""
)

md("## 10 — Courbes (les 3 composantes de loss + val R²)")

code(
    """fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
ep = list(range(len(history['total'])))

axes[0].plot(ep, history['total'],    lw=1.5, color='black', label='Total')
axes[0].plot(ep, history['forecast'], lw=1.5, color='steelblue', label='Forecast NLL')
axes[0].set_xlabel('Époque'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss train (totale vs forecast)')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, history['elastic'], lw=1.5, color='orange', label='β‖M‖₂² + α‖M‖₁')
axes[1].plot(ep, history['entropy'], lw=1.5, color='red',    label='γ·H(M)')
axes[1].set_xlabel('Époque'); axes[1].set_ylabel('Loss')
axes[1].set_title('Régularisations sur M')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep, history['val_r2'], lw=1.5, color='tomato', label='SoftCAM-H1 val R²')
axes[2].axhline(GATE_R2_MIN, color='red',  lw=1, ls='--', label=f'Gate ({GATE_R2_MIN})')
axes[2].axhline(FAYAM_BASELINE_R2, color='gray', lw=1, ls='--', label=f'FAYAM ({FAYAM_BASELINE_R2})')
axes[2].set_xlabel('Époque'); axes[2].set_ylabel('R²')
axes[2].set_title('Validation R²')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/training_curves.png', dpi=150)
plt.show()"""
)

md(
    """## 11 — Évaluation finale sur TEST + GATE H1.C

C'est ici qu'on décide GO / NO-GO.
"""
)

code(
    """test_loader = create_backtest_dataloader(cfg, FREQ, test_dataset, BATCH_SIZE_TEST)
test_r2, test_spear, test_r2s, test_spears = evaluate(
    model, test_loader, test_dataset, cfg, device, prefix='TEST'
)

# Sauvegarde JSON pour le suivi
results = {
    'run': RUN_NAME,
    'test_r2_mean': test_r2,
    'test_spear_mean': test_spear,
    'test_r2_per_series': test_r2s,
    'test_spear_per_series': test_spears,
    'baseline_fayam_r2': FAYAM_BASELINE_R2,
    'baseline_fayam_spear': FAYAM_BASELINE_SPEAR,
    'delta_r2_vs_fayam_pp': (test_r2 - FAYAM_BASELINE_R2) * 100,
    'delta_spear_vs_fayam_pp': (test_spear - FAYAM_BASELINE_SPEAR) * 100,
    'gate_r2_min': GATE_R2_MIN,
    'gate_spear_min': GATE_SPEARMAN_MIN,
    'hyperparameters': cfg.to_dict(),
}
with open(f'{DRIVE_BASE}/results/test_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print()
print('=' * 64)
print(f'  GATE H1.C — non-dégradation vs FAYAM')
print('=' * 64)
print(f'  Test R²       : {test_r2:.4f}  ({(test_r2-FAYAM_BASELINE_R2)*100:+.2f} pp vs FAYAM)')
print(f'  Test Spearman : {test_spear:.4f}  ({(test_spear-FAYAM_BASELINE_SPEAR)*100:+.2f} pp vs FAYAM)')
print(f'  Gate R²       : ≥ {GATE_R2_MIN}        →  {\"PASS\" if test_r2 >= GATE_R2_MIN else \"FAIL\"}')
print(f'  Gate Spearman : ≥ {GATE_SPEARMAN_MIN}  →  {\"PASS\" if test_spear >= GATE_SPEARMAN_MIN else \"FAIL\"}')
print('=' * 64)
if test_r2 >= GATE_R2_MIN and test_spear >= GATE_SPEARMAN_MIN:
    print('  GO  →  continuer H1 (extraction de M, vérification de l\\'explication)')
else:
    print('  NO-GO  →  pivoter vers H2 (TimeSHAP) sans attendre')
print('=' * 64)"""
)

md(
    """## 12 — Extraction de la carte d'évidence M (l'explication)

Pour chaque série du Cluster 4, on extrait `M (1, 120, 240)` via un forward teacher-forcé sur le test set, puis on visualise.
"""
)

code(
    """ev_loader = create_evidence_dataloader(cfg, FREQ, test_dataset, batch_size=1)

evidence_per_series = {}
model.eval()
with torch.no_grad():
    for ts_idx, b in enumerate(ev_loader):
        M = model.explain(
            past_values=b['past_values'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_values=b['future_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            static_categorical_features=b['static_categorical_features'].to(device)
                if cfg.num_static_categorical_features > 0 else None,
            future_observed_mask=b['future_observed_mask'].to(device),
        )  # (1, 120, 240)
        func_id = test_dataset[ts_idx]['function_id']
        evidence_per_series[func_id] = M[0].cpu().numpy()
        # Save
        np.save(f'{DRIVE_BASE}/evidence/M_function_{func_id}.npy', evidence_per_series[func_id])
        print(f'function_id={func_id}  M.shape={M[0].shape}  '
              f'min={M[0].min().item():.3e}  max={M[0].max().item():.3e}  '
              f'argmax_mean={M[0].argmax(dim=-1).float().mean().item():.1f}')

print(f'\\n{len(evidence_per_series)} cartes d\\'évidence sauvegardées dans {DRIVE_BASE}/evidence/')"""
)

md(
    """## 13 — Visualisation des cartes d'évidence

Une heatmap par fonction du Cluster 4. Axe X = positions du contexte passé (0 = plus ancien, 239 = juste avant la prédiction). Axe Y = pas futur (0 = immédiat, 119 = +120 min).
"""
)

code(
    """n_series = len(evidence_per_series)
fig, axes = plt.subplots(1, n_series, figsize=(5 * n_series, 5), squeeze=False)
for i, (func_id, M) in enumerate(sorted(evidence_per_series.items())):
    ax = axes[0, i]
    im = ax.imshow(M, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title(f'Fonction {func_id}')
    ax.set_xlabel('Position contexte passé (min)')
    ax.set_ylabel('Pas futur (min)')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/evidence_maps.png', dpi=150)
plt.show()"""
)

code(
    """# Ligne argmax par pas futur — quelle minute du passé est la plus "regardée" ?
fig, axes = plt.subplots(1, n_series, figsize=(5 * n_series, 4), squeeze=False)
for i, (func_id, M) in enumerate(sorted(evidence_per_series.items())):
    ax = axes[0, i]
    argmax_series = M.argmax(axis=-1)  # (120,)
    ax.plot(argmax_series, lw=1.5, color='steelblue')
    ax.set_title(f'Fonction {func_id} — argmax_s M[t,s] par pas futur t')
    ax.set_xlabel('Pas futur t (min)')
    ax.set_ylabel('Position contexte la plus regardée')
    ax.set_ylim(0, CONTEXT_LENGTH)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/argmax_by_step.png', dpi=150)
plt.show()"""
)

md(
    """## 14 — Sauvegarde finale du `run.md`

Permet de retrouver les conditions exactes de la run pour la rédaction.
"""
)

code(
    """from datetime import datetime

run_md = f'''# Run {RUN_NAME}

- **Date**          : {datetime.utcnow().isoformat()}Z
- **Phase**         : 2 — H1 SoftCAM-Transformer (first run)
- **Cluster**       : {CLUSTER_ID}
- **Hardware**      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}

## Hyperparams

- `d_model`         : {D_MODEL}
- `context_length`  : {CONTEXT_LENGTH}
- `prediction_len`  : {PREDICTION_LENGTH}
- `encoder_layers`  : {ENCODER_LAYERS}
- `decoder_layers`  : {DECODER_LAYERS}
- `n_heads`         : {N_HEADS}
- `lr`              : {LR}
- `batch_size`      : {BATCH_SIZE_TRAIN}

## Evidence Layer

- `alpha_l1`        : {ALPHA_L1}     (inert under softmax — kept for parity)
- `beta_l2`         : {BETA_L2}
- `gamma_entropy`   : {GAMMA_ENTROPY}

## Résultats TEST

- **R² mean**       : {test_r2:.4f}  ({(test_r2 - FAYAM_BASELINE_R2)*100:+.2f} pp vs FAYAM={FAYAM_BASELINE_R2})
- **Spearman mean** : {test_spear:.4f}  ({(test_spear - FAYAM_BASELINE_SPEAR)*100:+.2f} pp vs FAYAM={FAYAM_BASELINE_SPEAR})
- **Gate H1.C R²**       : {\"PASS\" if test_r2 >= GATE_R2_MIN else \"FAIL\"} (seuil {GATE_R2_MIN})
- **Gate H1.C Spearman** : {\"PASS\" if test_spear >= GATE_SPEARMAN_MIN else \"FAIL\"} (seuil {GATE_SPEARMAN_MIN})

## Per-series

| function_id | R²       | Spearman |
|-------------|----------|----------|
'''
for ts_idx in range(len(test_dataset)):
    func_id = test_dataset[ts_idx]['function_id']
    run_md += f'| {func_id} | {test_r2s[ts_idx]:.4f} | {test_spears[ts_idx]:.4f} |\\n'

run_md += f'''

## Artefacts

- Checkpoint : `{DRIVE_BASE}/checkpoints/softcam_h1_best.pt`
- Métriques  : `{DRIVE_BASE}/results/test_metrics.json`
- Heatmaps M : `{DRIVE_BASE}/results/evidence_maps.png`
- argmax M   : `{DRIVE_BASE}/results/argmax_by_step.png`
- Cartes M   : `{DRIVE_BASE}/evidence/M_function_*.npy`
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
