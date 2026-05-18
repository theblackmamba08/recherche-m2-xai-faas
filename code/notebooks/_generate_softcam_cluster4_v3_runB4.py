"""Génère code/notebooks/softcam-cluster4-v3-runB4.ipynb (Phase 2 — H1 v3 Run B4).

Run B4 : mix=0.10 constant, pas de schedules, modèle v3 (LayerNorm).

Diagnostic Run B3 :
  - LayerNorm corrige le collapse de M (max_weight 0.97 -> 0.06).
  - Mais R² reste -1.59 → la cause n'est pas M elle-même.
  - parameter_projection(0.7*dec + 0.3*h_evidence) est linéaire :
    R² ≈ 0.7*R²(dec) + 0.3*R²(h_evidence).
    h_evidence est structurellement plus pauvre que dec_output (single softmax
    vs cross-attention multi-tête multi-couches) → R²(h_evidence) est très négatif
    → la branche evidence à 30% tire R² vers le négatif.

Hypothèse Run B4 : avec mix=0.10, la branche evidence ne pèse que 10%.
  R² ≈ 0.9*0.53 + 0.1*R²(h_evidence). Même si R²(h_evidence)=-3,
  R² ≈ 0.9*0.53 + 0.1*(-3) = 0.477 - 0.3 = 0.18 → proche du gate 0.30.
  Réaliste : R²(h_evidence) plutôt vers -1, donc R² ≈ 0.477 - 0.1 = 0.38 → PASS.

Choix Run B4 :
  - mix=0.10 constant (pas de warm-up : la perturbation est petite, le décodeur
    peut s'adapter dès l'epoch 0).
  - gamma_entropy=0 (pas de pression de sparsité : à 10%, M peut être diffuse).
  - beta_l2=1e-3 (inchangé).
  - Modèle v3 (on garde le LayerNorm).
  - Pas de schedules.

Lance : python _generate_softcam_cluster4_v3_runB4.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4-v3-runB4.ipynb"

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
    """# SoftCAM-Transformer v3 — Run B4 (mix=0.10, simple, sans schedules)

## Pourquoi cette configuration

### Constat Run B3

| Run | mix | LayerNorm | M max_weight | R² | Spearman |
|-----|-----|-----------|--------------|-----|----------|
| Run A | 0 | — | — | **+0.53** | 0.92 |
| Run B | 0.3 fixe | non | 0.85 | −2.83 | 0.33 |
| Run B2 | warm-up→0.3 | non | 0.97 | −1.97 | 0.80 |
| Run B3 | warm-up→0.3 | **oui** | **0.06** | −1.59 | 0.78 |

Le LayerNorm a rendu M diffuse comme prévu (max_weight passe de 0.97 à 0.06).
Mais R² ne remonte presque pas. La cause n'est donc PAS la distribution de M.

### Analyse structurelle

`parameter_projection` est linéaire, donc :

```
parameter_projection(0.7·dec_output + 0.3·h_evidence)
   = 0.7·parameter_projection(dec_output)    ← R² ≈ 0.53
   + 0.3·parameter_projection(h_evidence)    ← R² très négatif
```

`dec_output` contient : cross-attention multi-tête × 4 couches + FFN + LayerNorm + features temporelles + self-attention causale.
`h_evidence` contient : **une seule softmax** d'agrégation des états encodeurs.

→ `h_evidence` est structurellement plus pauvre que `dec_output`. Le décodeur fait déjà ce que la couche d'évidence essaie de faire, en mieux.
→ À mix=0.3, on force 30% de la prédiction à venir d'une représentation plus faible. R² ne peut pas remonter.

### Hypothèse Run B4

**Si on réduit mix à 0.10**, la branche evidence ne pèse que 10%. La prédiction reste à 90% celle de `dec_output` (proche Run A, R²≈0.53). Estimation conservatrice :

- 0.9 × 0.53 ≈ 0.48 (contribution dec_output)
- 0.1 × (−1)  ≈ −0.1 (contribution h_evidence, estimée pessimistement)
- **R² total ≈ 0.38** → PASS H1.C

M reste extractible et exacte. La fidélité-par-construction tient toujours : M est le coefficient algébrique exact de la branche evidence (qui pèse 10%).

### Choix de configuration

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `evidence_mix` | **0.10 constant** | Pas de warm-up : perturbation petite, l'optimiseur s'adapte dès l'epoch 0 |
| `gamma_entropy` | **0.0** | Pas besoin de sparsité à 10% — M peut être diffuse |
| `beta_l2` | 1e-3 | Inchangé |
| Modèle | **v3 (LayerNorm)** | On garde le fix qui marche |
| Schedules | **aucun** | Diagnostic simple et clair |

## GATE H1.C

R²≥0.30 ET Spearman≥0.85.

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

md("## 2 — Récupération du code")

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

SEED = 998
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
    """# ── Hyperparams (identiques Run A/B/B2/B3) ─────────────────────
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
NUM_EPOCHS        = 51

# ── Evidence Layer v3 (Run B4 — mix petit, pas de schedules) ───
USE_EVIDENCE_LAYER = True
EVIDENCE_MIX       = 0.10           # ← PETIT mix constant
ALPHA_L1           = 0.0
BETA_L2            = 1e-3
GAMMA_ENTROPY      = 0.0            # ← PAS de pression de sparsité

# ── GATE H1.C ────────────────────────────────────────────────
GATE_R2    = 0.30
GATE_SPEAR = 0.85

# ── Références ────────────────────────────────────────────────
RUNA_R2,   RUNA_SPEAR  = 0.5299,  0.9176
RUNB_R2,   RUNB_SPEAR  = -2.8251, 0.3301
RUNB2_R2,  RUNB2_SPEAR = -1.9660, 0.8028
RUNB3_R2,  RUNB3_SPEAR = -1.5894, 0.7771
FAYAM_R2,  FAYAM_SPEAR = 0.3701,  0.9201

CLUSTER_ID = 4
RUN_NAME   = f'softcam-cluster{CLUSTER_ID}-v3-runB4'
print(f'Run : {RUN_NAME}')
print(f'Modèle : SoftCAMTransformerV3ForPrediction (Fix #4 LayerNorm)')
print(f'Configuration simple :')
print(f'  evidence_mix  = {EVIDENCE_MIX}  (constant, pas de schedule)')
print(f'  gamma_entropy = {GAMMA_ENTROPY}  (pas de pression de sparsité)')
print(f'  beta_l2       = {BETA_L2}')
print(f'GATE H1.C : R²≥{GATE_R2}  Spearman≥{GATE_SPEAR}')
print(f'\\nPrédiction du résultat attendu :')
print(f'  R² ≈ 0.9 × {RUNA_R2:.2f} + 0.1 × R²(h_evidence)')
print(f'  Si R²(h_evidence) ≈ -1 → R² ≈ {0.9*RUNA_R2 + 0.1*(-1):.2f}  → PASS')
print(f'  Si R²(h_evidence) ≈ -3 → R² ≈ {0.9*RUNA_R2 + 0.1*(-3):.2f}  → FAIL (très pessimiste)')"""
)

md("## 4 — Google Drive")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'
for subdir in ['checkpoints', 'results', 'logs', 'figures']:
    os.makedirs(f'{DRIVE_BASE}/{subdir}', exist_ok=True)
print(f'Dossier run : {DRIVE_BASE}')"""
)

md("## 5 — Chargement Cluster 4")

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

train_rows, test_rows = [], []
for ts_idx, s in enumerate(all_series):
    base = {'start': START_DATE, 'feat_static_cat': [ts_idx],
            'cluster': s['cluster'], 'function_id': s['function_id']}
    train_rows.append({**base, 'target': s['target_full'][:-PREDICTION_LENGTH].tolist()})
    test_rows.append( {**base, 'target': s['target_full'].tolist()})

train_dataset = Dataset.from_list(train_rows)
test_dataset  = Dataset.from_list(test_rows)
print(f'\\nDatasets : train={len(train_dataset)}  test={len(test_dataset)}')"""
)

md("## 6 — Pipeline GluonTS")

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
time_features  = time_features_from_frequency_str(FREQ)

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch['start'] = [convert_to_pandas_period(d, freq) for d in batch['start']]
    return batch

for ds in (train_dataset, test_dataset):
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
    """## 7 — Construction du modèle v3

mix=0.10 et gamma=0 sont passés directement à la config (pas de schedule).
"""
)

code(
    """from src.models.softcam_transformer_v3 import (
    SoftCAMTransformerV3Config,
    SoftCAMTransformerV3ForPrediction,
)

cfg = SoftCAMTransformerV3Config(
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
    use_evidence_layer=USE_EVIDENCE_LAYER,
    evidence_mix=EVIDENCE_MIX,
    alpha_l1=ALPHA_L1,
    beta_l2=BETA_L2,
    gamma_entropy=GAMMA_ENTROPY,
)

model = SoftCAMTransformerV3ForPrediction(cfg).to(device)

n_params   = sum(p.numel() for p in model.parameters())
n_evidence = sum(p.numel() for p in model.evidence_linear.parameters())
n_norm     = sum(p.numel() for p in model.evidence_norm.parameters())
n_parent   = n_params - n_evidence - n_norm
print(f'Paramètres totaux       : {n_params:,}')
print(f'  dont parent FAYAM     : {n_parent:,}')
print(f'  dont evidence_linear  : {n_evidence:,}')
print(f'  dont evidence_norm    : {n_norm:,}  ← LayerNorm Fix #4')
print(f'\\nmix={model.evidence_mix:.3f}  γ={model.gamma_entropy:.0e}  β_L2={model.beta_l2:.0e}')"""
)

md("## 8 — Helper d'évaluation")

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
        print(f'{prefix}  per-series R²       : {[f"{x:.3f}" for x in r2s]}')
        print(f'{prefix}  per-series Spearman : {[f"{x:.3f}" for x in spears]}')
    return r2, spear, r2s, spears

print('Helper évaluation prêt.')"""
)

md(
    """## 9 — Entraînement 51 epochs (mix=0.10 constant, pas de schedule)

Boucle simple comme Run A. Seuls `evidence_mix=0.10` et `gamma_entropy=0` changent.
On monitore les 4 composantes de loss comme d'habitude.
"""
)

code(
    """from torch.optim import AdamW

# Re-seed RNG avant training (fix5 de Run A).
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_loader = create_train_dataloader(cfg, FREQ, train_dataset, BATCH_SIZE_TRAIN, NUM_BATCHES_EPOCH)
optimizer    = AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

history = {'train_loss': [], 'forecast_loss': [], 'elastic_loss': [], 'entropy_loss': []}
ckpt_path = f'{DRIVE_BASE}/checkpoints/v3_runB4_final.pt'

for epoch in range(NUM_EPOCHS):
    model.train()
    losses_total, losses_fc, losses_el, losses_en = [], [], [], []
    pbar = tqdm(train_loader, desc=f'Ep {epoch:03d}/{NUM_EPOCHS-1}', leave=False)
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

        losses_total.append(out.loss.item())
        if out.forecast_loss is not None: losses_fc.append(out.forecast_loss.item())
        if out.elastic_loss  is not None: losses_el.append(out.elastic_loss.item())
        if out.entropy_loss  is not None: losses_en.append(out.entropy_loss.item())
        pbar.set_postfix({'loss': f'{out.loss.item():.3f}'})

    history['train_loss'].append(float(np.mean(losses_total)))
    history['forecast_loss'].append(float(np.mean(losses_fc)) if losses_fc else 0.0)
    history['elastic_loss'].append(float(np.mean(losses_el)) if losses_el else 0.0)
    history['entropy_loss'].append(float(np.mean(losses_en)) if losses_en else 0.0)

    print(f'Ep {epoch:03d}  loss={history["train_loss"][-1]:.4f}  '
          f'fc={history["forecast_loss"][-1]:.4f}  '
          f'el={history["elastic_loss"][-1]:.4f}  '
          f'H(M)={history["entropy_loss"][-1]:.4f}')

torch.save({
    'epoch': NUM_EPOCHS - 1, 'model': model.state_dict(),
    'history': history, 'hyperparameters': cfg.to_dict(),
}, ckpt_path)
print(f'\\nEntraînement terminé — {NUM_EPOCHS} epochs')"""
)

md("## 10 — Courbes de loss")

code(
    """fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ep = list(range(len(history['train_loss'])))

pairs = [
    (axes[0, 0], 'train_loss',    'steelblue',  'Loss totale'),
    (axes[0, 1], 'forecast_loss', 'darkorange', 'Forecast loss (NLL)'),
    (axes[1, 0], 'elastic_loss',  'firebrick',  'Elastic loss (β·‖M‖²)'),
    (axes[1, 1], 'entropy_loss',  'seagreen',   'γ·H(M)  (γ=0 → constant à 0)'),
]
for ax, key, color, title in pairs:
    ax.plot(ep, history[key], lw=1.5, color=color)
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)

plt.suptitle(f'Run B4 — mix={EVIDENCE_MIX} constant, γ=0', fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/training_curves.png', dpi=150)
plt.show()"""
)

md("## 11 — Évaluation finale + GATE H1.C")

code(
    """test_loader = create_backtest_dataloader(cfg, FREQ, test_dataset, BATCH_SIZE_TEST)
test_r2, test_spear, test_r2s, test_spears = evaluate(
    model, test_loader, test_dataset, cfg, device, prefix='TEST'
)

results = {
    'run': RUN_NAME,
    'mode': 'evidence-layer-on-small-mix',
    'model_version': 'v3',
    'fix': 'small constant mix (0.10), no schedules',
    'use_evidence_layer': USE_EVIDENCE_LAYER,
    'evidence_mix': EVIDENCE_MIX,
    'gamma_entropy': GAMMA_ENTROPY,
    'beta_l2': BETA_L2,
    'num_epochs': NUM_EPOCHS,
    'test_r2_mean': test_r2,
    'test_spear_mean': test_spear,
    'test_r2_per_series': test_r2s,
    'test_spear_per_series': test_spears,
    'gate_r2': GATE_R2,
    'gate_spear': GATE_SPEAR,
    'runa_r2': RUNA_R2,   'runa_spear': RUNA_SPEAR,
    'runb_r2': RUNB_R2,   'runb_spear': RUNB_SPEAR,
    'runb2_r2': RUNB2_R2, 'runb2_spear': RUNB2_SPEAR,
    'runb3_r2': RUNB3_R2, 'runb3_spear': RUNB3_SPEAR,
    'fayam_r2': FAYAM_R2, 'fayam_spear': FAYAM_SPEAR,
    'delta_r2_vs_runa_pp':  (test_r2  - RUNA_R2)  * 100,
    'delta_r2_vs_runb3_pp': (test_r2  - RUNB3_R2) * 100,
    'delta_spear_vs_runa_pp':  (test_spear  - RUNA_SPEAR)  * 100,
    'delta_spear_vs_runb3_pp': (test_spear  - RUNB3_SPEAR) * 100,
    'hyperparameters': cfg.to_dict(),
}
with open(f'{DRIVE_BASE}/results/test_metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

gate_r2_ok    = test_r2    >= GATE_R2
gate_spear_ok = test_spear >= GATE_SPEAR

print()
print('=' * 72)
print('  GATE H1.C — SoftCAM v3 Run B4 (mix=0.10 constant, γ=0)')
print('=' * 72)
print(f'  Test R²       : {test_r2:.4f}  (gate≥{GATE_R2})   {"PASS" if gate_r2_ok    else "FAIL"}')
print(f'  Test Spearman : {test_spear:.4f}  (gate≥{GATE_SPEAR})   {"PASS" if gate_spear_ok else "FAIL"}')
print(f'  Δ R²   vs Run A  : {(test_r2-RUNA_R2)*100:+.2f} pp   (Run A={RUNA_R2})')
print(f'  Δ R²   vs Run B3 : {(test_r2-RUNB3_R2)*100:+.2f} pp   (Run B3={RUNB3_R2})')
print(f'  Δ Spear vs Run A : {(test_spear-RUNA_SPEAR)*100:+.2f} pp')
print('=' * 72)
if gate_r2_ok and gate_spear_ok:
    print('  PASS H1.C  →  H1 défendable : mix=0.10 préserve la précision')
    print('                 tout en fournissant une carte M algébriquement exacte.')
    print('                 Prochaine étape : analyser les cartes M (H1.A, H1.D).')
else:
    failed = []
    if not gate_r2_ok:    failed.append(f'R²={test_r2:.4f} < {GATE_R2}')
    if not gate_spear_ok: failed.append(f'Spearman={test_spear:.4f} < {GATE_SPEAR}')
    print(f'  FAIL H1.C  →  {" | ".join(failed)}.')
    print('                 Plan B5 : essayer mix=0.05.')
    print('                 Plan B6 : changement architectural (M = last cross-attention).')
print('=' * 72)"""
)

md("## 12 — Extraction des evidence maps M")

code(
    """test_loader_expl = create_backtest_dataloader(cfg, FREQ, test_dataset, batch_size=1)
evidence_maps = []

model.eval()
with torch.no_grad():
    for ts_idx, b in enumerate(test_loader_expl):
        func_id = test_dataset[ts_idx]['function_id']
        target_full = np.array(test_dataset[ts_idx]['target'])
        future_vals = torch.tensor(
            target_full[-PREDICTION_LENGTH:], dtype=torch.float32
        ).unsqueeze(0).to(device)
        future_obs = torch.ones_like(future_vals)

        M = model.explain(
            past_values=b['past_values'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_values=future_vals,
            future_time_features=b['future_time_features'].to(device),
            static_categorical_features=b['static_categorical_features'].to(device)
                if cfg.num_static_categorical_features > 0 else None,
            future_observed_mask=future_obs,
        )
        M_np = M.squeeze(0).cpu().numpy()
        evidence_maps.append((func_id, M_np))
        print(f'  func_id={func_id}  M.shape={M_np.shape}  '
              f'argmax_mean={int(M_np.argmax(axis=1).mean())}  '
              f'max_weight={M_np.max():.4f}  '
              f'row_entropy_mean={-(M_np * np.log(M_np.clip(1e-9))).sum(axis=1).mean():.3f}')

print(f'\\n{len(evidence_maps)} cartes M extraites.')"""
)

md("## 13 — Heatmaps + H1.A + H1.D")

code(
    """fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

M_mean = np.mean([m for _, m in evidence_maps], axis=0)
ax = axes[0]
im = ax.imshow(M_mean, aspect='auto', origin='upper', cmap='YlOrRd', vmin=0, vmax=M_mean.max())
ax.set_title('Moyenne cluster (5 fonctions)', fontweight='bold')
ax.set_xlabel('Contexte encodeur s ∈ [0, 239]')
ax.set_ylabel('Horizon de prédiction t ∈ [0, 119]')
plt.colorbar(im, ax=ax)
argmax_mean = M_mean.argmax(axis=1)
ax.plot(argmax_mean, np.arange(PREDICTION_LENGTH), 'b--', lw=1.2, label='argmax')
ax.legend(loc='upper right', fontsize=8)

for i, (func_id, M_np) in enumerate(evidence_maps):
    ax = axes[i + 1]
    im = ax.imshow(M_np, aspect='auto', origin='upper', cmap='YlOrRd', vmin=0, vmax=M_mean.max())
    ax.set_title(f'Function {func_id}')
    ax.set_xlabel('s'); ax.set_ylabel('t')
    plt.colorbar(im, ax=ax)
    ax.plot(M_np.argmax(axis=1), np.arange(PREDICTION_LENGTH), 'b--', lw=1.0)

plt.suptitle(
    f'Evidence Maps M — Run B4 (mix={EVIDENCE_MIX}, γ=0)\\n'
    f'R²={test_r2:.4f}  Spearman={test_spear:.4f}',
    fontsize=12
)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/evidence_maps_heatmap.png', dpi=150)
plt.show()"""
)

code(
    """# H1.A : distribution + évolution des argmax
all_argmax = np.concatenate([m.argmax(axis=1) for _, m in evidence_maps])
argmax_matrix = np.stack([m.argmax(axis=1) for _, m in evidence_maps])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(all_argmax, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(all_argmax.mean(), color='crimson', lw=1.5, linestyle='--',
                label=f'mean={all_argmax.mean():.1f}')
axes[0].set_title('Distribution des argmax M[t,:]')
axes[0].set_xlabel('Indice encodeur s'); axes[0].set_ylabel('Fréquence')
axes[0].legend(); axes[0].grid(alpha=0.3)

t = np.arange(PREDICTION_LENGTH)
argmax_m = argmax_matrix.mean(axis=0)
argmax_s = argmax_matrix.std(axis=0)
axes[1].plot(t, argmax_m, color='steelblue', lw=1.5, label='mean argmax')
axes[1].fill_between(t, argmax_m - argmax_s, argmax_m + argmax_s, alpha=0.25, color='steelblue')
axes[1].set_title('Argmax M[t,:] par pas de prédiction')
axes[1].set_xlabel('Horizon t'); axes[1].set_ylabel('Indice encodeur s')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/argmax_profile.png', dpi=150)
plt.show()

print(f'Argmax mean global : {all_argmax.mean():.1f} / 239')"""
)

code(
    """# H1.D
from scipy.spatial.distance import cosine

M_flat   = [m.flatten() for _, m in evidence_maps]
func_ids = [fid for fid, _ in evidence_maps]
n = len(M_flat)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = 1.0 - cosine(M_flat[i], M_flat[j])

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(sim_matrix, cmap='Greens', vmin=0, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(func_ids)
ax.set_yticks(range(n)); ax.set_yticklabels(func_ids)
ax.set_title('H1.D — Cosine similarity des evidence maps')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/coherence_matrix.png', dpi=150)
plt.show()

off_diag = sim_matrix[np.triu_indices(n, k=1)]
print(f'H1.D cosine off-diagonal : mean={off_diag.mean():.3f}  min={off_diag.min():.3f}')"""
)

md("## 14 — Sauvegarde `run.md`")

code(
    """from datetime import datetime, timezone

gate_ok = gate_r2_ok and gate_spear_ok
off_diag = np.array([
    sim_matrix[i, j]
    for i in range(len(evidence_maps)) for j in range(i+1, len(evidence_maps))
])

run_md = f\'\'\'# Run {RUN_NAME}

- **Date**     : {datetime.now(timezone.utc).isoformat()}
- **Modèle**   : SoftCAMTransformerV3ForPrediction (LayerNorm Fix #4)
- **Cluster**  : {CLUSTER_ID}
- **Hardware** : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
- **Epochs**   : {NUM_EPOCHS}

## Configuration (simple, sans schedule)

- evidence_mix   = {EVIDENCE_MIX}  (constant)
- gamma_entropy  = {GAMMA_ENTROPY}
- beta_l2        = {BETA_L2}

## GATE H1.C

| Métrique | Valeur | Seuil | Statut |
|----------|--------|-------|--------|
| Test R² | {test_r2:.4f} | ≥{GATE_R2} | {"PASS" if gate_r2_ok else "FAIL"} |
| Test Spearman | {test_spear:.4f} | ≥{GATE_SPEAR} | {"PASS" if gate_spear_ok else "FAIL"} |

**Verdict H1.C : {"PASS" if gate_ok else "FAIL"}**

## Comparaison

| Run | mix | R² | Spearman |
|-----|-----|-----|---------|
| FAYAM C4 | — | {FAYAM_R2} | {FAYAM_SPEAR} |
| Run A | 0 | {RUNA_R2} | {RUNA_SPEAR} |
| Run B | 0.3 const | {RUNB_R2} | {RUNB_SPEAR} |
| Run B2 | warm-up→0.3 | {RUNB2_R2} | {RUNB2_SPEAR} |
| Run B3 (LayerNorm) | warm-up→0.3 | {RUNB3_R2} | {RUNB3_SPEAR} |
| **Run B4 (LayerNorm, simple)** | **{EVIDENCE_MIX}** | **{test_r2:.4f}** | **{test_spear:.4f}** |

## Per-series

| function_id | R² | Spearman |
|-------------|-----|---------|
\'\'\'
for ts_idx in range(len(test_dataset)):
    func_id = test_dataset[ts_idx]['function_id']
    run_md += f'| {func_id} | {test_r2s[ts_idx]:.4f} | {test_spears[ts_idx]:.4f} |\\n'

run_md += f\'\'\'
## H1.A

- Argmax mean global : {all_argmax.mean():.1f} / 239

## H1.D

- Cosine off-diagonal : mean={off_diag.mean():.3f}  min={off_diag.min():.3f}
- Verdict : {"PASS (≥0.7)" if off_diag.mean() >= 0.7 else "ATTENTION (<0.7)"}

## Fichiers

- `figures/training_curves.png`
- `figures/evidence_maps_heatmap.png`
- `figures/argmax_profile.png`
- `figures/coherence_matrix.png`
- `results/test_metrics.json`
- `checkpoints/v3_runB4_final.pt`
\'\'\'

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
