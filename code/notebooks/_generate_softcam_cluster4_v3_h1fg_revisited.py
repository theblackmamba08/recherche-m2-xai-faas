"""Génère code/notebooks/softcam-cluster4-v3-h1fg-revisited.ipynb.

Reprise des hypothèses H1.F (comprehensiveness) et H1.G (sufficiency) sur
le checkpoint B5, à différentes valeurs de evidence_mix à l'inférence.

Pourquoi ce notebook existe :

À l'origine, H1.F et H1.G ont été testées avec ``model.evidence_mix=0.05``
(la valeur d'entraînement de B5). À ce régime, h_evidence ne pèse que 5%
de la prédiction → même une M complètement détruite ne peut bouger la
prédiction que de ~5% par construction (*ceiling effect*). Verdict initial : ⚠️.

Avec Fix #5 actif et la découverte que mix=0.25 est l'optimum d'inférence
(R²=0.7563), on peut maintenant refaire H1.F/G dans un régime où le test
est mécaniquement informatif (plafond théorique ~25%, pas ~5%).

``predict_with_M_override`` passe par ``forward()`` teacher-forced, donc le
Fix #5 (qui ne touche que ``generate()``) ne change rien pour ce test
précis. Ce qui change est qu'on configure ``model.evidence_mix`` à des
valeurs différentes avant l'override de M.

Design du test :

  Grid mix    : [0.05, 0.10, 0.25]
  Grid k      : [1, 2, 5, 10, 20, 50, 100]
  Functions   : 5 (949, 951, 952, 953, 954)
  → 105 évaluations par hypothèse (H1.F + H1.G = 210 forward)

Coût estimé : ~5 min Colab T4 (forward teacher-forced, pas de génération).

Sortie :
  - Courbes MAE(k) par mix pour H1.F et H1.G
  - JSON avec tous les chiffres
  - Verdict refait pour les deux hypothèses au régime mix=0.25
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4-v3-h1fg-revisited.ipynb"

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
    """# SoftCAM v3 — H1.F / H1.G revisités (mix=0.05, 0.10, 0.25)

## Pourquoi ce notebook

H1.F (comprehensiveness) et H1.G (sufficiency) testent si M est
**causalement utilisée** par le modèle :

- H1.F : masquer les top-k entrées de M par ligne → la prédiction doit se dégrader
- H1.G : ne garder que les top-k entrées par ligne → la prédiction doit être préservée

Première version (verdict ⚠️) : testée à `model.evidence_mix=0.05`, donc
h_evidence ne contribue que 5% à la prédiction. Plafond mécanique → test
inconclant.

**Nouvelle version** : on rejoue le k-sweep à **3 valeurs de mix** pour
quantifier le ceiling effect ET produire un verdict propre au régime
opératoire (mix=0.25).

| mix | Contribution théorique max de h_evidence | Régime |
|-----|------------------------------------------|--------|
| 0.05 | 5% | mix d'entraînement (référence historique) |
| 0.10 | 10% | intermédiaire |
| 0.25 | 25% | optimum d'inférence (configuration finale) |

## Méthodologie

- `predict_with_M_override` passe par `forward()` teacher-forced (HuggingFace
  appelle `output_params` quand `future_values is not None`). **Fix #5
  n'intervient pas ici** (il ne concerne que `generate()`).
- Pour chaque (mix, k, fonction), on calcule MAE entre la moyenne de la
  distribution prédite et les valeurs futures réelles.
- Renormalisation des lignes de M après masque (chaque ligne somme à 1).

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
        'Essaie : Runtime → Disconnect and delete runtime, puis relance.'
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
print(f'Device  : {device}')"""
)

code(
    """# ── Hyperparams (identiques B5) ────────────────────────────────
FREQ              = '1T'
PREDICTION_LENGTH = 120
CONTEXT_LENGTH    = 240
D_MODEL           = 32
N_HEADS           = 2
ENCODER_LAYERS    = 4
DECODER_LAYERS    = 4
EMBEDDING_DIM     = [2]
DROPOUT           = 0.1

USE_EVIDENCE_LAYER   = True
ALPHA_L1             = 0.0
BETA_L2              = 1e-3
GAMMA_ENTROPY_TARGET = 1e-3

CLUSTER_ID = 4
SOURCE_RUN = 'softcam-cluster4-v3-runB5'
RUN_NAME   = 'softcam-cluster4-v3-h1fg-revisited'

# ── Grids du test ──
MIX_GRID = [0.05, 0.10, 0.25]
K_VALUES = [1, 2, 5, 10, 20, 50, 100]

print(f'Run : {RUN_NAME}')
print(f'Source checkpoint : {SOURCE_RUN}')
print(f'Mix testés : {MIX_GRID}')
print(f'k testés   : {K_VALUES}')
print(f'Évaluations : {len(MIX_GRID)} mix × {len(K_VALUES)} k × 5 functions × 2 hypothèses = '
      f'{len(MIX_GRID) * len(K_VALUES) * 5 * 2} forwards teacher-forced')"""
)

md("## 4 — Drive + checkpoint")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE_SRC = f'/content/drive/MyDrive/m2-xai-faas/experiments/{SOURCE_RUN}'
DRIVE_BASE     = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'

for sub in ['results', 'figures']:
    os.makedirs(f'{DRIVE_BASE}/{sub}', exist_ok=True)

ckpt_path = f'{DRIVE_BASE_SRC}/checkpoints/v3_runB5_final.pt'
assert os.path.exists(ckpt_path), f'Checkpoint B5 introuvable : {ckpt_path}'
print(f'Checkpoint : {ckpt_path}')
print(f'Taille     : {os.path.getsize(ckpt_path) / 1e6:.1f} MB')"""
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

train_rows, test_rows = [], []
for ts_idx, s in enumerate(all_series):
    base = {'start': START_DATE, 'feat_static_cat': [ts_idx],
            'cluster': s['cluster'], 'function_id': s['function_id']}
    train_rows.append({**base, 'target': s['target_full'][:-PREDICTION_LENGTH].tolist()})
    test_rows.append( {**base, 'target': s['target_full'].tolist()})

train_dataset = Dataset.from_list(train_rows)
test_dataset  = Dataset.from_list(test_rows)
func_ids = [test_dataset[i]['function_id'] for i in range(len(test_dataset))]
print(f'Cluster {CLUSTER_ID} — fonctions : {func_ids}')"""
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

def create_transformation(freq, config):
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

def create_backtest_dataloader(config, freq, data, batch_size):
    tr = create_transformation(freq, config)
    sp = create_instance_splitter(config, 'validation')
    return as_stacked_batches(sp.apply(tr.apply(data), is_train=True),
                              batch_size=batch_size, output_type=torch.tensor,
                              field_names=_pred_fields(config))

print('Pipeline GluonTS prête.')"""
)

md("## 7 — Construction du modèle + chargement B5")

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
    evidence_mix=0.05,  # peu importe ici — on l'écrasera par MIX_GRID
    alpha_l1=ALPHA_L1,
    beta_l2=BETA_L2,
    gamma_entropy=GAMMA_ENTROPY_TARGET,
)

model = SoftCAMTransformerV3ForPrediction(cfg).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
if missing:    print(f'Missing keys    : {len(missing)} (premier: {missing[0]})')
if unexpected: print(f'Unexpected keys : {len(unexpected)} (premier: {unexpected[0]})')
model.eval()
print('Modèle B5 chargé.')"""
)

md("## 8 — Helpers : extraction M, masque/keep top-k, prédiction MAE")

code(
    """def mask_topk(M: torch.Tensor, k: int, keep_topk: bool = False) -> torch.Tensor:
    '''Masque ou garde les top-k entrées de chaque ligne, puis renormalise.

    Args:
        M       : tensor (B, T, ctx)
        k       : nombre d'entrées top à traiter
        keep_topk : si True, garde seulement les top-k (H1.G).
                    si False, masque les top-k (H1.F).

    Returns:
        M_mod (B, T, ctx) avec lignes renormalisées à 1.
    '''
    _, topk_idx = M.topk(k, dim=-1)
    mask = torch.zeros_like(M, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    if keep_topk:
        M_mod = M * mask.float()
    else:
        M_mod = M * (~mask).float()
    row_sums = M_mod.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return M_mod / row_sums


@torch.no_grad()
def predict_mean(model, b, future_vals, future_obs, M_override=None):
    '''Forward teacher-forced. Si M_override fourni, l'injecte. Renvoie la mean.'''
    if M_override is not None:
        out = model.predict_with_M_override(
            M_override=M_override.to(device),
            past_values=b['past_values'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_values=future_vals,
            future_time_features=b['future_time_features'].to(device),
            static_categorical_features=b['static_categorical_features'].to(device)
                if cfg.num_static_categorical_features > 0 else None,
            future_observed_mask=future_obs,
        )
    else:
        out = model.forward(
            past_values=b['past_values'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_values=future_vals,
            future_time_features=b['future_time_features'].to(device),
            static_categorical_features=b['static_categorical_features'].to(device)
                if cfg.num_static_categorical_features > 0 else None,
            future_observed_mask=future_obs,
        )
    dist = model.output_distribution(out.params, loc=out.loc, scale=out.scale)
    return dist.mean.squeeze(0).cpu().numpy()


print('Helpers prêts.')"""
)

md("## 9 — Extraction de M_baseline + caching des batches")

code(
    """test_loader_expl = create_backtest_dataloader(cfg, FREQ, test_dataset, batch_size=1)
evidence_maps = []   # liste (func_id, M_tensor)
batches_cache = []   # liste (func_id, batch_dict, future_vals, future_obs, actual)

model.eval()
with torch.no_grad():
    for ts_idx, b in enumerate(test_loader_expl):
        func_id = test_dataset[ts_idx]['function_id']
        target_full = np.array(test_dataset[ts_idx]['target'])
        future_vals = torch.tensor(
            target_full[-PREDICTION_LENGTH:], dtype=torch.float32
        ).unsqueeze(0).to(device)
        future_obs = torch.ones_like(future_vals)
        actual = future_vals.squeeze(0).cpu().numpy()

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
        # M est (1, T, ctx)
        evidence_maps.append((func_id, M.detach().cpu()))
        batches_cache.append((func_id, {k: v for k, v in b.items()},
                              future_vals, future_obs, actual))
        print(f'  fn {func_id}  M.shape={tuple(M.shape)}  '
              f'max={M.max().item():.4f}  argmax_mean={int(M.squeeze(0).argmax(dim=1).float().mean().item())}')

print(f'\\n{len(evidence_maps)} cartes M extraites.')"""
)

md(
    """## 10 — Baseline MAE par mix (M apprise, pas d'override)

Pour chaque mix dans la grille, on calcule la prédiction baseline (avec
la M apprise par le modèle pour cette fonction) et la MAE associée.
Ce sera la référence à laquelle on compare les versions perturbées de M.
"""
)

code(
    """baseline_mae_by_mix = {f'{m:.3f}': {} for m in MIX_GRID}

for mix_val in MIX_GRID:
    model.evidence_mix = float(mix_val)
    maes = []
    for fid, b, fv, fo, actual in batches_cache:
        pred = predict_mean(model, b, fv, fo, M_override=None)
        mae = float(np.mean(np.abs(pred - actual)))
        baseline_mae_by_mix[f'{mix_val:.3f}'][fid] = mae
        maes.append(mae)
    mean_mae = float(np.mean(maes))
    print(f'  mix={mix_val:.3f}  MAE baseline moyenne = {mean_mae:.4f}  '
          f'per-fn = {[f"{m:.3f}" for m in maes]}')"""
)

md(
    """## 11 — Sweep H1.F (comprehensiveness) : masque top-k

Pour chaque (mix, k, fonction), on masque les top-k entrées de M par
ligne, on renormalise, on injecte dans le forward, et on mesure la MAE.

Attente : la MAE augmente avec k (au moins jusqu'au plafond mécanique
imposé par mix).
"""
)

code(
    """h1f_results = {f'{m:.3f}': {} for m in MIX_GRID}

for mix_val in tqdm(MIX_GRID, desc='H1.F mix'):
    model.evidence_mix = float(mix_val)
    for k in K_VALUES:
        maes = []
        for (fid, b, fv, fo, actual), (fid2, M_t) in zip(batches_cache, evidence_maps):
            assert fid == fid2
            M_masked = mask_topk(M_t, k, keep_topk=False)
            pred = predict_mean(model, b, fv, fo, M_override=M_masked)
            mae = float(np.mean(np.abs(pred - actual)))
            maes.append((fid, mae))
        mean_mae = float(np.mean([m for _, m in maes]))
        baseline = float(np.mean(list(baseline_mae_by_mix[f'{mix_val:.3f}'].values())))
        rel_deg = (mean_mae - baseline) / baseline
        h1f_results[f'{mix_val:.3f}'][k] = {
            'mae_mean': mean_mae,
            'mae_per_fn': dict(maes),
            'rel_degradation': float(rel_deg),
            'baseline_mae': baseline,
        }
        print(f'  H1.F  mix={mix_val:.3f}  k={k:3d}  '
              f'MAE={mean_mae:.4f}  Δ={rel_deg:+.2%}')"""
)

md(
    """## 12 — Sweep H1.G (sufficiency) : garde top-k uniquement

Pour chaque (mix, k, fonction), on garde uniquement les top-k entrées de
M par ligne, on met le reste à 0, on renormalise, on injecte, on mesure.

Attente : la MAE reste proche du baseline pour k petit (M est sparse).
"""
)

code(
    """h1g_results = {f'{m:.3f}': {} for m in MIX_GRID}

for mix_val in tqdm(MIX_GRID, desc='H1.G mix'):
    model.evidence_mix = float(mix_val)
    for k in K_VALUES:
        maes = []
        for (fid, b, fv, fo, actual), (fid2, M_t) in zip(batches_cache, evidence_maps):
            assert fid == fid2
            M_kept = mask_topk(M_t, k, keep_topk=True)
            pred = predict_mean(model, b, fv, fo, M_override=M_kept)
            mae = float(np.mean(np.abs(pred - actual)))
            maes.append((fid, mae))
        mean_mae = float(np.mean([m for _, m in maes]))
        baseline = float(np.mean(list(baseline_mae_by_mix[f'{mix_val:.3f}'].values())))
        rel_preservation = 1.0 - (mean_mae - baseline) / baseline
        h1g_results[f'{mix_val:.3f}'][k] = {
            'mae_mean': mean_mae,
            'mae_per_fn': dict(maes),
            'rel_preservation': float(rel_preservation),
            'baseline_mae': baseline,
        }
        print(f'  H1.G  mix={mix_val:.3f}  k={k:3d}  '
              f'MAE={mean_mae:.4f}  préservation={rel_preservation:.2%}')"""
)

md("## 13 — Visualisation : H1.F et H1.G à 3 valeurs de mix")

code(
    """fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ks = np.array(K_VALUES)
colors = {'0.050': 'steelblue', '0.100': 'darkorange', '0.250': 'crimson'}

# (1) H1.F — dégradation relative par mix
ax = axes[0]
for mix_val in MIX_GRID:
    mix_key = f'{mix_val:.3f}'
    deg = np.array([h1f_results[mix_key][k]['rel_degradation'] for k in K_VALUES])
    ax.plot(ks, deg * 100, 'o-', color=colors[mix_key], lw=2, ms=7,
            label=f'mix={mix_val:.2f}  (plafond théorique ≤ {mix_val*100:.0f}%)')
    # Plafond mécanique (mix×100%)
    ax.axhline(mix_val * 100, ls=':', color=colors[mix_key], alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('k (nombre d\\'entrées masquées par ligne de M)')
ax.set_ylabel('Δ MAE relative (%)')
ax.set_title('H1.F — Comprehensiveness\\n(↑ = M était utilisée)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.axhline(0, color='black', lw=0.5)

# (2) H1.G — préservation par mix
ax = axes[1]
for mix_val in MIX_GRID:
    mix_key = f'{mix_val:.3f}'
    pres = np.array([h1g_results[mix_key][k]['rel_preservation'] for k in K_VALUES])
    ax.plot(ks, pres * 100, 's-', color=colors[mix_key], lw=2, ms=7,
            label=f'mix={mix_val:.2f}')
ax.axhline(100, color='gray', lw=0.5, ls='--', label='baseline (100%)')
ax.set_xscale('log')
ax.set_xlabel('k (nombre d\\'entrées gardées par ligne de M)')
ax.set_ylabel('Préservation MAE (%)')
ax.set_title('H1.G — Sufficiency\\n(≈100% pour k petit = M sparse-fidèle)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('H1.F / H1.G — k-sweep à 3 valeurs de evidence_mix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/h1fg_revisited.png', dpi=150)
plt.show()"""
)

md(
    """## 14 — Verdict refait à mix=0.25

À mix=0.25, le plafond mécanique est de ~25% (4× plus large qu'à
mix=0.05). On peut maintenant juger :

- **H1.F (comprehensiveness)** : Δ MAE croît-elle avec k ?
- **H1.G (sufficiency)** : la préservation reste-t-elle ≥ 80-90% pour k=5-10 ?
"""
)

code(
    """MIX_KEY = '0.250'

print('=' * 72)
print(f'  VERDICT H1.F / H1.G @ mix=0.25')
print('=' * 72)

# H1.F
print('\\nH1.F (comprehensiveness — masquer top-k de M doit dégrader) :')
deg_k1   = h1f_results[MIX_KEY][1]['rel_degradation']
deg_k10  = h1f_results[MIX_KEY][10]['rel_degradation']
deg_k100 = h1f_results[MIX_KEY][100]['rel_degradation']
deg_max  = max(h1f_results[MIX_KEY][k]['rel_degradation'] for k in K_VALUES)
print(f'  Δ MAE @ k=1   : {deg_k1*100:+.2f}%')
print(f'  Δ MAE @ k=10  : {deg_k10*100:+.2f}%')
print(f'  Δ MAE @ k=100 : {deg_k100*100:+.2f}%')
print(f'  Δ MAE max     : {deg_max*100:+.2f}%  (plafond théorique ≤ 25%)')
h1f_ok = deg_max > 0.05  # > 5% de dégradation max au moins
print(f'  Verdict : {"✅ PASS — M est utilisée causalement" if h1f_ok else "⚠️ effet faible"}')

# H1.G
print('\\nH1.G (sufficiency — garder top-k doit préserver) :')
pres_k1   = h1g_results[MIX_KEY][1]['rel_preservation']
pres_k5   = h1g_results[MIX_KEY][5]['rel_preservation']
pres_k10  = h1g_results[MIX_KEY][10]['rel_preservation']
pres_k50  = h1g_results[MIX_KEY][50]['rel_preservation']
print(f'  Préservation @ k=1   : {pres_k1*100:.2f}%')
print(f'  Préservation @ k=5   : {pres_k5*100:.2f}%')
print(f'  Préservation @ k=10  : {pres_k10*100:.2f}%')
print(f'  Préservation @ k=50  : {pres_k50*100:.2f}%')
h1g_ok = pres_k10 > 0.90  # ≥ 90% à k=10 (sur 240 positions)
print(f'  Verdict : {"✅ PASS — M est sparse-fidèle" if h1g_ok else "⚠️ M utilise plus que top-k"}')

print('=' * 72)"""
)

md("## 15 — Sauvegarde JSON")

code(
    """all_results = {
    'run': RUN_NAME,
    'source_run': SOURCE_RUN,
    'context': (
        'Reprise H1.F (comprehensiveness) et H1.G (sufficiency) sur B5 à '
        '3 valeurs de evidence_mix [0.05, 0.10, 0.25]. predict_with_M_override '
        'passe par forward() teacher-forced, donc Fix #5 n\\'intervient pas. '
        'Le but est de quantifier le ceiling effect et de produire un '
        'verdict au régime opératoire mix=0.25 (config finale H1).'
    ),
    'mix_grid': MIX_GRID,
    'k_values': K_VALUES,
    'baseline_mae_by_mix': baseline_mae_by_mix,
    'h1f': h1f_results,
    'h1g': h1g_results,
    'verdict_at_mix_0.25': {
        'h1f_max_degradation_pct': float(deg_max * 100),
        'h1f_pass': bool(h1f_ok),
        'h1g_preservation_k10_pct': float(pres_k10 * 100),
        'h1g_pass': bool(h1g_ok),
    },
}

with open(f'{DRIVE_BASE}/results/h1fg_revisited.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'Résultats sauvegardés : {DRIVE_BASE}/results/h1fg_revisited.json')"""
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
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook généré : {NOTEBOOK_PATH}")
print(f"Cellules : {len(cells)}")
