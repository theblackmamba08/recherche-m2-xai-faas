"""Génère code/notebooks/softcam-cluster4-v3-mix-ablation.ipynb.

Notebook d'ablation du paramètre ``evidence_mix`` sur le checkpoint Run B5.
Le but est de répondre empiriquement à la question causale fondamentale :

    "M implique-t-elle la prédiction, ou la prédiction est-elle simplement
     un sous-produit de dec_output dont M est un autre sous-produit ?"

Le test : charger le checkpoint B5 (entraîné avec mix=0.05) puis faire
varier ``model.evidence_mix`` en inférence seulement (aucun ré-entraînement)
sur les valeurs ``[0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]``.

Interprétation :
  - Si R² ≈ 0.66 préservé à mix=0.0  → M est CAUSALEMENT DÉCOUPLÉE de la
    prédiction (scénario "intrinsèque faible"). La branche h_evidence est
    cosmétique — le modèle prédit aussi bien sans.
  - Si R² chute fortement à mix=0.0  → M contribue réellement à la
    prédiction. Le ceiling effect de H1.F/H1.G est juste un problème de
    mesure, et H1 est plus solide qu'on pensait.

Sortie : ``DRIVE_BASE/results/mix_ablation.json`` + figures comparatives.

Lance : python _generate_softcam_cluster4_v3_mix_ablation.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4-v3-mix-ablation.ipynb"

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
    """# SoftCAM-Transformer v3 — Ablation `evidence_mix` (Run B5)

Charge le checkpoint de Run B5 et fait varier ``model.evidence_mix`` en
**inférence uniquement** (zéro ré-entraînement) pour répondre à la
question causale fondamentale.

## Pourquoi ce notebook existe

À la fin de l'analyse H1, on a identifié un problème de structure :

```
dec_output ──→ M ──→ h_evidence ──→ (5%)  ──┐
   │                                         ├──→ prédiction
   └────────────────────────────── (95%) ───┘
```

À mix=0.05, h_evidence ne pèse que 5% de la prédiction. **H1.F et H1.G
ne peuvent pas distinguer deux scénarios** qui donneraient exactement les
mêmes chiffres :

| Scénario | Description | H1.F/G visible ? |
|----------|-------------|------------------|
| A. M cause faible | M influence la prédiction, plafond mécanique 5% | Non (~3% max) |
| B. M cosmétique | M est un sous-produit, la prédiction passe par dec_output | Non (~0%) |

## Le test direct

Mettre ``model.evidence_mix = 0.0`` force ``h = dec_output`` (h_evidence
complètement ignoré). On compare la performance à celle de B5 (mix=0.05).

- **R² ≈ B5** → scénario B confirmé. M est décorrélée causalement.
- **R² chute** → scénario A confirmé. M contribue, malgré le plafond.

On balaie ensuite ``mix ∈ {0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0}``
pour cartographier la courbe de contribution complète.

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

md("## 3 — Imports, seed, config (identiques B5)")

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
BATCH_SIZE_TEST   = 64

# ── Evidence Layer v3 (état final B5) ──
USE_EVIDENCE_LAYER   = True
EVIDENCE_MIX_TARGET  = 0.05   # ← valeur d'entraînement (B5)
ALPHA_L1             = 0.0
BETA_L2              = 1e-3
GAMMA_ENTROPY_TARGET = 1e-3

CLUSTER_ID = 4
SOURCE_RUN = 'softcam-cluster4-v3-runB5'
RUN_NAME   = 'softcam-cluster4-v3-mix-ablation'

# B5 reference metrics
B5_R2, B5_SPEAR = 0.6628, 0.9222

# ── Valeurs de mix à balayer en inférence ──
MIX_GRID = [0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

print(f'Source run : {SOURCE_RUN}')
print(f'Analysis   : {RUN_NAME}')
print(f'Reference  : R²={B5_R2}  Spearman={B5_SPEAR}')
print(f'Grille mix : {MIX_GRID}')"""
)

md("## 4 — Google Drive + localisation du checkpoint B5")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE_SRC = f'/content/drive/MyDrive/m2-xai-faas/experiments/{SOURCE_RUN}'
DRIVE_BASE     = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'

for subdir in ['results', 'figures', 'predictions']:
    os.makedirs(f'{DRIVE_BASE}/{subdir}', exist_ok=True)

ckpt_path = f'{DRIVE_BASE_SRC}/checkpoints/v3_runB5_final.pt'
assert os.path.exists(ckpt_path), (
    f'Checkpoint B5 introuvable : {ckpt_path}\\n'
    f'Vérifie que Run B5 a bien été exécuté et que le checkpoint a été sauvegardé.'
)
print(f'Checkpoint trouvé : {ckpt_path}')
print(f'Taille : {os.path.getsize(ckpt_path) / 1e6:.1f} MB')"""
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
    print(f"  function_id={s['function_id']}  longueur={len(s['target_full'])}")

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

md("## 6 — Pipeline GluonTS (identique B5)")

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

md("## 7 — Construction du modèle v3 + chargement checkpoint B5")

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
    evidence_mix=EVIDENCE_MIX_TARGET,
    alpha_l1=ALPHA_L1,
    beta_l2=BETA_L2,
    gamma_entropy=GAMMA_ENTROPY_TARGET,
)

model = SoftCAMTransformerV3ForPrediction(cfg).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f'Checkpoint chargé.')
print(f'  Missing keys    : {len(missing)} (premier: {missing[0] if missing else "—"})')
print(f'  Unexpected keys : {len(unexpected)} (premier: {unexpected[0] if unexpected else "—"})')
model.eval()
print(f'Mode eval. Mix actuel : {model.evidence_mix}')"""
)

md(
    """## 8 — Sanity check (mix=0.05) : reproduit-on R²=0.6628 ?

On vérifie d'abord que le checkpoint chargé reproduit bien les métriques
B5 avec le mix d'entraînement. Si oui, on peut faire confiance aux mesures
qui suivent quand on changera mix.
"""
)

code(
    """from sklearn.metrics import r2_score
from scipy.stats import spearmanr

@torch.no_grad()
def evaluate_generate(model, dataloader, dataset, config, device, prefix='', num_samples=100):
    '''Évalue le modèle en mode generate (autoregressive, stochastique).

    Retourne R², Spearman moyens + par série + array de prédictions médianes
    (shape (n_series, pred_length)).
    '''
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
        print(f'  per-series R²       : {[f"{x:.3f}" for x in r2s]}')
        print(f'  per-series Spearman : {[f"{x:.3f}" for x in spears]}')
    return r2, spear, r2s, spears, forecast_median

# Sanity check avec mix d'entraînement
model.evidence_mix = EVIDENCE_MIX_TARGET
test_loader = create_backtest_dataloader(cfg, FREQ, test_dataset, BATCH_SIZE_TEST)
sanity_r2, sanity_spear, sanity_r2s, sanity_spears, sanity_preds = evaluate_generate(
    model, test_loader, test_dataset, cfg, device,
    prefix=f'SANITY (mix={EVIDENCE_MIX_TARGET})'
)

delta_r2    = (sanity_r2    - B5_R2)    * 100
delta_spear = (sanity_spear - B5_SPEAR) * 100
print(f'\\nΔ vs B5 : R²={delta_r2:+.2f} pp   Spearman={delta_spear:+.2f} pp')
sanity_ok = abs(delta_r2) < 5.0 and abs(delta_spear) < 5.0
print(f'Sanity check : {"OK ✅" if sanity_ok else "DIVERGE ⚠️"}')

# Sauvegarde les prédictions de référence (mix=0.05) pour comparaison ultérieure
np.save(f'{DRIVE_BASE}/predictions/preds_mix0.05.npy', sanity_preds)"""
)

md(
    """## 9 — Balayage de `evidence_mix` (cœur du test)

On évalue le **même modèle** (mêmes poids) à différentes valeurs de
``evidence_mix``. Aucun ré-entraînement.

Conceptuellement :
- ``mix=0.0`` → h = dec_output uniquement (h_evidence ignoré)
- ``mix=0.05`` → la valeur d'entraînement (reference)
- ``mix=1.0`` → h = h_evidence uniquement (dec_output ignoré)

**Le verdict du Scénario A vs B se lit directement à mix=0.0.**
"""
)

code(
    """# Fixer la seed avant chaque évaluation pour rendre generate reproductible
def evaluate_at_mix(model, mix_val, dataloader, dataset, config, device):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model.evidence_mix = mix_val
    return evaluate_generate(model, dataloader, dataset, config, device, prefix='')


mix_results = {}
for mix_val in tqdm(MIX_GRID, desc='Mix sweep'):
    # On reconstruit le dataloader pour avoir des batches déterministes
    loader = create_backtest_dataloader(cfg, FREQ, test_dataset, BATCH_SIZE_TEST)
    r2, spear, r2s, spears, preds = evaluate_at_mix(
        model, mix_val, loader, test_dataset, cfg, device
    )
    func_ids = [test_dataset[i]['function_id'] for i in range(len(test_dataset))]
    mix_results[mix_val] = {
        'r2_mean':       float(r2),
        'spear_mean':    float(spear),
        'r2_per_series': [float(x) for x in r2s],
        'spear_per_series': [float(x) for x in spears],
        'func_ids':      func_ids,
    }
    # Sauvegarde les prédictions médianes pour ce mix
    np.save(f'{DRIVE_BASE}/predictions/preds_mix{mix_val:.3f}.npy', preds)
    print(f'  mix={mix_val:5.3f}  R²={r2:+.4f}  Spearman={spear:+.4f}')

# Restaure mix au value d'entraînement
model.evidence_mix = EVIDENCE_MIX_TARGET"""
)

md(
    """## 10 — Visualisation de la courbe R²/Spearman vs mix

C'est LA figure du notebook. Trois lectures possibles :

- **Plat à mix=0.0** : R² ≈ B5 → M cosmétique (Scénario B)
- **Chute à mix=0.0** : R² < B5 → M contribue (Scénario A)
- **Pic à mix=0.05** : la valeur d'entraînement est optimale → le modèle
  s'est adapté finement à ce mix-là
"""
)

code(
    """mixes_arr = np.array(MIX_GRID)
r2_arr    = np.array([mix_results[m]['r2_mean']    for m in MIX_GRID])
spear_arr = np.array([mix_results[m]['spear_mean'] for m in MIX_GRID])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1) R² vs mix
ax = axes[0]
ax.plot(mixes_arr, r2_arr, 'o-', color='crimson', lw=2, ms=8)
ax.axhline(B5_R2, ls='--', color='gray', label=f'B5 reference (R²={B5_R2})')
ax.axvline(EVIDENCE_MIX_TARGET, ls=':', color='black', alpha=0.5,
           label=f'mix entraînement ({EVIDENCE_MIX_TARGET})')
ax.set_xlabel('evidence_mix (inférence seule)')
ax.set_ylabel('R² moyen (5 fonctions)')
ax.set_title('R² vs mix — le test causal')
ax.set_xscale('symlog', linthresh=0.01)
ax.legend(); ax.grid(alpha=0.3)

# Annoter les points clés
for m, r in zip(mixes_arr, r2_arr):
    if m in (0.0, EVIDENCE_MIX_TARGET, 1.0):
        ax.annotate(f'{r:.3f}', (m, r), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

# (2) Spearman vs mix
ax = axes[1]
ax.plot(mixes_arr, spear_arr, 's-', color='steelblue', lw=2, ms=8)
ax.axhline(B5_SPEAR, ls='--', color='gray', label=f'B5 reference (ρ={B5_SPEAR})')
ax.axvline(EVIDENCE_MIX_TARGET, ls=':', color='black', alpha=0.5,
           label=f'mix entraînement ({EVIDENCE_MIX_TARGET})')
ax.set_xlabel('evidence_mix (inférence seule)')
ax.set_ylabel('Spearman moyen')
ax.set_title('Spearman vs mix')
ax.set_xscale('symlog', linthresh=0.01)
ax.legend(); ax.grid(alpha=0.3)

for m, s in zip(mixes_arr, spear_arr):
    if m in (0.0, EVIDENCE_MIX_TARGET, 1.0):
        ax.annotate(f'{s:.3f}', (m, s), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

plt.suptitle('Ablation evidence_mix sur B5 (zéro ré-entraînement)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/mix_ablation_curves.png', dpi=150)
plt.show()"""
)

md(
    """## 11 — Verdict empirique : Scénario A ou B ?

On répond directement à la question causale en comparant ``mix=0.0`` à
``mix=0.05`` (B5).
"""
)

code(
    """r2_at_0     = mix_results[0.0]['r2_mean']
spear_at_0  = mix_results[0.0]['spear_mean']
r2_at_b5    = mix_results[EVIDENCE_MIX_TARGET]['r2_mean']
spear_at_b5 = mix_results[EVIDENCE_MIX_TARGET]['spear_mean']

delta_r2_pp     = (r2_at_0 - r2_at_b5)    * 100
delta_spear_pp  = (spear_at_0 - spear_at_b5) * 100
rel_drop_r2     = (r2_at_b5 - r2_at_0) / max(abs(r2_at_b5), 1e-9)

# Seuils pour décider du scénario
THRESH_REL_DROP = 0.05  # < 5% de chute relative → Scénario B
THRESH_ABS_PP   = 3.0   # < 3pp absolus           → Scénario B

is_scenario_B = abs(delta_r2_pp) < THRESH_ABS_PP and rel_drop_r2 < THRESH_REL_DROP

print('=' * 72)
print('  VERDICT — Ablation mix=0.0 vs mix=0.05')
print('=' * 72)
print(f'  R² @ mix=0.0   : {r2_at_0:+.4f}')
print(f'  R² @ mix=0.05  : {r2_at_b5:+.4f}')
print(f'  Δ R²           : {delta_r2_pp:+.2f} pp')
print(f'  Chute relative : {rel_drop_r2:+.2%}')
print()
print(f'  Spearman @ mix=0.0   : {spear_at_0:+.4f}')
print(f'  Spearman @ mix=0.05  : {spear_at_b5:+.4f}')
print(f'  Δ Spearman           : {delta_spear_pp:+.2f} pp')
print('=' * 72)

if is_scenario_B:
    print('  ✅ SCÉNARIO B confirmé : M est COSMÉTIQUE')
    print('     La branche h_evidence n\\'est pas causalement nécessaire')
    print('     pour la prédiction. M révèle des patterns structurels')
    print('     mais ne pilote pas la décision du modèle.')
    print()
    print('  → Implication pour le mémoire : H1 doit être reformulée')
    print('    comme "interprétabilité structurelle" plutôt que causale.')
    print('    H1.A/D/E restent valides comme observations sur les')
    print('    représentations internes. H1.F/G sont par construction')
    print('    inconclants — c\\'est désormais expliqué.')
else:
    print('  ✅ SCÉNARIO A confirmé : M CONTRIBUE causalement')
    print('     La branche h_evidence influence réellement la prédiction.')
    print('     Le ceiling effect de H1.F/G était un problème de mesure,')
    print('     pas une absence de signal.')
    print()
    print('  → Implication pour le mémoire : H1 est plus solide qu\\'on')
    print('    pensait. La faithfulness par construction tient.')
    print('    H1.F/G doivent être reformulés avec des perturbations')
    print('    plus extrêmes (uniforme, permutation, etc.).')
print('=' * 72)"""
)

md(
    """## 12 — Analyse per-fonction

Si certaines fonctions chutent à mix=0.0 et d'autres pas, on apprend
quelque chose : M est utile à certaines fonctions mais pas d'autres.
"""
)

code(
    """func_ids_test = [test_dataset[i]['function_id'] for i in range(len(test_dataset))]

# Reshape : matrices (n_funcs, n_mix)
r2_matrix    = np.array([[mix_results[m]['r2_per_series'][i]
                          for m in MIX_GRID]
                         for i in range(len(func_ids_test))])
spear_matrix = np.array([[mix_results[m]['spear_per_series'][i]
                          for m in MIX_GRID]
                         for i in range(len(func_ids_test))])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i, fid in enumerate(func_ids_test):
    ax.plot(mixes_arr, r2_matrix[i], 'o-', label=f'fn {fid}', lw=1.5, ms=6)
ax.axvline(EVIDENCE_MIX_TARGET, ls=':', color='black', alpha=0.5)
ax.set_xlabel('evidence_mix'); ax.set_ylabel('R²')
ax.set_title('R² par fonction vs mix')
ax.set_xscale('symlog', linthresh=0.01)
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
for i, fid in enumerate(func_ids_test):
    ax.plot(mixes_arr, spear_matrix[i], 's-', label=f'fn {fid}', lw=1.5, ms=6)
ax.axvline(EVIDENCE_MIX_TARGET, ls=':', color='black', alpha=0.5)
ax.set_xlabel('evidence_mix'); ax.set_ylabel('Spearman')
ax.set_title('Spearman par fonction vs mix')
ax.set_xscale('symlog', linthresh=0.01)
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Per-function : sensibilité au mix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/mix_ablation_per_function.png', dpi=150)
plt.show()

print('R² par fonction à mix=0.0 vs mix=0.05 :')
for i, fid in enumerate(func_ids_test):
    delta = r2_matrix[i, MIX_GRID.index(0.0)] - r2_matrix[i, MIX_GRID.index(EVIDENCE_MIX_TARGET)]
    print(f'  fn {fid}  mix=0.0: R²={r2_matrix[i, MIX_GRID.index(0.0)]:+.3f}  '
          f'mix=0.05: R²={r2_matrix[i, MIX_GRID.index(EVIDENCE_MIX_TARGET)]:+.3f}  '
          f'Δ={delta*100:+.2f} pp')"""
)

md(
    """## 13 — Visualisation : prédictions vs réalité à différents mix

On prend une fonction représentative (la première du cluster) et on
visualise la prédiction à mix=0.0, mix=0.05 et mix=1.0 superposées avec
les vraies valeurs futures.
"""
)

code(
    """# Choisir une fonction représentative
target_idx = 0
target_full = np.array(test_dataset[target_idx]['target'])
actual = target_full[-PREDICTION_LENGTH:]
fn_id = test_dataset[target_idx]['function_id']

# Charger les prédictions médianes sauvegardées
preds_at = {}
for m in [0.0, EVIDENCE_MIX_TARGET, 1.0]:
    preds_at[m] = np.load(f'{DRIVE_BASE}/predictions/preds_mix{m:.3f}.npy')[target_idx]

fig, ax = plt.subplots(figsize=(14, 5))
t = np.arange(PREDICTION_LENGTH)
ax.plot(t, actual, color='black', lw=2, label='Réel', alpha=0.8)
ax.plot(t, preds_at[0.0],                 '-', color='gray',      lw=1.5,
        label=f'mix=0.0  (R²={mix_results[0.0]["r2_per_series"][target_idx]:.3f})')
ax.plot(t, preds_at[EVIDENCE_MIX_TARGET], '-', color='crimson',   lw=1.5,
        label=f'mix={EVIDENCE_MIX_TARGET}  (R²={mix_results[EVIDENCE_MIX_TARGET]["r2_per_series"][target_idx]:.3f})')
ax.plot(t, preds_at[1.0],                 '-', color='steelblue', lw=1.5,
        label=f'mix=1.0  (R²={mix_results[1.0]["r2_per_series"][target_idx]:.3f})')
ax.set_xlabel('Horizon t (min)'); ax.set_ylabel('Charge')
ax.set_title(f'Prédictions à différents mix — function {fn_id}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/predictions_at_different_mix.png', dpi=150)
plt.show()

# Différence absolue entre prédictions
diff_05_00 = np.abs(preds_at[EVIDENCE_MIX_TARGET] - preds_at[0.0])
diff_10_05 = np.abs(preds_at[1.0] - preds_at[EVIDENCE_MIX_TARGET])
print(f'fn {fn_id}  écart |pred(mix=0.05) - pred(mix=0.0)|  : '
      f'moy={diff_05_00.mean():.3f}  max={diff_05_00.max():.3f}')
print(f'fn {fn_id}  écart |pred(mix=1.0)  - pred(mix=0.05)| : '
      f'moy={diff_10_05.mean():.3f}  max={diff_10_05.max():.3f}')"""
)

md("## 14 — Sauvegarde JSON et synthèse finale")

code(
    """all_results = {
    'run': RUN_NAME,
    'source_run': SOURCE_RUN,
    'b5_reference': {'r2': B5_R2, 'spearman': B5_SPEAR},
    'training_mix': EVIDENCE_MIX_TARGET,
    'mix_grid': MIX_GRID,
    'sanity_check': {
        'r2_mean': float(sanity_r2),
        'spear_mean': float(sanity_spear),
        'delta_r2_vs_b5_pp': float(delta_r2),
        'delta_spear_vs_b5_pp': float(delta_spear),
        'ok': bool(sanity_ok),
    },
    'mix_results': {f'{m:.3f}': mix_results[m] for m in MIX_GRID},
    'verdict': {
        'r2_at_mix_0': float(r2_at_0),
        'r2_at_mix_b5': float(r2_at_b5),
        'delta_r2_pp':  float(delta_r2_pp),
        'spear_at_mix_0': float(spear_at_0),
        'spear_at_mix_b5': float(spear_at_b5),
        'delta_spear_pp': float(delta_spear_pp),
        'rel_drop_r2': float(rel_drop_r2),
        'thresholds': {
            'abs_pp':   float(THRESH_ABS_PP),
            'rel_drop': float(THRESH_REL_DROP),
        },
        'scenario': 'B (cosmetic)' if is_scenario_B else 'A (causal)',
        'is_scenario_B': bool(is_scenario_B),
    },
    'note': (
        'Aucun ré-entraînement. Le modèle B5 est évalué à différentes '
        'valeurs de evidence_mix en inférence uniquement. mix=0.0 force '
        'h = dec_output (h_evidence ignoré). mix=1.0 force h = h_evidence '
        '(dec_output ignoré). Le seed est fixé avant chaque évaluation '
        'pour rendre generate reproductible.'
    ),
}

with open(f'{DRIVE_BASE}/results/mix_ablation.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'Résultats sauvegardés : {DRIVE_BASE}/results/mix_ablation.json')
print()
print('=' * 72)
print(f'  SYNTHÈSE — Ablation evidence_mix sur Run B5')
print('=' * 72)
print(f'  Sanity check (mix=0.05) : R²={sanity_r2:.4f}, Spearman={sanity_spear:.4f} '
      f'({"OK" if sanity_ok else "DIVERGE"})')
print()
print(f'  {"mix":>6}   {"R²":>8}   {"Spearman":>9}   {"Δ R² vs B5 (pp)":>16}')
print(f'  {"---":>6}   {"---":>8}   {"---":>9}   {"---":>16}')
for m in MIX_GRID:
    r = mix_results[m]['r2_mean']
    s = mix_results[m]['spear_mean']
    d = (r - r2_at_b5) * 100
    marker = ' ←' if m == EVIDENCE_MIX_TARGET else ''
    print(f'  {m:6.3f}   {r:+8.4f}   {s:+9.4f}   {d:+16.2f}{marker}')
print()
print(f'  Verdict : Scénario {"B (cosmétique)" if is_scenario_B else "A (causal)"}')
print('=' * 72)"""
)

md(
    """## 15 — Conclusion à interpréter manuellement

Le verdict automatique ci-dessus utilise des seuils heuristiques
(``THRESH_ABS_PP=3.0``, ``THRESH_REL_DROP=0.05``). **L'interprétation
finale** doit considérer :

1. **La forme de la courbe R²(mix)** :
   - Plat partout sauf à mix=1.0 → confirme Scénario B (le mix ne change rien)
   - Pic clair à mix=0.05 → le modèle s'est sur-spécialisé à son mix d'entraînement
   - Monotone → contribution graduelle, à interpréter avec H1.D/E

2. **L'écart par-fonction** :
   - Si une seule fonction tire la moyenne → conclusion à nuancer
   - Si toutes les fonctions sont insensibles à mix → confirme Scénario B

3. **Le cas mix=1.0** :
   - R² catastrophique → h_evidence seule ne sait pas prédire (cohérent avec B5 où mix=0.05 était optimal)
   - R² proche de mix=0.05 → étrangement, h_evidence porte autant que dec_output

4. **Implications pour le mémoire** :
   - Si Scénario B → reformuler la contribution comme *interprétabilité structurelle*
     (M révèle des patterns mais ne pilote pas la décision). Distinguer
     *interprétabilité causale* (que l'on n'atteint pas) de *interprétabilité
     structurelle* (que l'on atteint). C'est une contribution conceptuelle
     originale.
   - Si Scénario A → la fidélité par construction tient. H1.F/G ont juste
     besoin d'être reformulés avec des perturbations plus extrêmes.

Quelle que soit la conclusion, ce notebook produit la preuve empirique
qui manquait à la Discussion du chapitre H1.
"""
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
