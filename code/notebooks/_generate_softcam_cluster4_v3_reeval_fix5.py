"""Génère code/notebooks/softcam-cluster4-v3-reeval-fix5.ipynb.

Notebook de ré-évaluation des checkpoints B5, B6, B7 avec Fix #5 actif
(generate() patché pour appeler output_params au lieu de parameter_projection).

Pourquoi ce notebook existe :

Tous les runs antérieurs (A, B…B7) ont été évalués via HuggingFace generate()
qui court-circuitait notre output_params() — donc l'Evidence Layer était
inactive à l'inférence. Les R² rapportés mesurent uniquement le chemin
dec_output, pas le modèle SoftCAM complet.

Ce qu'on veut savoir maintenant :
  1. Que produit B5 (entraîné mix=0.05) à différents mix d'inférence ?
     → Déjà fait dans mix-ablation. Pic à mix=0.25, R²=0.7563.
  2. Que produit B6 (entraîné mix=0.10) à différents mix d'inférence ?
     → Inconnu. R²=0.48 rapporté = R² à mix_eff=0 (bug).
  3. Que produit B7 (entraîné mix=0.15) à différents mix d'inférence ?
     → Inconnu. R²=-1.62 rapporté = R² à mix_eff=0 (bug).

Hypothèse à tester : un mix d'entraînement plus élevé donne un h_evidence
plus expressif mais un dec_output plus faible. Le R² total à l'inférence
dépend des deux. Existe-t-il un mix d'entraînement qui maximise le R²
total à un mix d'inférence proche ?

Si B6 ou B7 dépasse B5 (R²=0.7563 à mix=0.25), on a la réponse sans
retrain. Sinon, on saura que B5 + inférence mix=0.25 est l'optimum
accessible avec l'architecture actuelle.

Sortie : grille (3 modèles × 7 mix) de R² + Spearman + per-function.

Lance : python _generate_softcam_cluster4_v3_reeval_fix5.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "softcam-cluster4-v3-reeval-fix5.ipynb"

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
    """# SoftCAM v3 — Ré-évaluation B5/B6/B7 avec `generate()` patché (Fix #5)

## Pourquoi ce notebook

Tous les runs antérieurs ont été évalués avec un `generate()` HuggingFace
qui court-circuitait notre `output_params()` — donc l'Evidence Layer était
inactive à l'inférence.

Les chiffres rapportés mesurent uniquement la branche `dec_output` :

| Run | mix entraînement | R² rapporté (bug) | Ce que ça mesure réellement |
|-----|------------------|-------------------|-----------------------------|
| B5 | 0.05 | 0.6628 | dec_output seul de B5 |
| B6 | 0.10 | 0.4782 | dec_output seul de B6 |
| B7 | 0.15 | −1.6244 | dec_output seul de B7 |

**Lecture en creux** : plus le mix d'entraînement est élevé, plus
`dec_output` est faible (B7 collapse). Le modèle s'appuie davantage sur
`h_evidence` pendant l'entraînement → `dec_output` est moins optimisé.

## Question scientifique

Avec Fix #5 actif, on évalue les 3 checkpoints à 7 valeurs de mix
d'inférence : **0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0**.

Hypothèses testables :
- **H-reeval-1** : B6 (entraîné mix=0.10) à mix=0.10 inférence > B5 à mix=0.05 ?
- **H-reeval-2** : Pour chaque modèle, le mix optimal à l'inférence coïncide-t-il avec le mix d'entraînement ?
- **H-reeval-3** : Y a-t-il un (modèle, mix) qui dépasse B5+mix=0.25 (R²=0.7563) ?

Si oui à H-reeval-3 : pas besoin de retrain B8.
Si non : on aura la preuve diagnostique pour la rédaction + on pourra
décider du retrain en connaissance de cause.

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

md("## 2 — Récupération du code (Fix #5 inclus dans v3)")

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

md(
    """## 3 — Vérification Fix #5

Sanity check : la méthode `generate` doit être définie dans
`SoftCAMTransformerV3ForPrediction`, pas héritée de HuggingFace.
"""
)

code(
    """from src.models.softcam_transformer_v3 import (
    SoftCAMTransformerV3Config,
    SoftCAMTransformerV3ForPrediction,
)

# Le qualname doit pointer vers notre classe v3, pas vers HF
qn = SoftCAMTransformerV3ForPrediction.generate.__qualname__
print(f'generate() qualname : {qn}')
assert qn.startswith('SoftCAMTransformerV3ForPrediction.generate'), (
    f"Fix #5 absent. generate() est encore hérité de HuggingFace. "
    f"Vérifie que `git pull` a bien récupéré le commit Fix #5."
)
print('Fix #5 actif — generate() override v3 confirmé.')"""
)

md("## 4 — Imports, seed, config (identique B5)")

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
    """# ── Hyperparams (identiques B5/B6/B7) ───────────────────────────
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

USE_EVIDENCE_LAYER   = True
ALPHA_L1             = 0.0
BETA_L2              = 1e-3
GAMMA_ENTROPY_TARGET = 1e-3

CLUSTER_ID = 4
RUN_NAME   = 'softcam-cluster4-v3-reeval-fix5'

# ── Checkpoints à ré-évaluer + mix d'entraînement de chacun ──
CHECKPOINTS = [
    ('B5', 'softcam-cluster4-v3-runB5', 0.05),
    ('B6', 'softcam-cluster4-v3-runB6', 0.10),
    ('B7', 'softcam-cluster4-v3-runB7', 0.15),
]

# ── Grille d'évaluation : 7 valeurs de mix par modèle ──
MIX_GRID = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0]

# ── R² rapportés (buggés) pour comparaison ──
REPORTED_R2 = {'B5': 0.6628, 'B6': 0.4782, 'B7': -1.6244}
FAYAM_R2 = 0.37

print(f'Run : {RUN_NAME}')
print(f'Modèles à ré-évaluer : {[c[0] for c in CHECKPOINTS]}')
print(f'Grille mix : {MIX_GRID}')
print(f'Total évaluations : {len(CHECKPOINTS) * len(MIX_GRID)}')"""
)

md("## 5 — Google Drive + vérification des checkpoints")

code(
    """from google.colab import drive
drive.mount('/content/drive')

DRIVE_EXP_BASE = '/content/drive/MyDrive/m2-xai-faas/experiments'
DRIVE_BASE     = f'{DRIVE_EXP_BASE}/{RUN_NAME}'
os.makedirs(f'{DRIVE_BASE}/results', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/figures', exist_ok=True)

# Vérifier que les 3 checkpoints existent avant de lancer
ckpt_paths = {}
for tag, run_dir, _ in CHECKPOINTS:
    # Le nom du fichier suit le pattern v3_run{TAG}_final.pt
    fname = f'v3_run{tag}_final.pt'
    path = f'{DRIVE_EXP_BASE}/{run_dir}/checkpoints/{fname}'
    assert os.path.exists(path), (
        f'Checkpoint {tag} introuvable : {path}\\n'
        f'Vérifie que Run {tag} a bien été exécuté.'
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f'  {tag} : {path}  ({size_mb:.1f} MB)')
    ckpt_paths[tag] = path

print(f'\\n3 checkpoints trouvés. Résultats iront dans {DRIVE_BASE}/')"""
)

md("## 6 — Chargement Cluster 4")

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
func_ids_ordered = [test_dataset[i]['function_id'] for i in range(len(test_dataset))]
print(f'\\nDatasets : train={len(train_dataset)}  test={len(test_dataset)}')
print(f'Ordre des fonctions : {func_ids_ordered}')"""
)

md("## 7 — Pipeline GluonTS (identique B5)")

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

md(
    """## 8 — Helpers : build model + load checkpoint + evaluate

`build_model_from_checkpoint(ckpt_path)` :
  - Construit un v3 avec la config standard (identique au moment de l'entraînement)
  - Charge le checkpoint sur le modèle
  - Renvoie le modèle prêt à l'évaluation

`evaluate_at_mix(model, mix_val, ...)` :
  - Fixe `model.evidence_mix = mix_val`
  - Re-seed avant chaque appel pour rendre `generate()` reproductible
  - Renvoie R², Spearman, per-series
"""
)

code(
    """from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def build_cfg():
    return SoftCAMTransformerV3Config(
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
        evidence_mix=0.05,  # peu importe, on l'écrasera
        alpha_l1=ALPHA_L1,
        beta_l2=BETA_L2,
        gamma_entropy=GAMMA_ENTROPY_TARGET,
    )


def build_model_from_checkpoint(ckpt_path):
    cfg = build_cfg()
    model = SoftCAMTransformerV3ForPrediction(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}) — premier : {missing[0]}')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}) — premier : {unexpected[0]}')
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate_at_mix(model, cfg, mix_val, dataset, batch_size=BATCH_SIZE_TEST):
    # Re-seed pour rendre generate() reproductible (sampling stochastique)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model.evidence_mix = float(mix_val)
    model.eval()

    loader = create_backtest_dataloader(cfg, FREQ, dataset, batch_size)
    forecasts = []
    for b in loader:
        out = model.generate(
            static_categorical_features=b['static_categorical_features'].to(device)
                if cfg.num_static_categorical_features > 0 else None,
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
        actual = target[-cfg.prediction_length:]
        pred   = forecast_median[ts_idx]
        r2s.append(float(r2_score(actual, pred)))
        rho, _ = spearmanr(actual, pred)
        spears.append(float(rho))
    return {
        'r2_mean':       float(np.mean(r2s)),
        'spear_mean':    float(np.mean(spears)),
        'r2_per_series': r2s,
        'spear_per_series': spears,
    }


print('Helpers prêts.')"""
)

md(
    """## 9 — Boucle d'évaluation : 3 modèles × 7 mix = 21 évaluations

Pour chaque checkpoint :
  1. Build + load
  2. Pour chaque mix dans la grille : evaluate
  3. Free GPU memory avant le suivant

Durée estimée : ~30s par évaluation × 21 = ~10 min.
"""
)

code(
    """results = {}

for tag, run_dir, train_mix in CHECKPOINTS:
    print(f'\\n{"=" * 64}')
    print(f'  {tag} (entraîné mix={train_mix})  →  ckpt={run_dir}')
    print('=' * 64)

    model, cfg = build_model_from_checkpoint(ckpt_paths[tag])
    results[tag] = {
        'train_mix':       float(train_mix),
        'reported_buggy_r2': REPORTED_R2[tag],
        'by_mix':          {},
    }

    for mix_val in tqdm(MIX_GRID, desc=f'{tag} mix sweep'):
        out = evaluate_at_mix(model, cfg, mix_val, test_dataset)
        results[tag]['by_mix'][f'{mix_val:.3f}'] = out
        marker = ' ← entraîné' if abs(mix_val - train_mix) < 1e-6 else ''
        print(f'  mix={mix_val:5.3f}  R²={out["r2_mean"]:+8.4f}  '
              f'Spearman={out["spear_mean"]:+7.4f}{marker}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'  Modèle {tag} libéré.')

print('\\n21 évaluations terminées.')"""
)

md("## 10 — Tableau récapitulatif")

code(
    """# Matrice R² : lignes = modèles, colonnes = mix
matrix_r2    = np.zeros((len(CHECKPOINTS), len(MIX_GRID)))
matrix_spear = np.zeros((len(CHECKPOINTS), len(MIX_GRID)))
for i, (tag, _, _) in enumerate(CHECKPOINTS):
    for j, mix_val in enumerate(MIX_GRID):
        matrix_r2[i, j]    = results[tag]['by_mix'][f'{mix_val:.3f}']['r2_mean']
        matrix_spear[i, j] = results[tag]['by_mix'][f'{mix_val:.3f}']['spear_mean']

# DataFrame pour affichage propre
df_r2 = pd.DataFrame(
    matrix_r2,
    index=[c[0] for c in CHECKPOINTS],
    columns=[f'{m:.3f}' for m in MIX_GRID],
)
df_r2.index.name = 'model \\\\ mix'
print('=' * 72)
print('  R² par (modèle, mix d\\'inférence)')
print('=' * 72)
print(df_r2.round(4))
print()

# Mix optimal pour chaque modèle
print('Optimum par modèle :')
for i, (tag, _, train_mix) in enumerate(CHECKPOINTS):
    j_opt = int(matrix_r2[i].argmax())
    r2_opt = matrix_r2[i, j_opt]
    mix_opt = MIX_GRID[j_opt]
    r2_at_train = results[tag]['by_mix'][f'{train_mix:.3f}']['r2_mean']
    r2_at_zero  = results[tag]['by_mix']['0.000']['r2_mean']
    delta_causal = r2_at_train - r2_at_zero
    print(f'  {tag} (entraîné mix={train_mix}) :')
    print(f'    R² @ mix=0           : {r2_at_zero:+.4f}  (dec_output seul)')
    print(f'    R² @ mix d\\'entraînement : {r2_at_train:+.4f}')
    print(f'    R² OPTIMUM @ mix={mix_opt:.3f}  : {r2_opt:+.4f}')
    print(f'    Contribution causale de M (Δ vs mix=0) : {delta_causal*100:+.2f} pp')"""
)

md("## 11 — Figure : R²(mix) overlay pour les 3 modèles")

code(
    """fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
mixes_arr = np.array(MIX_GRID)
colors = {'B5': 'crimson', 'B6': 'steelblue', 'B7': 'darkorange'}

# (1) R² overlay
ax = axes[0]
for i, (tag, _, train_mix) in enumerate(CHECKPOINTS):
    r2_line = matrix_r2[i]
    ax.plot(mixes_arr, r2_line, 'o-', color=colors[tag], lw=2, ms=8,
            label=f'{tag} (entraîné mix={train_mix})')
    # Marquer le mix d'entraînement avec un anneau noir
    if train_mix in MIX_GRID:
        j_train = MIX_GRID.index(train_mix)
        ax.scatter([train_mix], [r2_line[j_train]], s=200,
                   facecolors='none', edgecolors='black', lw=2, zorder=10)
ax.axhline(FAYAM_R2, ls='--', color='gray', alpha=0.6, label=f'FAYAM ({FAYAM_R2})')
ax.set_xscale('symlog', linthresh=0.01)
ax.set_xlabel('evidence_mix (inférence)')
ax.set_ylabel('R² moyen (5 fonctions)')
ax.set_title('R² vs mix d\\'inférence — 3 checkpoints\\n(anneau noir = mix d\\'entraînement)')
ax.legend(); ax.grid(alpha=0.3)

# (2) Spearman overlay
ax = axes[1]
for i, (tag, _, train_mix) in enumerate(CHECKPOINTS):
    s_line = matrix_spear[i]
    ax.plot(mixes_arr, s_line, 's-', color=colors[tag], lw=2, ms=8,
            label=f'{tag} (entraîné mix={train_mix})')
    if train_mix in MIX_GRID:
        j_train = MIX_GRID.index(train_mix)
        ax.scatter([train_mix], [s_line[j_train]], s=200,
                   facecolors='none', edgecolors='black', lw=2, zorder=10)
ax.set_xscale('symlog', linthresh=0.01)
ax.set_xlabel('evidence_mix (inférence)'); ax.set_ylabel('Spearman moyen')
ax.set_title('Spearman vs mix — 3 checkpoints')
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Ré-évaluation Fix #5 — courbes R²/Spearman par checkpoint', fontsize=13)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/reeval_curves.png', dpi=150)
plt.show()"""
)

md(
    """## 12 — Figure : décomposition de la contribution causale par modèle

Pour chaque modèle, on isole :
- R² à mix=0 (dec_output seul, "baseline embarquée")
- R² à mix d'entraînement (le régime cohérent du modèle)
- R² à son mix optimal
"""
)

code(
    """fig, ax = plt.subplots(figsize=(10, 5))
tags = [c[0] for c in CHECKPOINTS]
train_mixes = [c[2] for c in CHECKPOINTS]

r2_at_zero  = np.array([results[t]['by_mix']['0.000']['r2_mean']
                        for t, _, _ in CHECKPOINTS])
r2_at_train = np.array([results[t]['by_mix'][f'{m:.3f}']['r2_mean']
                        for t, _, m in CHECKPOINTS])
r2_at_opt   = np.array([float(np.max(matrix_r2[i])) for i in range(len(CHECKPOINTS))])

x = np.arange(len(tags)); w = 0.27
b1 = ax.bar(x - w, r2_at_zero,  w, color='lightgray', label='mix=0 (dec_output seul)')
b2 = ax.bar(x,     r2_at_train, w, color='steelblue', label='mix d\\'entraînement')
b3 = ax.bar(x + w, r2_at_opt,   w, color='crimson',   label='mix optimal')

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h > 0 else -12), textcoords='offset points',
                    ha='center', fontsize=8)

ax.axhline(FAYAM_R2, ls='--', color='gray', alpha=0.5, label=f'FAYAM ({FAYAM_R2})')
ax.set_xticks(x); ax.set_xticklabels([f'{t}\\n(mix={m})' for t, _, m in CHECKPOINTS])
ax.set_ylabel('R² moyen'); ax.set_title('Comparaison checkpoint — 3 régimes de mix')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/figures/reeval_decomposition.png', dpi=150)
plt.show()"""
)

md("## 13 — Sauvegarde JSON")

code(
    """all_results = {
    'run': RUN_NAME,
    'context': (
        'Ré-évaluation des checkpoints B5, B6, B7 avec generate() patché Fix #5. '
        'Chaque checkpoint est évalué à 7 valeurs de evidence_mix en inférence. '
        'Aucun ré-entraînement. Le but est de connaître la vraie performance '
        'de chaque modèle avec l\\'Evidence Layer active, et identifier le mix '
        'd\\'inférence optimal de chacun.'
    ),
    'mix_grid': MIX_GRID,
    'fayam_r2_reference': FAYAM_R2,
    'reported_buggy_r2': REPORTED_R2,
    'per_checkpoint': results,
    'matrix_r2': matrix_r2.tolist(),
    'matrix_spearman': matrix_spear.tolist(),
    'matrix_index':   [c[0] for c in CHECKPOINTS],
    'matrix_columns': [f'{m:.3f}' for m in MIX_GRID],
}

with open(f'{DRIVE_BASE}/results/reeval_fix5.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'Résultats sauvegardés : {DRIVE_BASE}/results/reeval_fix5.json')"""
)

md(
    """## 14 — Synthèse pour décision retrain B8

Question pour le mémoire et la suite : **un (modèle, mix) bat-il B5+mix=0.25 (R²=0.7563) ?**
"""
)

code(
    """B5_INFERENCE_PEAK = 0.7563  # ce qu'on a déjà avec B5 + mix=0.25 inférence
print('=' * 72)
print('  DÉCISION RETRAIN B8 — un (modèle, mix) bat-il B5+mix=0.25 ?')
print('=' * 72)
print(f'  Référence à battre : B5 @ mix=0.25 → R²={B5_INFERENCE_PEAK}')
print()

best_per_model = []
for i, (tag, _, train_mix) in enumerate(CHECKPOINTS):
    j_opt = int(matrix_r2[i].argmax())
    r2_opt = matrix_r2[i, j_opt]
    mix_opt = MIX_GRID[j_opt]
    delta = (r2_opt - B5_INFERENCE_PEAK) * 100
    flag = '🟢 BAT' if r2_opt > B5_INFERENCE_PEAK else '🔴 inférieur'
    print(f'  {tag} optimum @ mix={mix_opt:.3f}  R²={r2_opt:+.4f}  '
          f'Δ vs B5+0.25 = {delta:+.2f} pp   {flag}')
    best_per_model.append((tag, mix_opt, r2_opt))

print()
overall_best = max(best_per_model, key=lambda x: x[2])
print(f'Meilleur global : {overall_best[0]} @ mix={overall_best[1]:.3f}  R²={overall_best[2]:+.4f}')
print()
if overall_best[2] > B5_INFERENCE_PEAK:
    print('  → Pas besoin de retrain B8. On utilise ce checkpoint à ce mix.')
else:
    print('  → Retrain B8 (fine-tune B5 mix=0.05→0.25) reste pertinent pour')
    print('    espérer dépasser B5+inférence mix=0.25.')
print('=' * 72)"""
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
