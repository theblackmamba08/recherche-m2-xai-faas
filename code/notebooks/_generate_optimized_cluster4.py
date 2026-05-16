"""Génère code/notebooks/optimized-cluster4.ipynb (Path B HPO + ablation 949).

Lance : python _generate_optimized_cluster4.py
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / 'optimized-cluster4.ipynb'

cells = []

def md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True),
    })

def code(text):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': text.splitlines(keepends=True),
    })

# ─────────────────────────────────────────────────────────────────────────────
md("""# Optimized Cluster 4 — HPO + Retrain + Ablation

Notebook de contribution (préparation H1) :
1. **HPO** des hyperparamètres `TimeSeriesTransformer` sur le Cluster 4 (Optuna TPE, 15 trials)
2. **Retrain final** avec early stopping sur val R²
3. **Comparaison** vs baseline FAYAM
4. **Ablation per-function** : modèle dédié sur la fonction 949 (la plus problématique)

| Champ | Valeur |
|-------|--------|
| **Run** | `optimized-cluster4` |
| **Phase** | Phase 1 bis — HPO ciblée |
| **Cluster** | 4 (fonctions 949, 951, 952, 953, 954) |
| **Outil HPO** | Optuna TPE + MedianPruner |
| **Espace** | `d_model ∈ {32,64,128}`, `context_length ∈ {240,480,1440}`, `encoder_layers ∈ {2,3,4}`, `lr ∈ log[1e-5, 1e-3]` |
| **Fixés** | `n_heads=2`, `embedding_dim=2`, `dropout=0.1`, `batch_size=256`, `decoder=encoder` |
| **Validation** | 120 min held-out avant le test horizon |

> Avant de lancer : Runtime → Change runtime type → **T4 GPU**
""")

md("## 1 — Setup")

code("""import subprocess
gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu.stdout if gpu.returncode == 0 else 'Pas de GPU — vérifier le runtime.')""")

code("""%%capture
!pip install -q transformers datasets "gluonts[torch]" accelerate evaluate scipy scikit-learn tqdm openpyxl ujson optuna""")

md("## 2 — Imports, seed et configuration")

code("""import random, os, json, gc
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache, partial
from tqdm.notebook import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

SEED = 998
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch : {torch.__version__}')
print(f'Optuna  : {optuna.__version__}')
print(f'Device  : {device}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')""")

code("""# ── Hyperparams FIXÉS (non recherchés) ──────────────────────────
FREQ              = '1T'
PREDICTION_LENGTH = 120
N_HEADS           = 2
EMBEDDING_DIM     = [2]
DROPOUT           = 0.1
BATCH_SIZE_TRAIN  = 256
BATCH_SIZE_TEST   = 64
NUM_BATCHES_EPOCH = 100
BETAS             = (0.9, 0.95)
WEIGHT_DECAY      = 1e-1
GRAD_CLIP_NORM    = 1.0

# ── HPO config ──────────────────────────────────────────────────
HPO_N_TRIALS          = 15
HPO_EPOCHS            = 20
HPO_EVAL_EVERY        = 4
HPO_NUM_BATCHES_EPOCH = 50   # plus petit pour trials rapides

# ── Final retrain ───────────────────────────────────────────────
FINAL_MAX_EPOCHS = 80
FINAL_PATIENCE   = 10

CLUSTER_ID = 4
RUN_NAME   = f'optimized-cluster{CLUSTER_ID}'
print(f'Config OK — Cluster {CLUSTER_ID} | RUN_NAME={RUN_NAME}')""")

md("## 3 — Google Drive")

code("""from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = f'/content/drive/MyDrive/m2-xai-faas/experiments/{RUN_NAME}'
for subdir in ['checkpoints', 'results', 'attentions', 'hpo']:
    os.makedirs(f'{DRIVE_BASE}/{subdir}', exist_ok=True)
print(f'Dossier run : {DRIVE_BASE}')""")

md("""## 4 — Chargement du dataset (Cluster 4)

On construit **4 datasets** à partir des 5 séries du Cluster 4 :

| Dataset | Cible (target) | Usage |
|---------|---------------|-------|
| `hpo_train_dataset` | `target_full[:-240]` | Entraînement pendant HPO |
| `hpo_val_dataset`   | `target_full[:-120]` | Val R² pour HPO et early stopping (prédit positions -240 à -120) |
| `train_dataset`     | `target_full[:-120]` | Train final (= train baseline FAYAM) |
| `test_dataset`      | `target_full`        | Test final (prédit dernières 120 min) |
""")

code("""from pathlib import Path
from datasets import Dataset

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
          f"moy={s['target_full'].mean():.0f}  max={s['target_full'].max():.0f}  min={s['target_full'].min():.0f}")

hpo_train_rows, hpo_val_rows, train_rows, test_rows = [], [], [], []
for ts_idx, s in enumerate(all_series):
    base = {
        'start':           START_DATE,
        'feat_static_cat': [ts_idx],
        'cluster':         s['cluster'],
        'function_id':     s['function_id'],
    }
    hpo_train_rows.append({**base, 'target': s['target_full'][:-2 * PREDICTION_LENGTH].tolist()})
    hpo_val_rows.append(  {**base, 'target': s['target_full'][:-PREDICTION_LENGTH].tolist()})
    train_rows.append(    {**base, 'target': s['target_full'][:-PREDICTION_LENGTH].tolist()})
    test_rows.append(     {**base, 'target': s['target_full'].tolist()})

hpo_train_dataset = Dataset.from_list(hpo_train_rows)
hpo_val_dataset   = Dataset.from_list(hpo_val_rows)
train_dataset     = Dataset.from_list(train_rows)
test_dataset      = Dataset.from_list(test_rows)

print(f'\\nDatasets construits :')
print(f'  hpo_train : {len(hpo_train_dataset)} séries × {len(hpo_train_rows[0][\"target\"])} min')
print(f'  hpo_val   : {len(hpo_val_dataset)} séries × {len(hpo_val_rows[0][\"target\"])} min')
print(f'  train     : {len(train_dataset)} séries × {len(train_rows[0][\"target\"])} min')
print(f'  test      : {len(test_dataset)} séries × {len(test_rows[0][\"target\"])} min')""")

code("""fig, axes = plt.subplots(len(all_series), 1, figsize=(14, 3 * len(all_series)), squeeze=False)
for i, (s, ax) in enumerate(zip(all_series, axes.flatten())):
    ax.plot(s['target_full'][-1440:], lw=0.7, color='steelblue')
    ax.set_title(f"Fonction {s['function_id']} — dernières 24h")
    ax.set_xlabel('Minutes'); ax.set_ylabel('Invocations/min')
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

md("## 5 — Pipeline GluonTS (paramétré par config)")

code("""from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
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

for ds in (hpo_train_dataset, hpo_val_dataset, train_dataset, test_dataset):
    ds.set_transform(partial(transform_start_field, freq=FREQ))

print(f'Lags : {len(lags_sequence)} | Time features : {len(time_features)}')""")

code("""def create_transformation(freq, config):
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

def create_attention_dataloader(config, freq, data, batch_size=1):
    fields = _pred_fields(config) + ['future_values', 'future_observed_mask']
    tr = create_transformation(freq, config)
    sp = create_instance_splitter(config, 'validation')
    return as_stacked_batches(sp.apply(tr.apply(data), is_train=True),
                              batch_size=batch_size, output_type=torch.tensor,
                              field_names=fields)

print('Pipeline GluonTS prêt.')""")

md("""## 6 — HPO (Optuna TPE + MedianPruner)

**Stratégie** :
- **Sampler** TPE (Tree-structured Parzen) — bien plus efficace que random search
- **Pruner** Median (élague les trials sous la médiane après 5 epochs de warm-up)
- **Objectif** : moyenne R² sur les 5 fonctions, sur la val (positions -240 à -120)
- **Budget** : 15 trials × 20 epochs, avec pruning agressif → ~1-2h sur T4
- **Storage SQLite sur Drive** : si Colab disconnect, on relance et on reprend
""")

code("""from sklearn.metrics import r2_score

@torch.no_grad()
def evaluate_mean_r2(model, dataloader, dataset, config, device):
    \"\"\"Mean R² across all series of the dataset.\"\"\"
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
    r2_scores = []
    for ts_idx in range(len(dataset)):
        target = np.array(dataset[ts_idx]['target'])
        actual = target[-config.prediction_length:]
        pred = forecast_median[ts_idx]
        r2_scores.append(r2_score(actual, pred))
    return float(np.mean(r2_scores)), r2_scores

print('Helper évaluation R² prêt.')""")

code("""from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from torch.optim import AdamW

def build_config(d_model, context_length, encoder_layers, decoder_layers=None, n_series=None):
    if decoder_layers is None:
        decoder_layers = encoder_layers
    if n_series is None:
        n_series = len(train_dataset)
    return TimeSeriesTransformerConfig(
        prediction_length=PREDICTION_LENGTH,
        context_length=context_length,
        lags_sequence=lags_sequence,
        num_time_features=len(time_features) + 1,
        num_static_categorical_features=1,
        cardinality=[n_series],
        embedding_dimension=EMBEDDING_DIM,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        d_model=d_model,
        encoder_attention_heads=N_HEADS,
        decoder_attention_heads=N_HEADS,
        encoder_ffn_dim=max(d_model, 32),
        decoder_ffn_dim=max(d_model, 32),
        dropout=DROPOUT,
    )

def objective(trial):
    d_model        = trial.suggest_categorical('d_model', [32, 64, 128])
    context_length = trial.suggest_categorical('context_length', [240, 480, 1440])
    encoder_layers = trial.suggest_int('encoder_layers', 2, 4)
    lr             = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    print(f'\\nTrial {trial.number}: d_model={d_model}, context_length={context_length}, '
          f'encoder_layers={encoder_layers}, lr={lr:.2e}')

    cfg = build_config(d_model, context_length, encoder_layers)

    try:
        model = TimeSeriesTransformerForPrediction(cfg).to(device)
        train_loader = create_train_dataloader(cfg, FREQ, hpo_train_dataset,
                                               BATCH_SIZE_TRAIN, HPO_NUM_BATCHES_EPOCH)
        val_loader   = create_backtest_dataloader(cfg, FREQ, hpo_val_dataset, BATCH_SIZE_TEST)
        optimizer = AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

        best_r2 = -float('inf')
        for epoch in range(HPO_EPOCHS):
            model.train()
            epoch_losses = []
            for b in train_loader:
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
                epoch_losses.append(out.loss.item())

            if (epoch + 1) % HPO_EVAL_EVERY == 0:
                val_r2, _ = evaluate_mean_r2(model, val_loader, hpo_val_dataset, cfg, device)
                best_r2 = max(best_r2, val_r2)
                print(f'  epoch {epoch:02d}  loss={np.mean(epoch_losses):.4f}  val_r2={val_r2:.4f}')
                trial.report(val_r2, epoch)
                if trial.should_prune():
                    print(f'  → pruned @ epoch {epoch}')
                    del model, optimizer, train_loader, val_loader
                    torch.cuda.empty_cache(); gc.collect()
                    raise optuna.TrialPruned()

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache(); gc.collect()
        return best_r2

    except torch.cuda.OutOfMemoryError:
        print('  → OOM, pruning trial')
        torch.cuda.empty_cache(); gc.collect()
        raise optuna.TrialPruned()

print('Objectif Optuna prêt.')""")

code("""study_path = f'{DRIVE_BASE}/hpo/study.db'
study = optuna.create_study(
    study_name='c4-hpo',
    storage=f'sqlite:///{study_path}',
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    load_if_exists=True,
    direction='maximize',
)

n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f'Trials déjà complétés (resume) : {n_done}')
print(f'Lancement (target : {HPO_N_TRIALS} trials)...\\n')

study.optimize(objective, n_trials=HPO_N_TRIALS, show_progress_bar=False)

print('\\n' + '=' * 60)
print(f'HPO terminée — {len(study.trials)} trials, best val R² = {study.best_value:.4f}')
print(f'Best params : {study.best_params}')

with open(f'{DRIVE_BASE}/hpo/best_params.json', 'w') as f:
    json.dump({
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned':    len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }, f, indent=2)""")

md("## 7 — Visualisation des résultats HPO")

code("""import optuna.visualization.matplotlib as ovm

fig = ovm.plot_optimization_history(study)
plt.gcf().set_size_inches(10, 4); plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/hpo/optimization_history.png', dpi=150); plt.show()""")

code("""try:
    fig = ovm.plot_param_importances(study)
    plt.gcf().set_size_inches(8, 4); plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/hpo/param_importances.png', dpi=150); plt.show()
except (ValueError, RuntimeError) as e:
    print(f'(Importances indisponibles — trop peu de trials complétés : {e})')""")

code("""fig = ovm.plot_parallel_coordinate(study)
plt.gcf().set_size_inches(12, 4); plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/hpo/parallel_coordinate.png', dpi=150); plt.show()""")

code("""print('=' * 60)
print('Best hyperparameters found:')
print('=' * 60)
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f'  {k:<20s} = {v:.2e}')
    else:
        print(f'  {k:<20s} = {v}')
print(f'\\n  Best val R²       = {study.best_value:.4f}')
print(f'  Baseline FAYAM R² = 0.3701 (référence)')
print(f'  Improvement       = {(study.best_value - 0.3701) * 100:+.2f} pp')

trials_df = study.trials_dataframe(attrs=('number', 'value', 'state', 'params'))
trials_df = trials_df[trials_df['state'] == 'COMPLETE'].sort_values('value', ascending=False).head(5)
print('\\n── Top 5 trials ──')
print(trials_df.to_string(index=False))""")

md("""## 8 — Entraînement final avec early stopping

- Hyperparams = `study.best_params`
- Train sur `train_dataset` (= train baseline FAYAM)
- Validation sur `hpo_val_dataset` pour early stopping
- Patience = 10 epochs, max = 80 epochs
- Sauvegarde du meilleur état (val R² max)
""")

code("""best_d_model = study.best_params['d_model']
best_ctx     = study.best_params['context_length']
best_layers  = study.best_params['encoder_layers']
best_lr      = study.best_params['lr']

cfg_opt = build_config(best_d_model, best_ctx, best_layers)
model_opt = TimeSeriesTransformerForPrediction(cfg_opt).to(device)

n_params = sum(p.numel() for p in model_opt.parameters())
print('Modèle optimisé construit :')
print(f'  d_model        = {best_d_model}')
print(f'  context_length = {best_ctx}')
print(f'  encoder_layers = {best_layers}')
print(f'  decoder_layers = {best_layers}')
print(f'  lr             = {best_lr:.2e}')
print(f'  paramètres     = {n_params:,}')""")

code("""train_loader_opt = create_train_dataloader(cfg_opt, FREQ, train_dataset,
                                           BATCH_SIZE_TRAIN, NUM_BATCHES_EPOCH)
val_loader_opt   = create_backtest_dataloader(cfg_opt, FREQ, hpo_val_dataset, BATCH_SIZE_TEST)

optimizer_opt = AdamW(model_opt.parameters(), lr=best_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

loss_history, val_r2_history = [], []
best_val_r2 = -float('inf')
patience_counter = 0
best_state = None
ckpt_path = f'{DRIVE_BASE}/checkpoints/model_optimized_final.pt'

for epoch in range(FINAL_MAX_EPOCHS):
    model_opt.train()
    epoch_losses = []
    pbar = tqdm(train_loader_opt, desc=f'Epoch {epoch:03d}', leave=False)
    for b in pbar:
        optimizer_opt.zero_grad()
        out = model_opt(
            static_categorical_features=b['static_categorical_features'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            future_values=b['future_values'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_observed_mask=b['future_observed_mask'].to(device),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model_opt.parameters(), GRAD_CLIP_NORM)
        optimizer_opt.step()
        epoch_losses.append(out.loss.item())
        pbar.set_postfix({'loss': f'{out.loss.item():.4f}'})

    mean_loss = float(np.mean(epoch_losses))
    loss_history.append({'epoch': epoch, 'loss': mean_loss})

    val_r2, _ = evaluate_mean_r2(model_opt, val_loader_opt, hpo_val_dataset, cfg_opt, device)
    val_r2_history.append({'epoch': epoch, 'val_r2': val_r2})

    line = f'Epoch {epoch:03d}  loss={mean_loss:.4f}  val_r2={val_r2:.4f}'
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model_opt.state_dict().items()}
        torch.save({'epoch': epoch, 'model': best_state, 'val_r2': val_r2,
                    'loss_history': loss_history, 'val_r2_history': val_r2_history,
                    'hyperparameters': study.best_params}, ckpt_path)
        print(line + '  ← best (saved)')
    else:
        patience_counter += 1
        print(line + f'  (patience {patience_counter}/{FINAL_PATIENCE})')
        if patience_counter >= FINAL_PATIENCE:
            print(f'\\nEarly stopping @ epoch {epoch} (best val_r2={best_val_r2:.4f})')
            break

if best_state is not None:
    model_opt.load_state_dict(best_state)
print(f'\\nEntraînement final terminé — best val_r2 = {best_val_r2:.4f}')""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot([x['epoch'] for x in loss_history], [x['loss'] for x in loss_history],
             lw=1.5, color='steelblue')
axes[0].set_xlabel('Époque'); axes[0].set_ylabel('Loss NLL')
axes[0].set_title('Courbe de perte (train)'); axes[0].grid(alpha=0.3)

axes[1].plot([x['epoch'] for x in val_r2_history], [x['val_r2'] for x in val_r2_history],
             lw=1.5, color='tomato')
axes[1].axhline(0.3701, color='gray', lw=1, ls='--', label='Baseline FAYAM (test R²)')
axes[1].set_xlabel('Époque'); axes[1].set_ylabel('Val R²')
axes[1].set_title('R² validation par époque'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/training_curves.png', dpi=150); plt.show()""")

md("## 9 — Inférence et métriques sur test")

code("""test_loader_opt = create_backtest_dataloader(cfg_opt, FREQ, test_dataset, BATCH_SIZE_TEST)
model_opt.eval()
forecasts_opt = []
with torch.no_grad():
    for b in tqdm(test_loader_opt, desc='Inférence test'):
        out = model_opt.generate(
            static_categorical_features=b['static_categorical_features'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
        )
        forecasts_opt.append(out.sequences.cpu().numpy())
forecasts_opt = np.vstack(forecasts_opt)
print(f'Shape forecasts : {forecasts_opt.shape}')""")

code("""from evaluate import load as load_metric
from gluonts.time_feature import get_seasonality
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

mase_metric  = load_metric('evaluate-metric/mase')
smape_metric = load_metric('evaluate-metric/smape')
forecast_median_opt = np.median(forecasts_opt, axis=1)

all_metrics_opt = []
for ts_idx in range(len(test_dataset)):
    pred   = forecast_median_opt[ts_idx]
    actual = np.array(test_dataset[ts_idx]['target'][-PREDICTION_LENGTH:])
    train_ = np.array(test_dataset[ts_idx]['target'][:-PREDICTION_LENGTH])

    mase  = mase_metric.compute(predictions=pred, references=actual,
                                 training=train_,
                                 periodicity=get_seasonality(FREQ))['mase']
    smape = smape_metric.compute(predictions=pred, references=actual)['smape']
    rmse  = float(np.sqrt(mean_squared_error(actual, pred)))
    r2    = float(r2_score(actual, pred))
    sp    = float(spearmanr(actual, pred).statistic)

    all_metrics_opt.append({
        'ts_index':    ts_idx,
        'function_id': test_dataset[ts_idx]['function_id'],
        'MASE':  mase, 'sMAPE': smape, 'RMSE': rmse,
        'R2':    r2,   'Spearman': sp,
    })

df_metrics_opt = pd.DataFrame(all_metrics_opt)
print('── Métriques optimisé par fonction ──')
print(df_metrics_opt[['function_id', 'MASE', 'sMAPE', 'RMSE', 'R2', 'Spearman']].to_string(index=False))
print('\\n── Moyennes ──')
print(df_metrics_opt[['MASE', 'sMAPE', 'RMSE', 'R2', 'Spearman']].mean().round(4))""")

code("""df_metrics_opt.to_csv(f'{DRIVE_BASE}/results/metrics_optimized.csv', index=False)
with open(f'{DRIVE_BASE}/results/metrics_optimized.json', 'w') as f:
    json.dump({
        'run_name': RUN_NAME, 'cluster': CLUSTER_ID, 'seed': SEED,
        'hyperparameters': study.best_params,
        'val_r2_best': best_val_r2,
        'global': df_metrics_opt[['MASE','sMAPE','RMSE','R2','Spearman']].mean().round(4).to_dict(),
        'per_function': df_metrics_opt.to_dict(orient='records'),
    }, f, indent=2)
print('Métriques sauvegardées.')""")

md("## 10 — Prévisions par fonction")

code("""CONTEXT_DISPLAY = 360

for ts_idx, s in enumerate(all_series):
    func_id = s['function_id']
    row     = df_metrics_opt[df_metrics_opt['ts_index'] == ts_idx].iloc[0]

    actual  = np.array(test_dataset[ts_idx]['target'])
    pred_m  = forecast_median_opt[ts_idx]
    pred_lo = forecasts_opt[ts_idx].mean(0) - forecasts_opt[ts_idx].std(0)
    pred_hi = forecasts_opt[ts_idx].mean(0) + forecasts_opt[ts_idx].std(0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    ax = axes[0]
    n_ctx  = CONTEXT_DISPLAY
    x_ctx  = np.arange(-n_ctx, 0)
    x_pred = np.arange(0, PREDICTION_LENGTH)
    ax.plot(x_ctx,  actual[-(n_ctx + PREDICTION_LENGTH):-PREDICTION_LENGTH],
            color='steelblue', lw=1, label='Réel (contexte)')
    ax.plot(x_pred, actual[-PREDICTION_LENGTH:],
            color='steelblue', lw=1.5, ls='--', label='Réel (horizon)')
    ax.plot(x_pred, pred_m,  color='tomato', lw=1.5, label='Médiane prédite')
    ax.fill_between(x_pred, pred_lo, pred_hi, color='tomato', alpha=0.2, label='±1 std')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Minutes (0 = début horizon)'); ax.set_ylabel('Invocations/min')
    ax.set_title(f'Fct {func_id} — zoom 6h + 2h'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[1]
    n_day = 1440
    x_day  = np.arange(-n_day, 0)
    ax2.plot(x_day, actual[-(n_day + PREDICTION_LENGTH):-PREDICTION_LENGTH],
             color='steelblue', lw=0.6, label='Réel (24h)')
    ax2.plot(x_pred, actual[-PREDICTION_LENGTH:],
             color='steelblue', lw=1.5, ls='--')
    ax2.plot(x_pred, pred_m, color='tomato', lw=1.5)
    ax2.fill_between(x_pred, pred_lo, pred_hi, color='tomato', alpha=0.2)
    ax2.axvline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('Minutes (0 = début horizon)')
    ax2.set_title(f'Fct {func_id} — vue 24h'); ax2.grid(alpha=0.3)

    fig.suptitle(
        f'Fonction {func_id} (Cluster {CLUSTER_ID}, OPTIMISÉ)  |  '
        f'MASE={row.MASE:.3f}  sMAPE={row.sMAPE:.3f}  RMSE={row.RMSE:.0f}  '
        f'R²={row.R2:.3f}  Spearman={row.Spearman:.3f}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{DRIVE_BASE}/results/forecast_function_{func_id}.png', dpi=150)
    plt.show()""")

md("## 11 — Comparaison vs FAYAM baseline")

code("""# Baseline FAYAM (du run baseline-cluster4 du 2026-05-05)
fayam_metrics = pd.DataFrame([
    {'function_id': 949, 'MASE': 1.698746, 'sMAPE': 0.221424, 'RMSE': 29.720628, 'R2': 0.146010, 'Spearman': 0.953155},
    {'function_id': 951, 'MASE': 1.621126, 'sMAPE': 0.212783, 'RMSE': 27.735318, 'R2': 0.231998, 'Spearman': 0.956879},
    {'function_id': 952, 'MASE': 0.973539, 'sMAPE': 0.239281, 'RMSE': 25.098935, 'R2': 0.429430, 'Spearman': 0.903363},
    {'function_id': 953, 'MASE': 0.927301, 'sMAPE': 0.180401, 'RMSE': 17.979290, 'R2': 0.604056, 'Spearman': 0.897334},
    {'function_id': 954, 'MASE': 0.927908, 'sMAPE': 0.233136, 'RMSE': 22.719225, 'R2': 0.438801, 'Spearman': 0.886872},
])

cmp = fayam_metrics.merge(df_metrics_opt, on='function_id', suffixes=('_fayam', '_opt'))

print('=' * 90)
print('Comparaison FAYAM baseline vs Optimisé — Cluster 4')
print('=' * 90)
for _, row in cmp.iterrows():
    print(f'\\nFonction {int(row.function_id)}:')
    for metric in ['MASE', 'sMAPE', 'RMSE', 'R2', 'Spearman']:
        f = row[f'{metric}_fayam']; o = row[f'{metric}_opt']; diff = o - f
        better = '↑' if (metric in ['R2', 'Spearman'] and diff > 0) or (metric in ['MASE', 'sMAPE', 'RMSE'] and diff < 0) else '↓'
        print(f'  {metric:<10s}  FAYAM={f:.4f}   Opt={o:.4f}   Δ={diff:+.4f}  {better}')

print('\\n' + '=' * 90)
print('Moyennes :')
print('=' * 90)
for metric in ['MASE', 'sMAPE', 'RMSE', 'R2', 'Spearman']:
    f = cmp[f'{metric}_fayam'].mean(); o = cmp[f'{metric}_opt'].mean()
    print(f'  {metric:<10s}  FAYAM={f:.4f}   Opt={o:.4f}   Δ={o-f:+.4f}')

cmp.to_csv(f'{DRIVE_BASE}/results/comparison_fayam_vs_optimized.csv', index=False)""")

code("""fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, m in enumerate(['MASE', 'sMAPE', 'RMSE', 'R2', 'Spearman']):
    x = np.arange(len(cmp)); w = 0.35
    axes[i].bar(x - w/2, cmp[f'{m}_fayam'], w, label='FAYAM', color='gray', alpha=0.7)
    axes[i].bar(x + w/2, cmp[f'{m}_opt'],   w, label='Optimisé', color='tomato', alpha=0.85)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([str(int(f)) for f in cmp['function_id']])
    axes[i].set_xlabel('Fonction'); axes[i].set_title(m); axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3, axis='y')

plt.suptitle('Comparaison par fonction — FAYAM vs Optimisé', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/comparison_barchart.png', dpi=150); plt.show()""")

md("## 12 — Extraction attention (sanity check pour H1)")

code("""attn_loader = create_attention_dataloader(cfg_opt, FREQ, test_dataset, batch_size=1)

model_opt.eval()
cross_attn_list, enc_attn_list = [], []

with torch.no_grad():
    for i, batch in enumerate(tqdm(attn_loader, total=len(all_series), desc='Attention')):
        if i >= len(all_series): break
        out = model_opt(
            past_values=batch['past_values'].to(device),
            past_time_features=batch['past_time_features'].to(device),
            past_observed_mask=batch['past_observed_mask'].to(device),
            future_values=batch['future_values'].to(device),
            future_time_features=batch['future_time_features'].to(device),
            future_observed_mask=batch['future_observed_mask'].to(device),
            static_categorical_features=batch['static_categorical_features'].to(device),
            output_attentions=True,
        )
        cross_attn_list.append(out.cross_attentions[-1].squeeze(0).cpu().numpy())
        enc_attn_list.append(out.encoder_attentions[-1].squeeze(0).cpu().numpy())

cross_attn_arr = np.stack(cross_attn_list)
enc_attn_arr   = np.stack(enc_attn_list)

np.save(f'{DRIVE_BASE}/attentions/cross_attentions_last_layer.npy', cross_attn_arr)
np.save(f'{DRIVE_BASE}/attentions/encoder_attentions_last_layer.npy', enc_attn_arr)
print(f'cross_attn_arr : {cross_attn_arr.shape}')
print(f'enc_attn_arr   : {enc_attn_arr.shape}')""")

code("""fig, axes = plt.subplots(1, len(all_series) + 1, figsize=(6 * (len(all_series) + 1), 4))

mean_cross = cross_attn_arr.mean(axis=(0, 1))
im = axes[0].imshow(mean_cross, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(im, ax=axes[0])
axes[0].set_title('Cross-attn moyenne')
axes[0].set_xlabel('Contexte passé'); axes[0].set_ylabel('Horizon prédit')

for ts_idx, s in enumerate(all_series):
    mc = cross_attn_arr[ts_idx].mean(axis=0)
    im2 = axes[ts_idx + 1].imshow(mc, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im2, ax=axes[ts_idx + 1])
    axes[ts_idx + 1].set_title(f'Fct {s["function_id"]}')
    axes[ts_idx + 1].set_xlabel('Contexte passé')

plt.suptitle(f'Cross-attention dernière couche — Cluster {CLUSTER_ID} (OPTIMISÉ)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/attentions/cross_attention_heatmap.png', dpi=150); plt.show()""")

md("""## 13 — Ablation : modèle dédié sur la fonction 949

Diagnostic per-function. On entraîne un modèle dédié à la 949 avec les hyperparams optimaux trouvés.
Résultat possible :
- R² 949-dédié **>** R² 949-multi → la fonction est *écrasée* par le multi-task
- R² 949-dédié **≈** R² 949-multi → la fonction est *intrinsèquement difficile*
- R² 949-dédié **<** R² 949-multi → le multi-task apporte un *transfert positif*
""")

code("""s_949 = next(s for s in all_series if s['function_id'] == 949)

train_949 = Dataset.from_list([{
    'start': START_DATE, 'feat_static_cat': [0], 'cluster': CLUSTER_ID, 'function_id': 949,
    'target': s_949['target_full'][:-PREDICTION_LENGTH].tolist(),
}])
test_949 = Dataset.from_list([{
    'start': START_DATE, 'feat_static_cat': [0], 'cluster': CLUSTER_ID, 'function_id': 949,
    'target': s_949['target_full'].tolist(),
}])
hpo_val_949 = Dataset.from_list([{
    'start': START_DATE, 'feat_static_cat': [0], 'cluster': CLUSTER_ID, 'function_id': 949,
    'target': s_949['target_full'][:-PREDICTION_LENGTH].tolist(),
}])
for ds in (train_949, test_949, hpo_val_949):
    ds.set_transform(partial(transform_start_field, freq=FREQ))

cfg_949 = build_config(best_d_model, best_ctx, best_layers, n_series=1)
model_949 = TimeSeriesTransformerForPrediction(cfg_949).to(device)
optimizer_949 = AdamW(model_949.parameters(), lr=best_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

train_loader_949 = create_train_dataloader(cfg_949, FREQ, train_949, BATCH_SIZE_TRAIN, NUM_BATCHES_EPOCH)
val_loader_949   = create_backtest_dataloader(cfg_949, FREQ, hpo_val_949, BATCH_SIZE_TEST)

loss_hist_949, val_hist_949 = [], []
best_val_949 = -float('inf')
patience = 0
best_state_949 = None

print('Entraînement modèle dédié 949...')
for epoch in range(FINAL_MAX_EPOCHS):
    model_949.train()
    losses = []
    for b in train_loader_949:
        optimizer_949.zero_grad()
        out = model_949(
            static_categorical_features=b['static_categorical_features'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            future_values=b['future_values'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
            future_observed_mask=b['future_observed_mask'].to(device),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model_949.parameters(), GRAD_CLIP_NORM)
        optimizer_949.step()
        losses.append(out.loss.item())

    val_r2_949, _ = evaluate_mean_r2(model_949, val_loader_949, hpo_val_949, cfg_949, device)
    loss_hist_949.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    val_hist_949.append({'epoch': epoch, 'val_r2': val_r2_949})

    line = f'Epoch {epoch:03d}  loss={np.mean(losses):.4f}  val_r2={val_r2_949:.4f}'
    if val_r2_949 > best_val_949:
        best_val_949 = val_r2_949
        patience = 0
        best_state_949 = {k: v.cpu().clone() for k, v in model_949.state_dict().items()}
        print(line + '  ← best')
    else:
        patience += 1
        print(line + f'  (patience {patience}/{FINAL_PATIENCE})')
        if patience >= FINAL_PATIENCE:
            print(f'Early stop @ epoch {epoch}')
            break

model_949.load_state_dict(best_state_949)
torch.save({'model': best_state_949, 'hyperparameters': study.best_params,
            'val_r2_best': best_val_949},
           f'{DRIVE_BASE}/checkpoints/model_949_dedicated.pt')""")

code("""test_loader_949 = create_backtest_dataloader(cfg_949, FREQ, test_949, BATCH_SIZE_TEST)
model_949.eval()
with torch.no_grad():
    forecasts_949 = []
    for b in test_loader_949:
        out = model_949.generate(
            static_categorical_features=b['static_categorical_features'].to(device),
            past_time_features=b['past_time_features'].to(device),
            past_values=b['past_values'].to(device),
            future_time_features=b['future_time_features'].to(device),
            past_observed_mask=b['past_observed_mask'].to(device),
        )
        forecasts_949.append(out.sequences.cpu().numpy())
forecasts_949 = np.vstack(forecasts_949)
pred_949 = np.median(forecasts_949, axis=1)[0]

actual_949 = np.array(test_949[0]['target'][-PREDICTION_LENGTH:])
train_949_vals = np.array(test_949[0]['target'][:-PREDICTION_LENGTH])

mase_949  = mase_metric.compute(predictions=pred_949, references=actual_949,
                                 training=train_949_vals,
                                 periodicity=get_seasonality(FREQ))['mase']
smape_949 = smape_metric.compute(predictions=pred_949, references=actual_949)['smape']
rmse_949  = float(np.sqrt(mean_squared_error(actual_949, pred_949)))
r2_949    = float(r2_score(actual_949, pred_949))
sp_949    = float(spearmanr(actual_949, pred_949).statistic)

print('=' * 70)
print('Ablation 949 — diagnostic per-function')
print('=' * 70)
print(f'{"Metric":<10s}  {"FAYAM":>10s}  {"Multi-Opt":>10s}  {"949-Dédié":>10s}')
fayam_949 = fayam_metrics[fayam_metrics['function_id'] == 949].iloc[0]
opt_949   = df_metrics_opt[df_metrics_opt['function_id'] == 949].iloc[0]
for m, f, mo, d in [
    ('MASE',     fayam_949.MASE,     opt_949.MASE,     mase_949),
    ('sMAPE',    fayam_949.sMAPE,    opt_949.sMAPE,    smape_949),
    ('RMSE',     fayam_949.RMSE,     opt_949.RMSE,     rmse_949),
    ('R2',       fayam_949.R2,       opt_949.R2,       r2_949),
    ('Spearman', fayam_949.Spearman, opt_949.Spearman, sp_949),
]:
    print(f'{m:<10s}  {f:>10.4f}  {mo:>10.4f}  {d:>10.4f}')

ablation_summary = {
    'function_id': 949,
    'hyperparameters': study.best_params,
    'fayam':         {'MASE': float(fayam_949.MASE), 'sMAPE': float(fayam_949.sMAPE),
                      'RMSE': float(fayam_949.RMSE), 'R2': float(fayam_949.R2),
                      'Spearman': float(fayam_949.Spearman)},
    'multi_opt':     {'MASE': float(opt_949.MASE), 'sMAPE': float(opt_949.sMAPE),
                      'RMSE': float(opt_949.RMSE), 'R2': float(opt_949.R2),
                      'Spearman': float(opt_949.Spearman)},
    'dedicated_949': {'MASE': mase_949, 'sMAPE': smape_949,
                      'RMSE': rmse_949, 'R2': r2_949, 'Spearman': sp_949},
}
with open(f'{DRIVE_BASE}/results/ablation_949.json', 'w') as f:
    json.dump(ablation_summary, f, indent=2)

delta = r2_949 - opt_949.R2
print(f'\\nDiagnostic : R² 949-dédié vs 949-multi-opt = {delta:+.4f}')
if delta > 0.10:
    print('  → La fonction 949 est ÉCRASÉE par le multi-task.')
    print('     Conclusion : un modèle dédié serait préférable pour cette fonction.')
elif delta < -0.10:
    print('  → Le multi-task AIDE la 949 (transfert positif).')
    print('     Conclusion : garder le multi-task est la bonne stratégie.')
else:
    print('  → Peu de différence : la 949 est intrinsèquement difficile, pas écrasée.')""")

code("""# Plot 949-dédié
fig, ax = plt.subplots(figsize=(14, 4))
n_ctx = 360
x_ctx = np.arange(-n_ctx, 0)
x_pred = np.arange(0, PREDICTION_LENGTH)
ax.plot(x_ctx,  np.array(test_949[0]['target'])[-(n_ctx + PREDICTION_LENGTH):-PREDICTION_LENGTH],
        color='steelblue', lw=1, label='Réel (contexte)')
ax.plot(x_pred, actual_949, color='steelblue', lw=1.5, ls='--', label='Réel (horizon)')
ax.plot(x_pred, pred_949,   color='darkgreen', lw=1.5, label='949-dédié')
ax.plot(x_pred, forecast_median_opt[0],
        color='tomato', lw=1.2, ls='-.', label='Multi-opt (référence)')
ax.axvline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel('Minutes (0 = début horizon)'); ax.set_ylabel('Invocations/min')
ax.set_title(f'Fonction 949 — comparaison Multi-opt vs Dédié  |  '
             f'R²(dédié)={r2_949:.3f}  R²(multi-opt)={opt_949.R2:.3f}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/ablation_949_plot.png', dpi=150); plt.show()""")

md("## 14 — Sauvegarde finale")

code("""final_summary = {
    'run_name': RUN_NAME, 'cluster': CLUSTER_ID, 'seed': SEED,
    'date': pd.Timestamp.now().isoformat(),
    'hpo': {
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned':    len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'best_params': study.best_params,
        'best_val_r2': float(study.best_value),
    },
    'final_training': {
        'epochs_trained': len(loss_history),
        'early_stopped': len(loss_history) < FINAL_MAX_EPOCHS,
        'best_val_r2': best_val_r2,
    },
    'test_metrics_optimized': df_metrics_opt[['MASE','sMAPE','RMSE','R2','Spearman']].mean().round(4).to_dict(),
    'test_metrics_fayam_reference': {
        'MASE': 1.2297, 'sMAPE': 0.2174, 'RMSE': 24.6507, 'R2': 0.3701, 'Spearman': 0.9195,
    },
    'ablation_949': ablation_summary,
}

with open(f'{DRIVE_BASE}/results/final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print('=' * 70)
print(f'Run complet — Cluster {CLUSTER_ID} (optimisé)')
print(f'Artefacts : {DRIVE_BASE}')
print('=' * 70)
print(json.dumps(final_summary, indent=2))""")

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.10',
        },
        'colab': {
            'provenance': [],
            'gpuType': 'T4',
        },
        'accelerator': 'GPU',
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Notebook généré : {NOTEBOOK_PATH}')
print(f'  Cellules : {len(cells)}')
print(f'  Taille   : {NOTEBOOK_PATH.stat().st_size / 1024:.1f} Ko')
