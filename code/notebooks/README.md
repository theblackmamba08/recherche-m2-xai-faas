# Notebooks

## EDA_clusters.ipynb — Analyse exploratoire des clusters FaaS

### Ce que fait ce notebook

Analyse complète des 4 clusters HDBSCAN retenus pour la baseline FAYAM (clusters 0, 4, 6, 8 — 19 fonctions, Azure Functions Trace 2019, 14 jours à la minute).

L'analyse est faite **par fonction** (19 séries individuelles) ET **par cluster** (agrégation). Elle couvre :

| Section | Contenu |
|---------|---------|
| 2 — Vue d'ensemble | Tableau récap : moyenne, std, CV, zéros (%), burstiness B=(σ-μ)/(σ+μ) |
| 3 — Stats descriptives | `describe()` par fonction |
| 4 — Séries temporelles | Série 14j + zoom 3j + heatmap 14×1440 + profil journalier moyen |
| 5 — Analyse des zéros | Taux de zéros, runs consécutifs max, distribution des plages silencieuses |
| 6 — Distributions | Histogrammes + KDE, boxplots, skewness/kurtosis |
| 7 — Périodicité | ACF (2880 lags = 48h) + FFT periodogramme + top-5 périodes dominantes |
| 8 — Stationnarité | Test ADF par fonction + ADF après diff(1) si non-stationnaire |
| 9 — Cohérence intra-cluster | Pearson + Spearman + distance euclidienne normalisée |
| 10 — Comparaison inter-cluster | CV × burstiness, profils journaliers normalisés overlay |
| 11 — Synthèse | Recommandations prétraitement + scatter CV vs burstiness |
| 12 — Sauvegarde | JSON métriques + REGISTER.md + téléchargement navigateur (Colab) |

### Données attendues

```
memoire/06-datasets/raw/
    cluster_0.csv   # 3 fonctions (942-944), ~100 000 invocations/min, aucun zéro
    cluster_4.csv   # 5 fonctions (949-954), ~50-200 invocations/min
    cluster_6.csv   # 5 fonctions (138-144), ~0-20 invocations/min, zero-inflaté
    cluster_8.csv   # 6 fonctions (964-977), ~0-50 invocations/min, bursty
```

Format : CSV avec `Function` en première colonne (ID de fonction) et les timesteps 1..20160 en colonnes.

### Dépendances

```
numpy, pandas, matplotlib, seaborn, scipy, statsmodels, scikit-learn
```

Installation locale :
```bash
cd code/
pip install -e .[dev]
# statsmodels n'est pas dans pyproject.toml — à ajouter ou installer manuellement :
pip install statsmodels
```

### Exécution locale

```bash
cd code/
jupyter notebook notebooks/EDA_clusters.ipynb
# Puis : Run All
```

La cellule 3 détecte automatiquement l'environnement local — aucune configuration Drive requise.  
Les résultats JSON sont sauvegardés dans `code/experiments/eda/`.

### Exécution sur Google Colab

1. Ouvrir Colab → Fichier → Ouvrir → Upload → `EDA_clusters.ipynb`
2. Uploader les 4 CSV dans `/content/` (ou adapter `DATA_DIR` vers Drive)
3. **Run All**
4. La cellule 3 demande l'autorisation de monter Google Drive — accepter
5. À la fin : téléchargement automatique du JSON + rappel pour exporter le HTML (Fichier → Télécharger → .html)

### Sorties produites

| Fichier | Emplacement | Contenu |
|---------|-------------|---------|
| `eda_results_YYYY-MM-DD_HH-MM.json` | `code/experiments/eda/` (local) ou Drive | Métriques clés : overview, zéros, FFT top périodes, ADF |
| `REGISTER.md` | `code/experiments/eda/` | Tableau cumulatif de tous les runs |
| `EDA_clusters_YYYY-MM-DD.html` | À archiver manuellement dans `code/experiments/eda/` | Notebook exécuté avec toutes les figures |

### Structure du JSON produit

```json
{
  "run_date": "2026-04-29_10-30",
  "clusters_analyzed": [0, 4, 6, 8],
  "n_functions": 19,
  "n_timesteps": 20160,
  "cluster_means": {"0": 121937.0, "4": 97.0, "6": 2.0, "8": 5.0},
  "overview": [ {"Cluster": 0, "Fonction": 942, "Moyenne": ..., "CV (%)": ..., ...}, ... ],
  "zeros": [ {"Cluster": 6, "Fonction": 138, "Zeros (%)": ..., ...}, ... ],
  "fft_top_periods": { "0": { "942": [{"period_min": 1440.0, "power_pct": ...}, ...] } },
  "adf": [ {"Cluster": 0, "Fonction": 942, "Stationnaire (p<0.05)": "Oui", ...}, ... ],
  "n_stationary": 19,
  "n_total_functions": 19
}
```

### Résultats attendus (profils connus)

| Cluster | Profil | Zéros | Stationnarité | Périodicité 24h |
|---------|--------|-------|---------------|-----------------|
| 0 | Charges très élevées (~100k-400k), signal propre | Aucun | Attendue | Forte |
| 4 | Charges moyennes (~50-200), peu de zéros | < 5% | Attendue | Modérée |
| 6 | Charges faibles (~0-20), zero-inflaté | Élevé | À vérifier | Faible |
| 8 | Charges faibles, bursty (pics sporadiques) | Modéré | À vérifier | Faible |

---

## Clustering_DBSCAN.ipynb

Notebook original reçu avec le code FAYAM. Contient le pipeline de clustering HDBSCAN
(malgré le nom DBSCAN — voir discordance documentée dans `src/baseline/fayam/BASELINE.md`).
**Ne pas modifier** — référence immuable.
