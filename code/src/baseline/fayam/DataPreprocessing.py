import os
import re
import json
import logging
import requests
from tqdm import tqdm
from pathlib import Path
import tarfile
from zipfile import ZipFile
import pandas as pd
import numpy as np
import random
from scipy.stats import describe, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.collections as mcoll
import matplotlib.colors
import seaborn as sns
import shutil
import csv
from joblib import Parallel, delayed
from pypdf import PdfWriter, PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
matplotlib.rc('text', usetex = True)
# plt.rc('axes', linewidth=2)
plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s[%(levelname)s|%(filename)s:%(lineno)s] %(message)s'))
    logging.getLogger().addHandler(handler)
except:
    logging.basicConfig(format='[%(levelname)s|%(filename)s:%(lineno)s] %(message)s')

class DataPreprocessing:

    def __init__(self, data_path):
        self._data_path = data_path
        self._init = False

    def init(self):
        assert os.path.exists(self._data_path), 'Check if the data conf file %s exists' % self._data_path
        with open(self._data_path) as f:
            data_conf = json.load(f)
        
        self._url = data_conf['url']
        self._download = data_conf['download']
        self._ext = data_conf['ext']
        self._decompress = data_conf['decompress']
        self._cols = data_conf['cols']
        self._primary_col = data_conf['primary_col']
        self._func_name_col = data_conf['func_name_col']
        self._time_unit = data_conf['time_unit']
        # self._exist_time_cols = data_conf['exist_time_cols']
        self._filters = data_conf['filters']
        self._file_path = data_conf['file_path']
        self._files = data_conf['files']
        self._is_similar_entry_files = data_conf['is_similar_entry_files']
        self._nb_days = data_conf['nb_days']
        self._mapping_funcs = data_conf['mapping_funcs']
        self._preprocessing = data_conf['preprocessing']
        self._n_jobs = data_conf['n_jobs']
        self._min_samples = data_conf['min_samples']
        self._min_cluster_size = data_conf['min_cluster_size']
        self._perplexity = data_conf['perplexity']
        self._plot_cluster = data_conf['plot_cluster']
        self._seed = data_conf['seed']
        self._nb_max_ifp = data_conf['nb_max_ifp']

        self._init = True


    def process(self):
        assert self._init, 'Please call init() before process :)'
        # 1. Downloading dataset
        if self._download:
            self._downloading()
        # 2. Sorting dataset according to the day
        parent_data_path = Path(self._data_path).parent.absolute()
        raw_dir = os.path.join(parent_data_path, self._file_path)
        matching_files = []
        for filename in os.listdir(raw_dir):
            if re.search(self._files, filename):
                matching_files.append(os.path.join(raw_dir, filename))
        matching_files = sorted(matching_files)
        if not self._nb_days == -1:
            assert self._nb_days <= len(matching_files), "Too many days!!! It must be less or equal to the number of files"
            matching_files = matching_files[:self._nb_days]
        # 3. Loading and preprocessing dataset (filtering, removing useless columns, etc.)
        progress = tqdm(matching_files)
        data_frames = []
        func_names = []
        for matching_file in progress:
            progress.set_description(f"Preprocessing of {Path(matching_file).name}")
            df = pd.DataFrame(pd.read_csv(matching_file))
            for filter in self._filters:
                if filter['enable']:
                    df = df[df[filter['col']] == filter['name']]
            drop_cols = [x for x in self._cols if not x == self._primary_col]
            df = df.drop(columns=drop_cols)
            if not len(self._preprocessing) == 0:
                for key,value in self._preprocessing.items():
                    if key == 'transpose' and value['enable']:
                        df = df.drop(columns=[value['col']])
                        df = df.T
                        new_cols = [str(1+int(x)) for x in df.columns.to_list()]
                        df.columns = new_cols
                        df.insert(loc=0,column=self._primary_col,value=df.index)
            df = df.fillna(0)
            if self._time_unit in ['sec']:
                pass
            if not len(self._func_name_col) == 0:
                func_names = func_names + list(df[self._func_name_col])
            else:
                func_names = func_names + list(df[self._primary_col])
            data_frames.append(df)
        nb_cols = len(data_frames[0].columns.to_list()) - 1
        func_names = set(func_names)
        if not self._is_similar_entry_files:
            progress = tqdm(list(range(self._nb_days)))
            for day in progress:
                progress.set_description(f"Normalizing dataset of day {day}")
                df = data_frames[day]
                remain_func_names = func_names - set(df[self._primary_col])
                if not len(remain_func_names) == 0:
                    rows_to_insert = [[str(x)]+list(range(1,nb_cols+1)) for x in remain_func_names]                    
                    df = pd.concat([df, pd.DataFrame(rows_to_insert, columns=df.columns.to_list())], ignore_index=True)
                data_frames[day] = df.sort_values(by=self._primary_col)
        # 4. Merging dataset
        df = data_frames[0]
        progress = tqdm(range(1,self._nb_days))
        for day in progress:
            progress.set_description(f"Merging day 0 with day {day}")
            data_frames[day] = data_frames[day].drop(columns=[self._primary_col])
            new_cols = [str(day*nb_cols+int(x)) for x in data_frames[day].columns.to_list()]
            data_frames[day].columns = new_cols
            df = pd.concat([df, data_frames[day]], axis=1)
        if self._mapping_funcs:
            func_names = list(df[self._primary_col])
            # func_names = list(func_names)
            func_new_names_dict = {self._primary_col: list(range(len(func_names)))}
            df = df.reset_index()
            df.pop('index')
            df.update(pd.DataFrame(func_new_names_dict))
            df = df.rename(columns={self._primary_col: "Function"})
            mapping_df = pd.DataFrame({"Function": list(range(len(func_names))), self._primary_col: func_names})
            mapping_df.to_csv(os.path.join(parent_data_path, 'mapping_funcs.csv'), index=False)
        else:
            df = df.rename(columns={self._primary_col: "Function"})
        # 5. Clustering dataset
        # Supprimer la colonne contenant les noms des fonctions
        # df = df.head(2000)
        functions = df.pop('Function')
        # Normalizing dataset
        print('Normalizing dataset...')
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        distance_matrix = pairwise_distances(scaled_data, metric='euclidean', n_jobs=self._n_jobs)
        # Applying clustering with hdbscan
        print('Applying clustering with hdbscan')
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=self._min_cluster_size, min_samples=self._min_samples, n_jobs=self._n_jobs)
        clusters = clusterer.fit_predict(distance_matrix)
        # Analyzing stats of each cluster
        progress = tqdm(np.unique(clusters))
        clusters_path = os.path.join(parent_data_path, 'clusters')
        if os.path.exists(clusters_path):
            shutil.rmtree(clusters_path) 
        os.makedirs(clusters_path, exist_ok=True)
        self._parallel_process_clusters(clusters, df, functions, clusters_path, self._process_single_cluster, self._analyze_cluster, self._plot_cluster['stats'], self._n_jobs)
        if self._plot_cluster['visualize']:
            # TSNE for visualization
            # perplexity = range(500, 1005, 200)
            divergence = []
            progress = tqdm(self._perplexity)
            # progress = tqdm([])
            for i in progress:
                progress.set_description(f"Visualizing clusters for perplexity {i}")
                tsne = TSNE(n_components=2, metric="precomputed", perplexity=i, random_state=42, init='random', n_jobs=self._n_jobs)
                embedding = tsne.fit_transform(distance_matrix)
                divergence.append(tsne.kl_divergence_)
                plt.figure(figsize=(10, 7))
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='plasma')
                # plt.title(f"Perplexity {i}")
                plt.colorbar(scatter, label="Cluster")
                plt.xticks([])
                plt.yticks([])
                for ext in ['.png','.pdf']:
                    plt.savefig(os.path.join(clusters_path, f"tsne.{i}"+ext))
                plt.close()
            plt.plot(self._perplexity, divergence, label="Divergence")
            # plt.legend()
            plt.tight_layout()
            for ext in ['.png','.pdf']:
                plt.savefig(os.path.join(clusters_path, f"tsne.divergence"+ext))
            plt.close()
        # print(list(cluster_data.index))
        # print(cluster_data.index[0])
        # print(cluster_data.loc[cluster_data.index[0]])
        # Plot invocation frequency per cluster
        def create_pdf(output_path, text):
            width, height = A4
            c = canvas.Canvas(output_path, pagesize=A4)
            font_size = 20
            c.setFont("Helvetica", font_size)
            text_width = c.stringWidth(text, "Helvetica", font_size)
            x = (width - text_width) / 2
            y = height / 2
            c.drawString(x, y, text)
            c.showPage()
            c.save()
        def plot_invocation_frequency(df, cluster_label, funcs, num, save_dir):
            time_label = 'time'
            nb_invoc_label = 'number of invocations'
            delay = 'minute'
            nb_cols = 6
            nb_rows = 3
            fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(18,8), constrained_layout = True, sharex=False, sharey=False)
            for func_number,func_id in enumerate(funcs):
                y = (int(func_number)) % 6
                x = (int(func_number)) // 6
                pldf = df[[time_label, func_id]]
                pldf = pldf.rename(columns={func_id : nb_invoc_label})
                plot = sns.lineplot(data=pldf[nb_invoc_label], ax=axes[x][y])
                if y == 0:
                    axes[x][y].set_ylabel(r'\textbf{'+nb_invoc_label.capitalize()+'}', family='monospace', fontsize=14)
                    # axes[x][y].set_ylabel(nb_invoc_label.capitalize(), family='monospace', fontsize=14)
                else:
                    plot.set(ylabel=None)
                axes[x][y].set_xlabel(r'\textbf{'+'Invocation frequency in ' + delay+'}',family='monospace', fontsize=12)
                # axes[x][y].set_xlabel('Invocation frequency in ' + delay,family='monospace', fontsize=12)
                plot.set_title(r'\texttt{'+'Function ' + str(func_id)+'}', family='monospace', fontsize=12)
                # plot.set_title('Function ' + str(func_id), family='monospace', fontsize=12)
                max_val = pldf[nb_invoc_label].max()
                min_val = 0 #df[nb_invoc_label].min()
                max = max_val
                med0 = round((min_val + max_val) / 2.0, 1)
                med1 = round((min_val + med0) / 2.0, 1)
                med2 = round((med0 + max_val) / 2.0, 1)
                axes[x][y].set_yticks([int(x) for x in [min_val, med2, med0, med1, max_val]])
                if 3 * 6 == func_number + 1:
                    break
                
            fig_name = f'cluster_{cluster_label}_invocation_frequency.{num}'
            fig_path = os.path.join(save_dir, fig_name)
            for ext in ['.png','.pdf']:
                fig.savefig(fig_path+ext, bbox_inches='tight')#
            plt.close()
        
        output = PdfWriter()
        progress = tqdm(np.unique(clusters))
        for cluster_label in progress:
            progress.set_description(f"Plot invocation frequency of cluster {cluster_label}")
            time_label = 'time'
            df = pd.read_csv(os.path.join(clusters_path, f'cluster_{cluster_label}.csv'))
            functions = df['Function']
            df = df.set_index('Function').T
            # df = df.rename(columns={'Function': time_label})
            # df = df.rename(columns={"Function": "time"})
            df = df.rename_axis(time_label).reset_index()
            #Remove the index name
            df = df.rename_axis(None, axis = 1)
            # print(df)
            # print(df.columns.to_list())
            # print(df.index)
            # break
            random.seed(self._seed)
            obs = list()
            tmp = set(functions)
            i = 0
            while len(tmp) >= 18 and i < self._nb_max_ifp:
                r = set(random.sample(list(tmp), 18))
                tmp -= r
                obs.append(list(r))
                i = i + 1
            if len(obs) < self._nb_max_ifp:
                obs.append(list(tmp))
            
            if len(obs) == self._nb_max_ifp and len(obs[self._nb_max_ifp - 1]) == 18 and self._nb_max_ifp * 18 < len(functions):
                txt = f'Some functions of cluster {cluster_label}'
            else:
                txt = f'Functions of cluster {cluster_label}'
            create_pdf(os.path.join(clusters_path,'title.pdf'), txt)
            input = PdfReader(os.path.join(clusters_path,'title.pdf'))
            output.add_page(input.get_page(0))
            for num, funcs in enumerate(obs):
                plot_invocation_frequency(df, cluster_label, funcs, num, clusters_path)
                input = PdfReader(os.path.join(clusters_path,f'cluster_{cluster_label}_invocation_frequency.{num}.pdf'))
                output.add_page(input.get_page(0))
        output.write(os.path.join(clusters_path, f'clusters_invocation_frequency.pdf'))        
        os.remove(os.path.join(clusters_path,'title.pdf'))
        

        # Utilisation
        # output = PdfWriter(PdfReader(os.path.join(clusters_path, figName)))
        # figName = f'cluster_0_invocation_frequency.pdf'
        # input1 = PdfReader(os.path.join(clusters_path, figName))
        # # add page 1 from input1 to output document, unchanged
        # output.add_page(input1.get_page(0))

        # # add page 2 from input1, but rotated clockwise 90 degrees
        # # output.add_page(input1.get_page(0).rotate(90))
        # # finally, write "output" to document-output.pdf
        # figName = f'clusters_invocation_frequency.pdf'
        # # outputStream = os.path.join(clusters_path, figName)
        # output.write(os.path.join(clusters_path, figName))
        # outputStream.close()

    def _parallel_process_clusters(self, clusters, df, functions, clusters_path, process_single_cluster_func, analyze_cluster_func, plot_stats=True, n_jobs=-1):
        progress = tqdm(np.unique(clusters), desc="Analysing and saving the cluster")
        func = lambda cid: process_single_cluster_func(cid, clusters, df, functions, clusters_path, analyze_cluster_func, plot_stats)
        results = Parallel(n_jobs=n_jobs)(delayed(func)(cluster_id) for cluster_id in progress)
        if plot_stats:
            stats_file = os.path.join(clusters_path, 'clusters.stats.csv')
            with open(stats_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["cluster_id", "n_functions", "n_observations", "min", "max", "mean", "variance", "std", "skewness", "kurtosis"])
            for result in results:
                with open(stats_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)

    def _process_single_cluster(self, cluster_id, clusters, df, functions, clusters_path, analyze_cluster_func, plot_stats):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = df.iloc[cluster_indices]

        stats_row = None
        if plot_stats:
            summary_stats = analyze_cluster_func(cluster_data, cluster_id, clusters_path)
            stats_row = [cluster_id, len(cluster_indices), summary_stats['nobs'],
                        summary_stats['min'], summary_stats['max'], summary_stats['mean'],
                        summary_stats['variance'], summary_stats['std'], summary_stats['skewness'],
                        summary_stats['kurtosis']]
        cluster_functions = functions.iloc[cluster_indices]
        cluster_df = pd.concat([cluster_functions, cluster_data], axis=1)
        cluster_csv_path = os.path.join(clusters_path, f"cluster_{cluster_id}.csv")
        cluster_df.to_csv(cluster_csv_path, index=False)
        return stats_row

    def _analyze_cluster(self, cluster_data, cluster_label, save_dir="cluster_analysis"):
        label = f"cluster_{cluster_label}" if cluster_label is not None else "cluster"

        flat_values = cluster_data.to_numpy().flatten()
        stats = describe(flat_values)

        mean_profile = cluster_data.mean(axis=0)
        std_profile = cluster_data.std(axis=0)
        var_profile = cluster_data.var(axis=0)
        amp_profile = cluster_data.max(axis=0) - cluster_data.min(axis=0)
        # threshold = 1
        # if stats.variance > threshold:
        #     skew_profile = skew(cluster_data, axis=0, bias=False)
        #     kurtosis_profile = kurtosis(cluster_data, axis=0, bias=False)

        def plot_and_save(y, title, ylabel, filename, threshold=None):
            plt.figure(figsize=(10, 5))
            plt.plot(np.asarray(y, float), label=ylabel)
            if threshold is not None:
                plt.axhline(threshold, color='red', linestyle='--', label=f"Seuil {threshold}")
            plt.title(title)
            plt.xlabel("Minute")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            for ext in ['.png','.pdf']:
                plt.savefig(os.path.join(save_dir, filename+ext))
            plt.close()

        plot_and_save(mean_profile, f"Profil moyen d'appels - {label}", "Moyenne", f"{label}_mean_profile")
        plot_and_save(std_profile, f"Écart-type par minute - {label}", "Écart-type", f"{label}_std_profile")
        plot_and_save(var_profile, f"Variance par minute - {label}", "Variance", f"{label}_variance_profile", threshold=2)
        plot_and_save(amp_profile, f"Amplitude par minute - {label}", "Amplitude", f"{label}_amplitude_profile")
        # if stats.variance > threshold:
        #     plot_and_save(skew_profile, f"Skewness par minute - {label}", "Skewness", f"{label}_skewness_profile")
        #     plot_and_save(kurtosis_profile, f"Kurtosis par minute - {label}", "Kurtosis", f"{label}_kurtosis_profile")

        plt.figure(figsize=(10, 5))
        plt.plot(np.asarray(mean_profile, float), label="Profil moyen", color='blue')
        plt.fill_between(range(cluster_data.shape[1]), mean_profile - std_profile, mean_profile + std_profile, alpha=0.3, label="±1 écart-type", color='skyblue')
        plt.title(f"Profil moyen d'appels avec variabilité - Cluster {label if label is not None else ''}")
        plt.xlabel("Minute")
        plt.ylabel("Moyenne + Variabilité")
        plt.legend()
        plt.tight_layout()
        for ext in ['.png','.pdf']:
            plt.savefig(os.path.join(save_dir, f"{label}_mean_var_profile"+ext))
        plt.close()

        summary = {
            "nobs": stats.nobs,
            "min": stats.minmax[0],
            "max": stats.minmax[1],
            "mean": stats.mean,
            "variance": stats.variance,
            "std": stats.variance**0.5,
            "skewness": stats.skewness,
            "kurtosis": stats.kurtosis
        }
        return summary

    def _downloading(self):
        response = requests.get(self._url, stream=True)
        data_path = Path(self._data_path)
        parent_data_path = data_path.parent.absolute()
        filename = os.path.join(parent_data_path, 'raw', 'raw'+self._ext)
        os.makedirs(os.path.join(parent_data_path, 'raw'), exist_ok=True)
        if response.status_code == 200:
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, 'wb') as handle:
                progress = tqdm(response.iter_content(chunk_size=1024), unit='kB')
                for data in progress:
                    progress.set_description(f'Downloading dataset')
                    handle.write(data)
            if self._decompress:
                if self._ext in ['.gz','.tar.gz','.xz','.tar','.tar.xz']:
                    # https://thepythoncode.com/code/compress-decompress-files-tarfile-python
                    tar = tarfile.open(filename)
                    members = tar.getmembers()
                    progress = tqdm(members)
                    for member in progress:
                        tar.extract(member, path=os.path.join(parent_data_path, 'raw'))
                        progress.set_description(f"Extracting {member.name}")
                    os.remove(filename)
                elif self._ext in ['.zip']:
                    zipf = ZipFile(filename)
                    members = zipf.namelist()
                    progress = tqdm(members)
                    for member in progress:
                        progress.set_description(f"Extracting {member}")
                        zipf.extract(member, path=os.path.join(parent_data_path, 'raw'))
                    os.remove(filename)
                else:
                    logging.error('This extension is not yet supported')                    
        else:
            logging.error('This extension is not yet supported')                    