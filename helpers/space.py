import math
import random
from functools import reduce
from itertools import combinations, product
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from ipywidgets import fixed, interact, interact_manual, interactive
from matplotlib.colors import Colormap
from notebook_helper.altair_theme import mysoc_palette_colors
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from helpers import viz


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + (0,)


def fnormalize(s):
    return (s - s.mean())/s.std()


class mySocMap(Colormap):

    def __call__(self, X, alpha=None, bytes=False):
        return mysoc_palette_colors[int(X)]


class Cluster:
    """
    Helper class for finding kgram clusters.
    """

    def __init__(self,
                 source_df: pd.DataFrame,
                 id_col: Optional[str] = None,
                 cols: Optional[List[str]] = None,
                 label_cols: Optional[List[str]] = None,
                 normalize: bool = True,
                 k: Optional[int] = None):
        """
            Initalised with a dataframe, an id column in that dataframe,
            the columns that are the dimensions in question.
            A 'normalize' paramater on if those columns should be normalised before use. 
            and 'label_cols' which are columns that contain categories for items.
            These can be used to help understand clusters. 

        """
        self.default_seed = 1221
        self.cluster_results = {}
        self.label_names = {}
        self.label_descs = {}
        self.normalize = normalize
        self.source_df = source_df
        df = source_df.copy()
        self.k = k
        if id_col:
            df = df.set_index(id_col)
        if cols:
            df = df[cols]
        if normalize:
            df = df.apply(fnormalize, axis=0)
        self.df = df
        self.cols = cols
        self.id_col = id_col
        self.label_cols = label_cols
        label_df = source_df.copy()
        if id_col:
            label_df = label_df.set_index(id_col)
        if not label_cols:
            label_cols = []
        label_df = label_df.drop(
            columns=[x for x in label_df.columns if x not in cols + label_cols])
        for c in cols:
            try:
                labels = ["Low", "Medium", "High"]
                q = pd.qcut(label_df[c], 3, labels=labels, duplicates="drop")
                label_df[c] = q
            except ValueError:
                pass

        label_df["Total"] = "Total"

        self.label_df = label_df

    def set_k(self, k: int):
        self.k = k

    def get_label_name(self, k, n, include_short=True):
        if k is None:
            k = self.k
        short_label = n + 1
        name = self.label_names.get(k, {}).get(n, short_label)
        if include_short:
            if name != short_label:
                name = f"{short_label}: {name}"
        return name

    def get_label_desc(self, k, n):
        if k is None:
            k = self.k
        short_label = n + 1
        name = self.label_descs.get(k, {}).get(n, short_label)
        return name

    def get_label_options(self, k: Optional[int] = None):
        if k is None:
            k = self.k
        return [self.get_label_name(k, x) for x in range(0, k)]

    def get_cluster_labels(self, k: Optional[int] = None, include_short=True):
        if k is None:
            k = self.k
        labels = pd.Series(self.get_clusters(k).labels_)
        labels = labels.apply(lambda x: self.get_label_name(
            k=k, n=x, include_short=include_short))
        return np.array(labels)

    def get_cluster_descs(self, k: Optional[int] = None):
        if k is None:
            k = self.k
        labels = pd.Series(self.get_clusters(k).labels_)
        labels = labels.apply(lambda x: self.get_label_desc(k=k, n=x))
        return np.array(labels)

    def assign_name(self, n: int, name: str, desc: Optional[str] = "", k: Optional[int] = None):
        if k is None:
            k = self.k
        if k not in self.label_names:
            self.label_names[k] = {}
            self.label_descs[k] = {}
        self.label_names[k][n-1] = name
        self.label_descs[k][n-1] = desc

    def plot(self,
             k: Optional[int] = None,
             limit_columns: Optional[List[str]] = None,
             only_one: Optional[Any] = None,
             show_legend: bool = True):
        """
        Plot either all possible x, y graphs for k clusters
        or just the subset with the named x_var and y_var.
        """
        if k is None:
            k = self.k
        df = self.df

        num_rows = 3

        vars = self.cols
        if limit_columns:
            vars = [x for x in vars if x in limit_columns]
        combos = list(combinations(vars, 2))
        rows = math.ceil(len(combos)/num_rows)

        plt.rcParams["figure.figsize"] = (15, 5*rows)

        df["labels"] = self.get_cluster_labels(k)
        if only_one:
            df["labels"] = df["labels"] == only_one
            df["labels"] = df["labels"].map(
                {True: only_one, False: "Other clusters"})
        chart_no = 0

        rgb_values = sns.color_palette("Set2", len(df["labels"].unique()))
        color_map = dict(zip(df["labels"].unique(), rgb_values))
        fig = plt.figure()

        for x_var, y_var in combos:
            chart_no += 1
            ax = fig.add_subplot(rows, num_rows, chart_no)
            for c, d in df.groupby("labels"):
                scatter = ax.scatter(d[x_var], d[y_var],
                                     color=color_map[c], label=c)

            ax.set_xlabel(self._axis_label(x_var))
            ax.set_ylabel(self._axis_label(y_var))
            if show_legend:
                ax.legend()

        plt.show()

    def plot_tool(self, k: Optional[int] = None):
        if k == None:
            k = self.k

        def func(cluster, show_legend, **kwargs):
            if cluster == "All":
                cluster = None
            limit_columns = [x for x, y in kwargs.items() if y is True]
            self.plot(k=k, only_one=cluster, show_legend=show_legend,
                      limit_columns=limit_columns)

        cluster_options = ["All"] + self.get_label_options(k)

        analysis_options = {x: True if n <
                            2 else False for n, x in enumerate(self.cols)}

        tool = interactive(func, cluster=cluster_options, **
                           analysis_options, show_legend=False,)
        display(tool)

    def _get_clusters(self, k):
        """
        fetch k means results for this cluster
        """
        return KMeans(n_clusters=k, random_state=self.default_seed).fit(self.df)

    def get_clusters(self, k: None):
        """
        fetch from cache if already run for this value of k
        """
        if k is None:
            k = self.k
        if k not in self.cluster_results:
            self.cluster_results[k] = self._get_clusters(k)
        return self.cluster_results[k]

    def find_k(self, start: int = 15, stop: Optional[int] = None, step: int = 1):
        """
        Graph the elbow and Silhoette method for finding the optimal k.
        Parameters are the search space 
        """
        if start and not stop:
            stop = start
            start = 2

        def s_score(kmeans):
            return silhouette_score(self.df,
                                    kmeans.labels_,
                                    metric='euclidean')

        df = pd.DataFrame({"n": range(start, stop, step)})
        df["k_means"] = df["n"].apply(self.get_clusters)
        df["sum_squares"] = df["k_means"].apply(lambda x: x.inertia_)
        df["silhouette"] = df["k_means"].apply(s_score)

        plt.rcParams["figure.figsize"] = (10, 5)
        plt.subplot(1, 2, 1)
        plt.plot(df["n"], df["sum_squares"], 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')

        plt.subplot(1, 2, 2)
        plt.plot(df["n"], df["silhouette"], 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette Method For Optimal k')
        plt.show()

    def stats(self, k: Optional[int] = None, label_lookup: Optional[dict] = None):
        """
        Simple description of sample size
        """
        if k is None:
            k = self.k
        if label_lookup is None:
            label_lookup = {}

        df = pd.DataFrame({"labels": self.get_cluster_labels(k)})
        df.index = self.df.index
        df = df.reset_index()
        pt = df.pivot_table(self.id_col,
                            index="labels", aggfunc="count")
        pt = pt.rename(columns={self.id_col: "count"})
        pt["%"] = (pt["count"] / len(df)).round(3) * 100

        def random_set(s):
            l = [label_lookup.get(x, x) for x in s]
            random.shuffle(l)
            return l[:5]

        d = df.groupby("labels").agg(random_set)
        pt = pt.join(d)
        pt = pt.rename(columns={self.id_col: "random members"})
        return pt

    def raincloud(self, column: str, k: Optional[int] = None, one_value: Optional[str] = None, groups: Optional[str] = "Cluster", use_source: bool = True):
        if k is None:
            k = self.k
        if use_source:
            df = self.source_df.copy()
        else:
            df = self.df
        df["Cluster"] = self.get_cluster_labels(k)
        df.viz.raincloud(values=column, groups=groups, one_value=one_value,
                         title=f"Raincloud plot for {column} variable.")

    def raincloud_tool(self, k: Optional[int] = None):
        if k is None:
            k = self.k

        def func(variable, comparison, use_source_values):
            groups = "Cluster"
            if comparison == "all":
                comparison = None
            if comparison == "none":
                groups = None
                comparison = None
            self.raincloud(variable, k, one_value=comparison,
                           groups=groups, use_source=use_source_values)

        tool = interactive(func,
                           variable=self.cols,
                           use_source_values=True,
                           comparison=["all", "none"] + self.get_label_options(k))

        display(tool)

    def label_tool(self, k: Optional[int] = None):
        """
        tool to review how labels assigned for each cluster

        """
        if k is None:
            k = self.k

        def func(cluster, sort, include_data_labels):
            if sort == "Index":
                sort = None
            df = self.label_review(label=cluster,
                                   sort=sort,
                                   include_data=include_data_labels)
            display(df)
            return df

        sort_options = ["Index", "% of cluster", "% of label"]
        tool = interactive(func,
                           cluster=self.get_label_options(k),
                           sort=sort_options,
                           include_data_labels=True)
        display(tool)

    def label_review(self,
                     k: Optional[int] = None,
                     label: Optional[int] = 1,
                     sort: Optional[str] = None,
                     include_data: bool = True):
        """
        Review labeled data for a cluster
        """
        if k is None:
            k = self.k

        def to_count_pivot(df):
            mdf = df.drop(columns=["label"]).melt()
            mdf["Count"] = mdf["variable"] + mdf["value"]
            return mdf.pivot_table("Count", index=["variable", "value"], aggfunc="count")

        df = self.label_df
        if include_data is False:
            df = df[[x for x in df.columns if x not in self.cols]]
        df["label"] = self.get_cluster_labels(k)
        opt = to_count_pivot(df).rename(columns={"Count": "overall_count"})
        df = df.loc[df["label"] == label]
        pt = to_count_pivot(df).join(opt)
        pt = pt.rename(columns={"Count": "cluster_count"})
        pt["% of cluster"] = (pt["cluster_count"] / len(df)).round(3) * 100
        pt["% of label"] = (pt["cluster_count"] /
                            pt["overall_count"]).round(3) * 100
        if sort:
            pt = pt.sort_values(sort, ascending=False)
        return pt

    def _axis_label(self, label_txt):
        """
        Extend axis label with extra notes        
        """
        txt = label_txt
        if self.normalize:
            txt = txt + " (normalized)"
        return txt

    def df_with_labels(self, k: Optional[int] = None):
        """
        return the original df but with a label column attached
        """
        if k is None:
            k = self.k
        df = self.source_df.copy()
        df["label"] = self.get_cluster_labels(k, include_short=False)
        df["label_desc"] = self.get_cluster_descs(k)
        return df

    def plot3d(self,
               k: Optional[int] = None,
               x_var: Optional[str] = None,
               y_var: Optional[str] = None,
               z_var: Optional[str] = None):
        if k is None:
            k = self.k
        """
        Plot either all possible x, y, z graphs for k clusters
        or just the subset with the named x_var and y_var.
        """
        df = self.df

        labels = self.get_cluster_labels(k)
        combos = list(combinations(df.columns, 3))
        if x_var:
            combos = [x for x in combos if x[0] == x_var]
        if y_var:
            combos = [x for x in combos if x[1] == y_var]
        if z_var:
            combos = [x for x in combos if x[1] == y_var]
        rows = math.ceil(len(combos)/2)

        plt.rcParams["figure.figsize"] = (20, 10*rows)

        chart_no = 0
        fig = plt.figure()
        for x_var, y_var, z_var in combos:
            chart_no += 1
            ax = fig.add_subplot(rows, 2, chart_no, projection="3d")
            ax.scatter(df[x_var], df[y_var], df[z_var], c=labels)
            ax.set_xlabel(self._axis_label(x_var))
            ax.set_ylabel(self._axis_label(y_var))
            ax.set_zlabel(self._axis_label(z_var))
            plt.title(f'Data with {k} clusters')

        plt.show()


def join_distance(df_label_dict: Dict[pd.DataFrame, str]):
    """
    get multiple authority distances into the same dataframe
    """

    def prepare(df, label):
        return (df
                .set_index(list(df.columns[:2]))
                .rename(columns={"distance": label}))

    to_join = [prepare(df, label) for label, df in df_label_dict.items()]
    df = reduce(lambda x, y:  x.join(y), to_join)
    df = df.reset_index()
    return df


@pd.api.extensions.register_dataframe_accessor("space")
class SpacePDAccessor(object):
    """
    extention to pandas dataframe
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def self_distance(self, id_col: str, cols: Optional[List] = None, normalize: bool = False):
        """
        Calculate the distance between all objects in a dataframe in an n-dimensional space.
        get back a dataframe with two labelled columns as well as the distance.
        id_col : unique column containing an ID or similar
        cols: all columns to be used in the calculation of distance
        normalize: should these columns be normalised before calculating distance

        """
        source_df = self._obj

        if id_col not in source_df.columns:
            source_df = source_df.reset_index()

        if cols is None:
            cols = [x for x in source_df.columns if x != id_col]

        a_col = id_col + "_A"
        b_col = id_col + "_B"
        df = pd.DataFrame(
            list(product(source_df[id_col], source_df[id_col])), columns=[a_col, b_col])

        grid = source_df[cols]
        # normalise columns
        if normalize:
            grid = grid.apply(fnormalize, axis=0)
        distance = pdist(grid.to_numpy())
        # back into square grid, flatten to 1d
        df["distance"] = squareform(distance).flatten()
        df = df.loc[~(df[a_col] == df[b_col])]
        return df


@pd.api.extensions.register_dataframe_accessor("joint_space")
class JointSpacePDAccessor(object):
    """
    handles dataframes that have joined several distance calculations

    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def same_nearest_k(self, k: int = 5):
        """
        Expects the dataframe returned by `join_distance`.
        Groups by column 1, Expects first two columns to be id columns.
        Beyond that, will see if all columns (representing distances)
        have the same items
        in their lowest 'k' matches. 
        Returns a column that can be averaged to get the overlap between
        two metrics.
        """
        df = self._obj

        def top_k(df, k=5):
            df = (df
                  .set_index(list(df.columns[:2]))
                  .rank())
            df = df <= k
            same_rank = df.sum(axis=1).reset_index(
                drop=True) == len(list(df.columns))
            return pd.DataFrame([[same_rank.sum() / k]], columns=[f"same_top_{k}"])

        return (df
                .groupby(df.columns[0])
                .apply(top_k, k=k)
                .reset_index()
                .drop(columns="level_1"))

    def agreement(self, ks: List[int] = [1, 2, 3, 5, 10, 25]):
        """
        Given the result of 'join_distance' explore how similar 
        items fall in 'top_k' for a range of values of k.
        """

        df = self._obj

        def get_average(k):
            return (df
                    .joint_space.same_nearest_k(k=k)
                    .mean()
                    .round(2)[0])

        r = pd.DataFrame({"top_k": ks})
        r["agreement"] = r["top_k"].apply(get_average)
        return r

    def plot(self, sample=None, kind="scatter", title="", **kwargs):
        """
        simple plot of distance
        """
        df = self._obj
        if sample:
            df = df.sample(sample)
        plt.rcParams["figure.figsize"] = (10, 5)
        df.plot(x=df.columns[2], y=df.columns[3],
                kind=kind, title=title, **kwargs)
