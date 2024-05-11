# import os, sys, gc

import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

import torch as t
from torch import nn, Tensor
import torch.nn.functional as F

# from datasets import load_dataset
# from transformer_lens import HookedTransformer

from sklearn.cluster import HDBSCAN
# from sklearn.linear_model import LogisticRegression

from jaxtyping import Float
from typing import Tuple
from copy import deepcopy
from itertools import product
import random

def normalize_cluster(
            # self,
            x: Float[Tensor, "batch d_hidden"],
            y: Float[Tensor, "batch d_hidden"],
            device="cuda",
            preserve_norm : bool = False,
    ) -> Tuple[Float[Tensor, "batch d_hidden"], Float[Tensor, "batch d_hidden"]]:
        # average the contrastive examples
        v = (x + y) / 2.
        # cluster
        hdb = HDBSCAN(min_cluster_size=5, metric="euclidean")
        hdb.fit(v)
        # normalize each cluster independently
        for label in set(hdb.labels_):
            ixs = np.where(hdb.labels_ == label)
            m_x, m_y = x[ixs].mean(dim=0, keepdim=True), y[ixs].mean(dim=0, keepdim=True)
            s_x, s_y = x[ixs].std(dim=0, keepdim=True), y[ixs].std(dim=0, keepdim=True)
            x[ixs] = ((x[ixs] - m_x) / s_x)
            y[ixs] = ((y[ixs] - m_y) / s_y)
        return x.to(device), y.to(device)