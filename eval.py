import numpy as np 
import graph_tool.all as gt

def compute_umap(d,nn = 20):
    from umap import UMAP 
    return UMAP(nn,metric="precomputed").fit_transform(d)