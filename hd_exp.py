import numpy as np 
import pylab as plt
import graph_tool.all as gt 

from sklearn.metrics import pairwise_distances
from modules.metrics import compute_metrics
from hd_data.load_data import load_data, sample_data
from modules.graph_io import plot_data as plot
from modules.graph_metrics import apsp

from modules.cython_l2g import L2G_opt, standard_mds
from modules.L2G import diffusion_weights, L2G
from sklearn.manifold import TSNE,MDS 
from umap import UMAP

from csv import writer

def append_row(fname: str,new_row: list):
    with open(fname, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_row)
        f_object.close()


def run_proj(data_name: str, K=[8,16,32,64,85,100,150,200,300,500], sample=500,save=False):

    data, y = load_data(data_name)
    print(f"Data shape is {data.shape}")
    if sample > 0 and sample < data.shape[0]:
        data, y = sample_data(data,y,n_samples=sample)
        print(f"Sampled down to {data.shape, y.shape}")

    data = (data - data.min(0)) / np.maximum(data.max(0) - data.min(0),1e-9)


    d = pairwise_distances(data)

    X = TSNE(perplexity=20,init="pca",learning_rate="auto").fit_transform(data)
    plot(X,y,output=f"hd_data/{data_name}/tsne.png",title=f"{data_name.upper()} t-SNE")
    metrics = compute_metrics(data,d,X,y)
    if save: append_row(f"hd_data/{data_name}/tsne_metrics.txt",metrics)

    X = MDS(dissimilarity="precomputed",n_init=1).fit_transform(d)
    plot(X,y,output=f"hd_data/{data_name}/mds.png",title=f"{data_name.upper()} MDS")
    metrics = compute_metrics(data,d,X,y)
    if save: append_row(f"hd_data/{data_name}/mds_metrics.txt",metrics)

    X = UMAP().fit_transform(data)
    plot(X,y,output=f"hd_data/{data_name}/umap.png",title=f"{data_name.upper()} UMAP")
    metrics = compute_metrics(data,d,X,y)
    if save: append_row(f"hd_data/{data_name}/umap_metrics.txt",metrics)

    G,dists = gt.generate_knn(data,k=7)
    dw = apsp(G)

    for k in K:
        w = diffusion_weights(d,k=k)
        X = L2G(G,k,alpha=0.1)
        # X = L2G_opt(d.astype("double"),w.astype(np.int16),alpha=0.1)

        plot(X,y,output=f"hd_data/{data_name}/l2g_{k}.png",title=f"{data_name.upper()} L2G({k})")
        metrics = compute_metrics(data,d,X,y)
        if save: append_row(f"hd_data/{data_name}/l2g_{k}_metrics.txt",metrics)

if __name__ == "__main__":
    datasets = [
        "har",
        "fashion-mnist",
        "imdb",
        "mnist",
        "coil20",
        "cifar10",
        "cnae9",
        "fmd",
        "spambase"
    ]

    for ds in datasets:
        for i in range(15):
            run_proj(ds,sample=1000,save=True)
    # run_proj("har",sample=-1)
