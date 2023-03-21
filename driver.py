import numpy as np
import graph_tool.all as gt 
from modules.L2G import find_neighbors
from modules.cython_l2g import L2G_opt
from modules.graph_metrics import apsp
from modules.thesne import tsnet 
from modules.graph_io import draw_tsnet_like as draw


from modules.graph_metrics import get_stress, get_neighborhood, compute_graph_cluster_metrics, get_cluster_ids


def embed_tsnet(G,output=None,dr=True):
    d = apsp(G)
    X = tsnet(d)
    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,get_cluster_ids(G)[0]),get_stress(X,d)

def embed_umap(G,output=None,dr=True):
    from umap import UMAP 
    d = apsp(G)
    X = UMAP(metric="precomputed",n_neighbors=10).fit_transform(d)
    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,get_cluster_ids(G)[0]), get_stress(X,d)

def embed_mds(G,output=None,dr=True):
    import s_gd2 
    E = np.array([(u,v) for u,v in G.iter_edges()],dtype=np.int32)
    I,J = E[:,0],E[:,1]
    X = s_gd2.layout(I,J)

    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,get_cluster_ids(G)[0]), get_stress(X,apsp(G)),


def gen_l2g_spectrum(G,K=[10,35,72,100,1000]):
    d = apsp(G)
    print(d.dtype)
    c_ids,_ = get_cluster_ids(G)
    s,n,c = list(), list(), list()
    for k in K:
        k = int(k) if k < G.num_vertices() else G.num_vertices() - 1
        print(k)
        w = find_neighbors(G,k=k,a=8)
        X = L2G_opt(d,w)
        s.append(get_stress(X,d))
        n.append(1-get_neighborhood(G,X))
        c.append(compute_graph_cluster_metrics(G,X,c_ids))

        draw(G,X,f"outs/out_k{k}.png")
    print(f"S : {s}\n NE: {n}\n CD: {c}")
    return n,c,s


def sample_k(max):

    accept = False

    while not accept:

        k = np.random.randint(1,max+1)

        accept = np.random.random() < 1.0/k

    return k

def get_graph(n=200):
    return gt.random_graph(n, lambda: sample_k(40), model="probabilistic-configuration",

                    edge_probs=lambda i, k: 1.0 / (1 + abs(i - k)), directed=False,

                    n_iter=100)

def measure_time(repeat=5):
    import time
    sizes = list(range(200,4005,200))
    print(len(sizes))
    times = np.zeros(len(sizes))

    for i,n in enumerate(sizes):
        print(f"On the {n}th size")
        for _ in range(repeat):
            G = get_graph(n)
            start = time.perf_counter() 
            d = apsp(G)
            w = find_neighbors(G,a=10,k=35)
            X = L2G_opt(d,w,200)
            end = time.perf_counter()

            val = end-start
            times[i] += val

        times[i] /= repeat
        np.savetxt("data/03_13_new_timeexp.txt",times)

import pylab as plt
if __name__ == "__main__":
    G = gt.load_graph("graphs/powerlaw500.dot")

    n = 30

    K = np.linspace(10,200,n,dtype=np.int32)
    l2g_vals = gen_l2g_spectrum(G,K)

    mds_vals = embed_mds(G,dr=False)
    tsnet_vals = embed_tsnet(G,dr=False)
    umap_vals = embed_umap(G,dr=False)

    titles = ["NE", "CD", "Stress"]

    fig,axes = plt.subplots(1,3)
    
    for i in range(3):
        ax = axes[i]

        ax.plot(K,l2g_vals[i],'o-',label="L2G")
        ax.plot(K,[mds_vals[i]] * n, '-', label="MDS")
        ax.plot(K,[tsnet_vals[i]] * n, '-', label="tsNET")
        ax.plot(K,[umap_vals[i]] * n, '-', label="UMAP")

        ax.legend()
        ax.set_title(titles[i])


    plt.show()



