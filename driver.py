import numpy as np
import graph_tool.all as gt 
from modules.L2G import find_neighbors
from modules.cython_l2g import L2G_opt
from modules.graph_metrics import apsp
from modules.thesne import tsnet 
from modules.graph_io import draw_tsnet_like as draw
from modules.graph_io import read_cids


from modules.graph_metrics import get_stress, get_neighborhood, compute_graph_cluster_metrics, get_cluster_ids



def embed_tsnet(G,output=None,dr=True,c_ids=None):
    d = apsp(G)
    X = tsnet(d)
    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,c_ids),get_stress(X,d)

def embed_umap(G,output=None,dr=True,c_ids=None):
    from umap import UMAP 
    d = apsp(G)
    X = UMAP(metric="precomputed",n_neighbors=10).fit_transform(d)
    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,c_ids), get_stress(X,d)

def embed_mds(G,output=None,dr=True,c_ids=None):
    import s_gd2 
    E = np.array([(u,v) for u,v in G.iter_edges()],dtype=np.int32)
    I,J = E[:,0],E[:,1]
    X = s_gd2.layout(I,J)

    if dr: draw(G,X,output)
    return 1-get_neighborhood(G,X), compute_graph_cluster_metrics(G,X,c_ids), get_stress(X,apsp(G)),


def gen_l2g_spectrum(G,K=[10,35,72,100,1000],c_ids=None,alpha=0.6):
    d = apsp(G)
    print(d.dtype)
    s,n,c = list(), list(), list()
    for k in K:
        k = int(k) if k < G.num_vertices() else G.num_vertices() - 1
        print(k)
        w = find_neighbors(G,k=k,a=8)
        X = L2G_opt(d,w,alpha=alpha)
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

def tuple_add(t1,t2):
    return [e1+e2 for e1,e2 in zip(t1,t2)]

def compute_stats(G,n=30):
    mds_vals = [0,0,0]
    tsnet_vals = [0,0,0]
    umap_vals = [0,0,0]
    for _ in range(n):
        mds_vals = tuple_add(mds_vals,embed_mds(G,dr=False))
        tsnet_vals = tuple_add(tsnet_vals,embed_mds(G,dr=False))
        umap_vals = tuple_add(umap_vals,embed_mds(G,dr=False))
    return [e/n for e in mds_vals], [e/n for e in tsnet_vals], [e/n for e in umap_vals]

def testing():
    gname = "connected_watts_500"
    G = gt.load_graph(f"graphs/{gname}.dot")
    c_ids = read_cids(gname)
    n = 10

    K = np.linspace(10,200,n,dtype=np.int32)
    l2g_vals = gen_l2g_spectrum(G,K,c_ids=c_ids,alpha=0.01)

    mds_vals = embed_mds(G,dr=False,c_ids=c_ids)
    tsnet_vals = embed_tsnet(G,dr=False,c_ids=c_ids)
    umap_vals = embed_umap(G,dr=False,c_ids=c_ids)

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


import pylab as plt
if __name__ == "__main__":
    # G = gt.load_graph("graphs/block_2000.dot")
    # embed_tsnet(G,output="tsnet_block_2000.pdf")
    testing()



