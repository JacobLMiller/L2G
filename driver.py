import numpy as np
import graph_tool.all as gt 
from modules.L2G import find_neighbors
from modules.cython_l2g import L2G_opt
from modules.graph_metrics import apsp
from modules.thesne import tsnet 
from modules.graph_io import draw_tsnet_like as draw


def embed_tsnet(G):
    d = apsp(G)
    X = tsnet(d)
    draw(G,X)

def embed_umap(G):
    from umap import UMAP 
    d = apsp(G)
    X = UMAP(metric="precomputed",n_neighbors=10).fit_transform(d)
    draw(G,X)

def embed_mds(G):
    import s_gd2 
    E = np.array([(u,v) for u,v in G.iter_edges()],dtype=np.int32)
    I,J = E[:,0],E[:,1]
    X = s_gd2.layout(I,J)
    draw(G,X)

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


if __name__ == "__main__":
    G = gt.load_graph("graphs/connected_watts_1000.dot")
    embed_mds(G)