import graph_tool.all as gt 
import numpy as np
import pylab as plt
from modules.graph_io import get_table_graphs, read_cids
from modules.graph_metrics import apsp, get_metrics
from modules.L2G import find_neighbors 
from modules.cython_l2g import L2G_opt
from umap import UMAP
from modules.thesne import tsnet
import s_gd2



def matrix_power_exp(n=5):
    graphs = get_table_graphs()
    for G,name in graphs:
        d = apsp(G)
        c_ids = read_cids(name.split(".")[0])
        spectrum = np.arange(1,50,3)
        n_scores, c_scores, s_scores = np.zeros_like(spectrum,dtype=np.float64),np.zeros_like(spectrum,dtype=np.float64),np.zeros_like(spectrum,dtype=np.float64)
        for i,c in enumerate(spectrum):
            print(f"On graph {name.split('.')[0]} experiment {c}",end='\r')
            NE, CD, ST = 0,0,0
            for _ in range(n):
                w = find_neighbors(G,k=25,a=c)
                X = L2G_opt(d,w)
                ne, cd, st = get_metrics(G,X,d,c_ids)
                NE += ne 
                CD += cd 
                ST += st 
            n_scores[i] = NE/n 
            c_scores[i] = CD/n
            s_scores[i] = ST/n
        plt.plot(spectrum,n_scores,'o-',label="NE")
        plt.plot(spectrum,c_scores,'o-',label="CD")        
        plt.plot(spectrum,s_scores,'o-',label="ST")
        plt.legend()
        plt.xlabel("c (largest adjacency matrix power)")
        plt.ylabel("Metric score")
        plt.ylim(0,1)
        plt.suptitle(f"c parameter on {name.split('.')[0]} graph")
        plt.savefig(f"figures/c_exp/{name.split('.')[0]}.pdf")
        plt.clf()
        print("",end="\r")

        data = np.concatenate((n_scores[:,np.newaxis],c_scores[:,np.newaxis],s_scores[:,np.newaxis]),axis=1)
        print(f"My shape is {data.shape}")
        np.savetxt(
            f"data/c_scores/{name.split('.')[0]}.txt",
            data            
        )


def alpha_exp(n=5):
    graphs = get_table_graphs()
    for G,name in graphs:
        d = apsp(G)
        c_ids = read_cids(name.split(".")[0])
        spectrum = np.linspace(0,1,15)
        n_scores, c_scores, s_scores = np.zeros_like(spectrum,dtype=np.float64),np.zeros_like(spectrum,dtype=np.float64),np.zeros_like(spectrum,dtype=np.float64)
        for i,a in enumerate(spectrum):
            print(f"On graph {name.split('.')[0]} experiment {a}",end='\r')
            NE, CD, ST = 0,0,0
            for _ in range(n):
                w = find_neighbors(G,k=25,a=8)
                X = L2G_opt(d,w,alpha=a)
                ne, cd, st = get_metrics(G,X,d,c_ids)
                NE += ne 
                CD += cd 
                ST += st 
            n_scores[i] = NE/n 
            c_scores[i] = CD/n
            s_scores[i] = ST/n
        plt.plot(spectrum,n_scores,'o-',label="NE")
        plt.plot(spectrum,c_scores,'o-',label="CD")        
        plt.plot(spectrum,s_scores,'o-',label="ST")
        plt.legend()
        plt.xlabel("alpha (strength of repulsive force)")
        plt.ylabel("Metric score")
        plt.ylim(0,1)
        plt.suptitle(f"alpha parameter on {name.split('.')[0]} graph")
        plt.savefig(f"figures/a_exp/{name.split('.')[0]}.pdf")
        plt.clf()
        print("\n",end="\r")
        np.savetxt(
            f"data/c_scores/{name.split('.')[0]}.txt",
            np.concatenate((n_scores[:,np.newaxis],c_scores[:,np.newaxis],s_scores[:,np.newaxis]),axis=1)
        )        

def sample_k(max):
    accept = False
    while not accept:
        k = np.random.randint(1,max+1)
        accept = np.random.random() < 1.0/k
    return k

def get_graph(n=200):
    return gt.triangulation(np.random.uniform(-1,1,size=(n,2)))[0]

def embed_l2g(G):
    d = apsp(G)
    w = find_neighbors(G,k=25,a=8)
    X = L2G_opt(d,w,alpha=0.01)



def embed_tsne(G):
    d = apsp(G)
    X = tsnet(d)

def embed_mds(G):
    E = np.array([(u,v) for u,v in G.iter_edges()],dtype=np.int32)
    I,J = E[:,0], E[:,1]
    X = s_gd2.layout_convergent(I,J)

def embed_umap(G):
    d = apsp(G)
    X = UMAP(metric="precomputed").fit_transform(d)

def measure_time(f,f_name,repeat=5):
    import time
    sizes = list(range(1400,2001,200))
    print(len(sizes))
    times = np.zeros((len(sizes),repeat))

    for i,n in enumerate(sizes):
        print(f"On the {n}th size")
        for j in range(repeat):
            G = get_graph(n)
            start = time.perf_counter() 
            f(G)
            end = time.perf_counter()

            val = end-start
            times[i,j] = val

        # times[i] /= repeat
        np.savetxt(f"data/03_28_time_{f_name}.txt",times)

def time_driver(n=15):
    funcs = {
        # "mds": embed_mds,
        # "l2g": embed_l2g,
        "tsnet": embed_tsne,
        # "umap": embed_umap
    }
    for fname, f in funcs.items():
        measure_time(f,fname,n)

if __name__ == "__main__":
    time_driver(15)