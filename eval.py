import numpy as np
import graph_tool.all as gt
import time
from tqdm import tqdm 
import s_gd2

from modules.metrics import get_stress,get_neighborhood
from modules.graph_metrics import compute_graph_cluster_metrics, get_cluster_ids, apsp

from modules.L2G import find_neighbors
from modules.cython_l2g import L2G_opt, standard_mds

from modules.thesne import tsnet

from modules.graph_io import draw_tsnet_like as draw


K = [4,8,16,32,64,85,100,150,200,400,600]


def diffusion_weights(d,a=5, k = 20, sigma=1):
    #Transform distance matrix
    diff = np.exp( -(d**2) / (sigma **2) )
    diff /= np.sum(diff,axis=0)

    #Sum powers from 1 to a
    mp = np.linalg.matrix_power
    A = sum( pow(0.1,i) * mp(diff,i) for i in range(1,a+1) )

    #Find k largest points for each row 
    Neighbors = set()
    for i in range(diff.shape[0]):
        args = np.argsort(A[i])[::-1][1:k+1]
        for j in args:
            Neighbors.add( (int(i),int(j)) )

    #Set pairs to 1
    w = np.zeros_like(diff)
    for i,j in Neighbors:
        w[i,j] = 1
        w[j,i] = 1
    return w


def embed_l2g(d,num_params,name,G,f_name="transformation"):
    Xs = list()
    outs = list()
    for i, k in enumerate(K):
        k = int(k) if k < G.num_vertices() else G.num_vertices() - 1
        w = find_neighbors(G,k=k,a=10)
        X = L2G_opt(d,w)

        outstr = f"drawings/l2g/{name}_{k}.pdf"
        outs.append(outstr)
        Xs.append(X)

    return Xs, outs

def embed_tsne(d,num_params,name,G):
    X = tsnet(d)

    outstr = f"drawings/tsne/{name}.pdf"
    return [X], [outstr]

def embed_mds(d,num_params,name,G):
    E = np.array([(u,v) for u,v in G.iter_edges()],dtype=np.int32)
    I,J = E[:,0], E[:,1]
    X = s_gd2.layout(I,J)

    outstr = f"drawings/mds/{name}.pdf"
    return [X], [outstr]

def embed_umap(d,num_params,name,G):
    from umap import UMAP
    X = UMAP(metric="precomputed").fit_transform(d)

    outstr = f"drawings/umap/{name}.pdf"
    return [X], [outstr]


def get_zeros(n):
    return np.zeros(n,dtype=float)


#Function that takes a graph, embedding function, parameter range, num_params, num_repeats
#Returns list of NP, stress, and draws at drawings/func_name/num_params.pdf
def embedding(G,d, f,g_name,num_params=5, num_repeats=5,c_ids=None,state=None):

    g_name, _ = g_name.split(".")
    num_params = len(K)


    NE, stress, times = get_zeros(num_params), get_zeros(num_params), get_zeros(num_params)
    m1,m2,m3,m4 = [get_zeros(num_params) for _ in range(4)]
    for _ in range(num_repeats):
        start = time.perf_counter()
        Xs, output = f(d,num_params,g_name,G)
        end = time.perf_counter()


        NE += np.array( [get_neighborhood(X,d) for X in Xs] )
        stress += np.array([get_stress(X,d) for X in Xs])
        times += (end-start)/len(Xs)

        m = np.array([compute_graph_cluster_metrics(G,X,c_ids) for X in Xs])
        m1 += m[:,0]
        m2 += m[:,1] 

        for X,out in zip(Xs,output):
            draw(G,X,out)

    
    NE /= num_repeats 
    stress /= num_repeats
    times /= num_repeats
    m1 /= num_repeats
    m2 /= num_repeats

    return NE,stress, times,m1,m2


def experiment(n=5):
    import os
    import pickle
    import copy

    path = 'table_graphs/'
    graph_paths = os.listdir(path)

    # graph_paths = list( map(lambda s: s.split('.')[0], graph_paths) )
    graphs = [(gt.load_graph(f"{path+graph}"),graph) for graph in graph_paths]
    graphs = sorted(graphs,key=lambda x: x[0].num_vertices())
    graph_paths = [g for _,g in graphs]
    print(graph_paths)


    e_funcs = {
        "l2g": embed_l2g,
        # # "tsne": embed_tsne,
        "mds": embed_mds,
        "umap": embed_umap
    }

    data = {
        e: {g: {"NE": None, "stress": None} for g in graph_paths}
        for e in e_funcs.keys()
    }

    import pickle

    for i,graph in enumerate(tqdm(graph_paths)):

        G = gt.load_graph(f"{path+graph}")

        c_ids, state = get_cluster_ids(G)
        d = apsp(G)
        for f_name, f in e_funcs.items():
            if f_name == "tsne" and G.num_vertices() > 2100: continue
            NE, stress,times,m1,m2 = embedding(G,d,f,graph,c_ids=c_ids,state=state)
            data[f_name][graph]["NE"] = 1-NE
            data[f_name][graph]["stress"] = stress
            data[f_name][graph]["time"] = times
            data[f_name][graph]["m1"] = m1
            data[f_name][graph]["m2"] = m2
    
            filehandler = open("data/03_19_2.pkl", 'wb') 

            pickle.dump(data, filehandler)
            filehandler.close()

    

if __name__ == '__main__':
    experiment(n=30)