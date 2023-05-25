import numpy as np
import graph_tool.all as gt
from sklearn.metrics import pairwise_distances


def find_cluster_centers(X,c_ids):
    c_ids = [list(c) for c in c_ids]
    cluster_centers = np.zeros( (len(c_ids),X.shape[1]) )
    for i,clusters in enumerate(c_ids):
        Xi = X[clusters]
        center = Xi.mean(axis=0)
        
        cluster_centers[i] = center
    return cluster_centers


def get_stress(X,d):
    from math import comb
    N = len(X)
    ss = (X * X).sum(axis=1)

    diff = np.sqrt( abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)) )

    np.fill_diagonal(diff,0)
    stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) ) / comb(N,2)

    from scipy.optimize import minimize_scalar
    min_a = minimize_scalar(stress)
    #print("a is ",min_a.x)
    return stress(a=min_a.x)

def normal_stress(X,d):
    N = len(X)
    ss = (X * X).sum(axis=1)
    diff = np.sqrt(abs(ss.reshape((N,1)) + ss.reshape((1,N)) - 2 * np.dot(X,X.T)))
    np.fill_diagonal(diff, 0)
    stress = lambda a:  np.sum( np.square( np.divide( (a*diff-d), d , out=np.zeros_like(d), where=d!=0) ) )

    from scipy.optimize import minimize_scalar
    min_a = minimize_scalar(stress)

    ss = 0 
    for i in range(N):
        for j in range(i):
            ss += (d[i,j] * d[i,j])
    return stress(a=min_a.x) / ss


def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

def cluster_preservation2(dh,dl,c_ids):
    from scipy.stats import kendalltau


    percision = 0
    for i in range(len(dh)):
        percision += normalised_kendall_tau_distance(dh[i],dl[i])
    return percision / len(dh)

"""
A function (G, X, cluster_ids)

Compute a super-graph, defining distance between distance somehow (adjacecny, weighted based on number of edges, stephen suggestion)

Compute how similar the graph theoretic distnaces and embedded distances are between clusters. 


Load a graph, 
compute clusters,
compute cluster-graph 
compute embedding 
compute quality metric

"""

def apsp(G,weights=None):
    d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
    if weights:
        d /= np.sum(d,axis=0)
    return d


def get_cluster_ids(G: gt.Graph,r: int):
    """
    G -> a graph-tool graph 

    returns a list of sets corresponding to cluster ids
    """
    state = gt.minimize_blockmodel_dl(G)
    for _ in range(r):
        tmp_state = gt.minimize_blockmodel_dl(G)
        if tmp_state.entropy() < state.entropy():
            state = tmp_state
    membership = list(state.get_blocks())
    clusters = np.unique(membership)
    block_map = {c: i for i,c in enumerate(clusters)}
    c_ids = [set() for _ in block_map]
    for v,c in enumerate(membership):
        c_ids[block_map[c]].add(v)
    
    return c_ids,state

def maybe_add_vertex(G,H,u,v,c_ids):
    if any(G.edge( i,j ) for i in c_ids[u] for j in c_ids[v]):
        H.add_edge(u,v)

def weight_cluster_edge(G,H,u,v,c_ids):
    return sum(1 if G.edge( i,j ) else 0 for i in c_ids[u] for j in c_ids[v])
            
def get_cluster_graph(G,c_ids):
    H = gt.Graph(directed=False)
    n,m = len(c_ids), G.num_vertices()
    H.add_vertex(n)
    counts = np.zeros((n,n))
    for u in range(n):
        for v in range(n):
            maybe_add_vertex(G,H,u,v,c_ids)
    
    gt.remove_parallel_edges(H)
    gt.remove_self_loops(H)
    return H
    

def get_neighborhood(G: gt.Graph, X: np.ndarray, r=2):
    d = pairwise_distances(X)
    NE = 0
    for v in G.iter_vertices():
        neighbors = set(G.iter_all_neighbors(v))
        for _ in range(r-1):
            for u in neighbors.copy():
                neighbors.update(G.iter_all_neighbors(u))

        k = len(neighbors)
        if k == 0: continue
        top_k = set(np.argsort(d[v])[1:k+1])
        NE += (len(top_k.intersection(neighbors)) / len(top_k.union(neighbors)))
    return NE / G.num_vertices()


def compute_cluster_metric(G,X,c_ids):
    H = get_cluster_graph(G,c_ids)
    low_d_clusters = find_cluster_centers(X,c_ids)

    return get_neighborhood(H,low_d_clusters)

def get_cluster_distances(G,c_ids):
    n = len(c_ids)
    s = np.zeros((n,n))

    for u in range(n):
        for v in range(u+1):
            x = sum(1 if G.edge(i,j) else 0 for i in c_ids[u] for j in c_ids[v])
            x = np.exp(-(x**2))
            s[u,v] = x
            s[v,u] = s[u,v]
    d = 1-(s/np.max(s,axis=0))
    return (d + d.T) / 2

def compute_graph_cluster_metrics(G,X,c_ids):
    high_d = get_cluster_distances(G,c_ids)
    low_d = find_cluster_centers(X,c_ids)

    return get_stress(low_d,high_d)

def get_metrics(G: gt.Graph,X: np.array,d: np.array,c_ids: list[set]):
    ne = 1-get_neighborhood(G,X)
    cd = compute_graph_cluster_metrics(G,X,c_ids)
    st = get_stress(X,d)
    tst = compute_cluster_metric(G,X,c_ids)
    return ne,tst,st

if __name__ == "__main__":
    G = gt.load_graph("graphs/12square.dot")

    n = G.num_vertices()
    rep = 20

    layout = np.random.uniform(-1,1,(n,2))

    d = apsp(G)

    random_layouts_vahan = list()
    random_layouts = list()
    random_np = list()
    for _ in range(rep):
        X = np.random.uniform(-20,20,(n,2))
        random_layouts_vahan.append(normal_stress(X,d))
        random_layouts.append(get_stress(X,d))
        random_np.append(get_neighborhood(G,X))

    tsne_layouts_vahan = list()
    tsne_layouts = list()
    from sklearn.manifold import TSNE
    for _ in range(rep):
        X = TSNE(metric="precomputed").fit_transform(d)
        tsne_layouts_vahan.append(normal_stress(X,d))
        tsne_layouts.append(get_stress(X,d))

    mds_layouts_vahan = list()
    mds_layouts = list()
    from s_gd2 import layout_convergent
    I = [u for u,_ in G.iter_edges()]
    J = [v for _,v in G.iter_edges()]    
    for _ in range(rep):
        X = layout_convergent(I,J)
        mds_layouts_vahan.append(normal_stress(X,d))
        mds_layouts.append(get_stress(X,d))


    import pylab as plt 
    plt.plot([0]*rep, random_layouts, 'o', alpha=0.3, label="random layouts")
    plt.plot([1]*rep, tsne_layouts, 'o', alpha=0.3, label="tsne layouts") 
    plt.plot([2]*rep, mds_layouts, 'o', alpha=0.3, label="mds layouts")   
    plt.suptitle("Old stress computation (rescaling)")    
    plt.legend()
    plt.xticks([])
    plt.show()

    plt.plot([0]*rep, random_layouts_vahan, 'o', alpha=0.3, label="random layouts")
    plt.plot([1]*rep, tsne_layouts_vahan, 'o', alpha=0.3, label="tsne layouts") 
    plt.plot([2]*rep, mds_layouts_vahan, 'o', alpha=0.3, label="mds layouts")   
    plt.suptitle("Normalized stress computation (division by sum of square graph distances)")    
    plt.legend()
    plt.xticks([])
    plt.show()    

               
