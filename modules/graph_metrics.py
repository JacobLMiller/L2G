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
    return d


def get_cluster_ids(G: gt.Graph):
    """
    G -> a graph-tool graph 

    returns a list of sets corresponding to cluster ids
    """
    state = gt.minimize_blockmodel_dl(G)
    for _ in range(5):
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
    

def get_neighborhood(G,X):
    d = pairwise_distances(X)
    NE = 0
    for v in G.iter_vertices():
        neighbors = set(G.iter_all_neighbors(v))
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
    # c_ids,state = get_cluster_ids(G)
    # H = get_cluster_graph(G,c_ids)
    # high_d = apsp(H)
    high_d = get_cluster_distances(G,c_ids)
    low_d = find_cluster_centers(X,c_ids)

    return get_stress(low_d,high_d)


if __name__ == "__main__":
    G = gt.load_graph("graphs/connected_watts_400.dot")


    from modules.L2G import L2G,get_w
    w = get_w(G,k=30)
    X = L2G(apsp(G),weighted=True,w=w).solve(100)
    # X = L2G(apsp(G),weighted=False).solve(100)

    m1,m2,c_ids,state = compute_graph_cluster_metrics(G,X)
    print(m1,m2)

    def get_c_id(v):
        for i,c in enumerate(c_ids):
            if v in c:
                return i

    clusters = [cmap[get_c_id(v)] for v in G.iter_vertices()]
    clust_vp = G.new_vp("string",vals=clusters)

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    state.draw(pos=pos)
    # gt.graph_draw(G,pos=pos,vertex_fill_color=clust_vp)

    cX = find_cluster_centers(X,c_ids)
    import pylab 
    pylab.scatter(cX[:,0],cX[:,1])
    pylab.show()
