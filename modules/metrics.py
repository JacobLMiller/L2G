import numpy as np

from sklearn.metrics import pairwise_distances

# @jit(nopython=True)
# def get_norm_stress(X,d):
#     norm, stress, n = np.linalg.norm, 0, len(X)
#
#     for i in range(n):
#         for j in range(n):
#             stress += pow(d[i][j] - norm(X[i]-X[j]),2)
#     return stress / np.sum(np.square(d))


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


def get_cost(X,d,w,t):                 # Define a function
    stress, l_sum,eps = 0, 1+t, 1e-13
    N = len(X)

    #Stress
    ss = (X * X).sum(axis=1)
    diff = ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X,X.T)
    diff = np.sqrt(np.maximum(diff,eps))
    stress = np.sum( w * np.square(d-diff) )

    #repulsion
    diff = diff + eps

    # condlist = [diff<10, diff>=10]
    # choicelist = [np.log(diff), 0]
    # r = np.select(condlist, choicelist, 0)
    # r = -np.sum( r )
    r = -np.sum( np.log(diff+eps) )

    return ((1/l_sum) * np.sum(stress) + (t/l_sum) * r) / N **2


def get_neighborhood(X,d,rg = 2):
    """
    How well do the local neighborhoods represent the theoretical neighborhoods?
    Closer to 1 is better.
    Measure of percision: ratio of true positives to true positives+false positives
    """
    norm = np.linalg.norm
    def get_k_embedded(X,k_t):
        dist_mat = pairwise_distances(X)
        return [np.argsort(dist_mat[i])[1:len(k_t[i])+1] for i in range(len(dist_mat))]

    k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

    k_embedded = get_k_embedded(X,k_theory)


    NP = 0
    for i in range(len(X)):
        if len(k_theory[i]) <= 0:
            continue
        intersect = np.intersect1d(k_theory[i],k_embedded[i]).size
        jaccard = intersect / (2*k_theory[i].size - intersect)

        NP += jaccard

    return NP / len(X)

import graph_tool.all as gt
def get_neighborhood_graph(X: np.array, G: gt.Graph, rg = 2):
    degs = G.get_total_degrees(G.iter_vertices())
    print(degs)





def MAP(G,X):
    V = G.num_vertices()

    embed_dist = pairwise_distances(X)
    R = np.array([np.argsort(embed_dist[i])[1:] for i in range(V)])

    outer_sum = 0
    for a in G.vertices():
        Na = np.array([int(v) for v in a.out_neighbors()])
        v = int(a)

        inner_sum = 0
        for i in range(Na.size):
            b_i = R[v].tolist().index(Na[i])
            Ra_bi = R[v][:b_i+1]

            inner_sum += np.intersect1d(Na,Ra_bi).size / Ra_bi.size

        outer_sum += inner_sum/Na.size

    return outer_sum / V

def KL_div(X,d,s):
    p = lambda d: np.exp(-pow(d,2)/s)
    q = lambda m: np.exp(-pow(np.linalg.norm(m),2)/s)
    n = len(d)

    P = np.array([[p(d[i][j]) if i != j else 0 for j in range(n)]
            for i in range(n)])

    for i in range(len(P)):
        P[i] = P[i] / np.linalg.norm(P[i])

    Q = np.array([[q(X[i]-X[j]) if i != j else 0 for j in range(n)]
            for i in range(n)])

    for i in range(len(Q)):
        Q[i] = Q[i] / np.linalg.norm(Q[i])



    KL = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                p_ij = P[i][j]
                q_ij = Q[i][j]

                KL += p_ij*np.log(p_ij/q_ij) + q_ij*np.log(q_ij/p_ij)
    return KL/2



def chen_neighborhood(D,X,k):
    embed_dist = pairwise_distances(X)

    percision = 0
    for i in range(len(D)):
        embed_k = set(np.argsort(embed_dist[i])[1:k+1])
        hd_k = set(np.argsort(D[i])[1:k+1])

        N_ki = len(embed_k.intersection(hd_k))
        percision += N_ki

    return percision / (len(D) * k)

def avg_lcl_err(X,D):
    embed = pairwise_distances(X)

    n = len(D)
    err = [0 for _ in range(n)]
    for i in range(n):
        max_theory = np.max(D[i])
        max_embed = np.max(embed[i])

        local = 0
        for j in range(n):
            if i != j:
                local += abs(D[i][j]/max_theory - embed[i][j]/max_embed)
        err[i] = local / (n-1)
    return err

def find_cluster_centers(X,y):
    unq,inv = np.unique(y,return_inverse=True)
    c_ids = [ list() for _ in unq]
    [c_ids[inv[i]].append(i) for i,_ in enumerate(y)]
    cluster_centers = np.zeros( (len(c_ids),X.shape[1]) )
    for i,clusters in enumerate(c_ids):
        Xi = X[clusters]
        center = Xi.mean(axis=0)
        
        cluster_centers[i] = center
    return cluster_centers

def cluster_distance(H,X,y):
    high_d_clusters = find_cluster_centers(H,y)
    low_d_clusters = find_cluster_centers(X,y)

    dh = pairwise_distances(high_d_clusters)
    dl = pairwise_distances(low_d_clusters)
    return get_stress(low_d_clusters,dh)


def cluster_preservation(H,X,c_ids):
    high_d_clusters = find_cluster_centers(H,c_ids)
    low_d_clusters = find_cluster_centers(X,c_ids)

    dh = pairwise_distances(high_d_clusters)
    return chen_neighborhood(dh,low_d_clusters,k=2)

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


from scipy.spatial.distance import mahalanobis

def comp_avg_mahalanobis(A,B,Si):
    b = np.mean(B,axis=0)
    s = 0
    for row in A:
        s += mahalanobis(row,b,Si)
    return s / A.shape[0]

def mahalonobis_metric(HD,LD,c_ids):
    num_c = len(c_ids)
    for i in range(num_c):
        for j in range(i):
            HD_c1 = HD[c_ids[i]]
            HD_c2 = HD[c_ids[j]]
            Si = np.linalg.inv(np.cov(HD_c2.T))
            print(f"Distance between cluster {i} and cluster {j} in HD is {comp_avg_mahalanobis(HD_c1,HD_c2,Si)}")
            Si = np.linalg.inv(np.cov(HD_c1.T))
            print(f"Distance between cluster {j} and cluster {i} in HD is {comp_avg_mahalanobis(HD_c2,HD_c1,Si)}")

def knn_accuracy(X,y,k=7):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=k).fit(X,y).score(X,y)


def compute_metrics(H: np.array,d: np.array,X: np.array,y: np.array):
    return 1-knn_accuracy(X,y),cluster_distance(H,X,y) ,get_stress(X,d)
