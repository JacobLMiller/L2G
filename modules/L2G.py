import graph_tool.all as gt
import numpy as np 

def L2G(G: gt.Graph, k: int, a=10,weights=None,alpha=0.6):
    from modules.cython_l2g import L2G_opt 
    from modules.graph_metrics import apsp 
    d = apsp(G,weights)
    w = find_neighbors(G,k,a)
    return L2G_opt(d,w,alpha=alpha)


def sum_diag_powers(L,a):
    print(sum(L ** p for p in range(1,a+1)))

def norm_counts(A: np.array, p: int):
    s = 0.1
    B = np.linalg.matrix_power(A,p)
    B = pow(s,p) * B 
    return B / np.max(B)

def find_neighbors_large(G,k=25,a=10,eps=0):
    A = gt.adjacency(G).toarray()
    L,Q = np.linalg.eigh(A)
    Lp = sum(pow(L,p) for p in range(1,a+1))
    A = np.matmul(np.matmul(Q,np.diag(Lp)),Q.T)
    return A

def find_neighbors_small(G,k,a,eps):
    A = gt.adjacency(G).toarray()
    mp = np.linalg.matrix_power
    A = sum([norm_counts(A,i) for i in range(1,a+1)])
    return A

def find_neighbors(G,k=5,a=5,eps=0):
    if G.num_vertices() > 1000:
        A = find_neighbors_large(G,k,a,eps)
    else: A = find_neighbors_small(G,k,a,eps)

    B = np.argsort(A,axis=1)
    w = np.zeros(A.shape,dtype=np.int16)
    for i in range(len(A)):
        A_star = B[i][::-1][:k+1]
        for v in A_star:
            if A[i][v] == 0: break
            if i != v and v:
                w[i,v] = 1 
                w[v,i] = 1


    return w

    

if __name__ == "__main__":
    G = gt.lattice((5,5))
    w = find_neighbors(G,k=5)
    print(w)

