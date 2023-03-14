import graph_tool.all as gt
import numpy as np 

def sum_diag_powers(L,a):
    print(sum(L ** p for p in range(1,a+1)))

def norm_counts(A: np.array, p: int):
    s = 0.1
    B = np.linalg.matrix_power(A,p)
    B = pow(s,p) * B 
    return B / np.max(B)

def find_neighbors(G,k=5,a=5,eps=0):
    import graph_tool.all as gt
    A = gt.adjacency(G).toarray()
    mp = np.linalg.matrix_power
    A = sum([norm_counts(A,i) for i in range(1,a+1)])

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

