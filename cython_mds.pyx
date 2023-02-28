import numpy as np 

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, malloc, free
from libc.math cimport sqrt

cdef struct Pair:
    int u 
    int v 
    double d
    double w

cdef struct Vertex:
    float x
    float y

ctypedef Pair Pair_t


cdef  Pair * fisheryates(Pair *arr, int n):
    cdef Pair tmp

    for i in range(n-1,0,-1):
        tmp = arr[i]
        j = rand() % i 
        arr[i] = arr[j]
        arr[j] = tmp
    return arr

cdef float get_step_size():
    return 0.01

cdef double *sgd(double *X, Pair *pairs, int n_vert,int n_pairs):
    cdef int n_iter = 10
    cdef int epoch, p
    cdef Pair pair
    cdef int i, j 
    cdef double d_ij, w_ij, mu, step, dx, dy, mag, r, r_x, r_y
    
    for epoch in range(n_iter):
        step = 0.01
        for p in range(n_pairs):
            pair = pairs[p]
            i = pair.u
            j = pair.v 
            d_ij = pair.d 
            w_ij = pair.w 

            mu = step * w_ij 
            if mu > 1: mu = 1

            dx = X[i*2] - X[j*2]
            dy = X[i*2+1] - X[j*2+1]
            mag = sqrt(dx*dx + dy*dy)

            r = (mu * (mag-d_ij)) / (2*mag)
            r_x = r*dx 
            r_y = r*dy

            X[i*2] -= r_x 
            X[i*2+1] -= r_y 
            X[j*2] += r_x 
            X[j*2+1] += r_y



            
    return X


cdef Pair *dummy_pair(int n_vertices):
    cdef int n = (n_vertices * (n_vertices - 1 )) / 2
    cdef Pair *pairs = <Pair *> malloc(
        n * sizeof(Pair)
    )
    cdef int u,v, i
    i = 0
    for u in range(n_vertices):
        for v in range(u):
            pairs[i].u = u
            pairs[i].v = v 
            pairs[i].d = 1.0
            pairs[i].w = 1.0
            i += 1
    return pairs



def shuffle(d):
    cdef int n = (len(d) * (len(d)-1)) / 2
    cdef int i = 0


    cdef Pair *pairs = <Pair *> malloc(
        n * sizeof(Pair))
    for u in range(len(d)):
        for v in range(u):
            pairs[i].u = u
            pairs[i].v = v 
            pairs[i].dist = d[u,v]

            i += 1
    pairs = fisheryates(pairs,n)
    print(pairs[0])
    







def test_sgd():
    cdef int n = 50
    cdef int i
    cdef double *pos = <double *> malloc(2 * n * sizeof(double))
    for i in range(2*n):
        pos[i] = i
    cdef Pair *pairs = dummy_pair(n)
    pos = sgd(pos, pairs, n, (n*(n-1))/2)
    cdef double[:] view1 = pos 
    free(pos)
    return view1
