import numpy as np 

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, malloc, free

cdef struct Pair:
    int u 
    int v 
    float dist
    float weight

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

cdef double sgd(double *X, Pair *pairs, int n_vert,int n_pairs):
    cdef int n_iter = 10
    
    for epoch in range(n_iter):
        for p in range(n_pairs):
            Pair pair = pairs[p]
            cdef int i = pair.u 
            cdef int j = pair.v 
            cdef float dist = pair.dist
            cdef float weight = pair.weight
    return 0.0


cdef Pair *dummy_pair(int n_vertices):
    cdef int n = (n_vertices * (n_vertices - 1 )) / 2
    cdef Pair *pairs = <Pair *> malloc(
        n * sizeof(Pair)
    )
    cdef int u,v, i
    i = 0
    for u in range(n):
        for v in range(u):
            pairs[i].u = u
            pairs[i].v = v 
            pairs[i].dist = 1.0
            pairs[i].weight = 1.0
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
    


def get_pair():
    ptr = Pair(1,2,3.0)
    return ptr

def test_sgd():
    cdef int n = 50
    cdef int i
    cdef double *pos = <double *> malloc(n * sizeof(double))
    for i in range(n):
        pos[i] = i
    cdef Pair *pairs = dummy_pair(n)
    cdef double s
    s = sgd(pos,n)
    free(pos)
    print(s)
    return s
