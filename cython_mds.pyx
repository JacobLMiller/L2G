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

cdef float* sgd(float *pos, Pair *pairs, int n_iter, int n_pairs):
    cdef float step, dist
    cdef int epoch, p, i, j

    for epoch in range(n_iter):
        step = get_step_size()

        for p in range(n_pairs):
            i = pairs[p].u
            j = pairs[p].v
            dist = pairs[p].dist

            x1,y1 = pos[i*2], pos[i*2 + 1]
            x2,y2 = pos[j*2], pos[j*2 + 1]







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