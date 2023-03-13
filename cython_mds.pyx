import numpy as np 

#This resource is good: https://byumcl.bitbucket.io/bootcamp2014/_downloads/Lab20v1.pdf

cimport numpy as np
from numpy cimport ndarray as ar
cimport cython

from libc.stdlib cimport rand, malloc, free
from libc.math cimport sqrt, log, exp

cdef struct Pair:
    int u 
    int v 
    double d
    double w
    int in_neighbor

ctypedef Pair Pair_t


cdef void fisheryates(Pair *arr, int n):
    cdef Pair tmp

    for i in range(n-1,0,-1):
        tmp = arr[i]
        j = rand() % i 
        arr[i] = arr[j]
        arr[j] = tmp

@cython.cdivision(True)
cdef ar[double] schedule_convergent(Pair *pairs, int n_pairs, int t_max, double eps, int t_maxmax):
    cdef double w_min, w_max, new_val, eta
    w_min = (1/(pairs[0].d**2))
    w_max = w_min 
    for i in range(1,n_pairs):
        new_val = 1/(pairs[i].d ** 2)
        if new_val > w_max: w_max = new_val 
        if new_val < w_min: w_min = new_val 

    cdef double eta_max, eta_min 
    eta_max = 1.0 / w_min 
    eta_min = eps / w_max 

    cdef int t
    cdef double lamb = log( eta_max/eta_min ) / (t_max-1)
    cdef double eta_switch = 1.0/w_max 
    cdef ar[double] etas = np.zeros(t_maxmax)
    for t in range(t_maxmax):
        eta = eta_max * exp(-lamb*t)
        if (eta < eta_switch): break 
        etas[t] = eta
    cdef int tau = t 
    for t in range(tau,t_maxmax):
        eta = eta_switch / (1 + lamb*(t-tau))
        etas[t] = eta 
    
    return etas
        

@cython.cdivision(True)
cdef ar[double] sgd(ar[double] X, Pair *pairs, ar[double] steps, int n_vert,int n_pairs):
    cdef int epoch, p, i, j
    cdef int n_iter = steps.shape[0]
    cdef Pair pair
    cdef double d_ij, w_ij, mu, step, dx, dy, mag, r, r_x, r_y
    
    for epoch in range(n_iter):
        step = steps[epoch]
        fisheryates(pairs,n_pairs)
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


cdef Pair *get_pairs(ar[double, ndim=2] d):
    cdef int n = d.shape[0]
    cdef int n_pairs = (n * (n-1) ) / 2
    cdef Pair *pairs = <Pair *> malloc(
        n_pairs * sizeof(Pair)
    )

    cdef int i,u,v 
    i = 0
    for u in range(n):
        for v in range(u):
            pairs[i].u = u 
            pairs[i].v = v 
            pairs[i].d = d[u,v]
            pairs[i].w = 1.0
            pairs[i].in_neighbor = 1
            i += 1
    return pairs


def standard_mds(d,n_iter = 200, init_pos = None):
    cdef int n = d.shape[0]
    cdef int n_pairs = (n*(n-1))/2
    cdef ar[double] pos = np.random.uniform(-1,1,2*n)
    if init_pos: pos = init_pos

    cdef Pair *pairs = get_pairs(d)

    cdef ar[double] steps = schedule_convergent(pairs,n_pairs,30,0.01,200)

    pos = sgd(pos,pairs,steps,n,n_pairs)

    free(pairs)

    return pos.reshape((n,2))

def L2G(d,w,n_iter,init_pos):
    pass
