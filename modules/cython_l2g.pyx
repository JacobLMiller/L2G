import numpy as np 
import random

#This resource is good: https://byumcl.bitbucket.io/bootcamp2014/_downloads/Lab20v1.pdf

cimport numpy as np
from numpy cimport ndarray as ar
cimport cython

from libc.stdlib cimport rand, malloc, free, srand
from libc.math cimport sqrt, log, exp

srand(random.randint(0,100000))

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
cdef ar[double] sgd(ar[double] X, Pair *pairs, ar[double] steps, int n_vert,int n_pairs,float alpha):
    cdef int epoch, p, i, j
    cdef int n_iter = steps.shape[0]
    cdef Pair pair
    cdef double d_ij, w_ij, mu, step, dx, dy, mag, r, r_x, r_y, gx, gy, repx, repy, s_x, s_y
    cdef double l_sum = 1 + alpha
    
    for epoch in range(n_iter):
        step = steps[epoch]
        fisheryates(pairs,n_pairs)
        # if epoch > 31: 
        #     t = 0.
        #     l_sum = 1 + t
        for p in range(n_pairs):
            pair = pairs[p]
            i = pair.u
            j = pair.v 
            d_ij = pair.d 
            w_ij = pair.w 

            mu = (step * w_ij) / (d_ij * d_ij)
            if mu > 1: mu = 1

            dx = X[i*2] - X[j*2]
            dy = X[i*2+1] - X[j*2+1]
            mag = sqrt(dx*dx + dy*dy)

            r = (mu * (mag-d_ij)) / (2*mag)
            s_x = r*dx 
            s_y = r*dy

            mu = step
            if mu > 1: mu = 1
            gx,gy = dx/(mag*mag), dy/(mag*mag)
            repx,repy = (1-w_ij) * (-mu * gx), (1-w_ij) * (-mu * gy) 
            repx /= mag 
            repy /= mag


            r_x = (1/l_sum)*s_x + (alpha/l_sum)*repx 
            r_y = (1/l_sum)*s_y + (alpha/l_sum)*repy

            X[i*2] -= r_x
            X[i*2+1] -= r_y
            X[j*2] += r_x
            X[j*2+1] += r_y

    return X


cdef Pair *get_pairs(ar[double, ndim=2] d, ar[np.int16_t,ndim=2] w):
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
            pairs[i].w = w[u,v]
            pairs[i].in_neighbor = w[u,v]
            i += 1
    return pairs


cdef ar[double] mds_driver(
        ar[double] pos, 
        Pair *pairs, 
        int n_iter, 
        double eps,
        int n, 
        int n_pairs,
        float alpha
    ):

    cdef ar[double] steps = schedule_convergent(pairs,n_pairs, 30, eps, n_iter)
    pos = sgd(pos,pairs,steps,n,n_pairs,alpha)
    free(pairs)

    return pos

def standard_mds(d,n_iter = 200, init_pos = None,eps=0.01,alpha=0):
    cdef int n = d.shape[0]
    cdef int n_pairs = (n*(n-1))/2
    cdef ar[double] pos = np.random.uniform(-1,1,2*n)
    if init_pos: pos = init_pos

    cdef ar[np.int16_t, ndim=2] w = np.ones(d.shape,dtype=np.int16)
    cdef Pair *pairs = get_pairs(d,w)

    return mds_driver(pos,pairs,n_iter,eps,n,n_pairs,alpha).reshape((n,2))

def L2G_opt(d,w,n_iter=200,init_pos=None,eps=0.01,alpha=0.6):
    cdef int n = d.shape[0]
    cdef int n_pairs = (n*(n-1))/2
    cdef ar[double] pos = np.random.uniform(-1,1,2*n)
    if init_pos: pos = init_pos

    cdef Pair *pairs = get_pairs(d,w)

    return mds_driver(pos,pairs,n_iter,eps,n,n_pairs,alpha).reshape((n,2))
