import numpy as np

cimport numpy as np 
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _naive_dot(np.ndarray[np.float32_t, ndim=2] A, np.ndarray[np.float32_t, ndim=2] B):
    cdef np.ndarray[np.float32_t, ndim=2] C 
    cdef int n, p, m
    cdef np.float32_t s
    if A.shape[1] != B.shape[0]: raise ValueError("shape doesn't match")

    n,p,m = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((n,m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += A[i,k] * B[k,j]
            C[i,j] = s 
    return C 


def naive_dot(A,B):
    return _naive_dot(A,B)