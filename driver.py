import numpy as np
import graph_tool.all as gt 
from modules.L2G import find_neighbors
from modules.cython_l2g import L2G_opt
from modules.graph_metrics import apsp

def sample_k(max):

    accept = False

    while not accept:

        k = np.random.randint(1,max+1)

        accept = np.random.random() < 1.0/k

    return k

def get_graph(n=200):
    return gt.random_graph(n, lambda: sample_k(40), model="probabilistic-configuration",

                    edge_probs=lambda i, k: 1.0 / (1 + abs(i - k)), directed=False,

                    n_iter=100)

def measure_time(repeat=5):
    import time
    sizes = list(range(200,4005,200))
    print(len(sizes))
    times = np.zeros(len(sizes))

    for i,n in enumerate(sizes):
        print(f"On the {n}th size")
        for _ in range(repeat):
            G = get_graph(n)
            start = time.perf_counter() 
            d = apsp(G)
            w = find_neighbors(G,a=10,k=35)
            X = L2G_opt(d,w,200)
            end = time.perf_counter()

            val = end-start
            times[i] += val

        times[i] /= repeat
        np.savetxt("data/03_13_new_timeexp.txt",times)


if __name__ == "__main__":
    measure_time()
    # G = gt.lattice((5,5))
    # d = apsp(G)

    # import time 
    # start = time.perf_counter()
    # print("Starting neighbor")
    # w = find_neighbors(G,k=2,a=5)
    # print("Starting optimization")
    # X = L2G_opt(d,w)
    # print(f"Optimization took {time.perf_counter()-start}s")

    # pos = G.new_vp("vector<float>")
    # pos.set_2d_array(X.T)

    # gt.graph_draw(G,pos=pos)