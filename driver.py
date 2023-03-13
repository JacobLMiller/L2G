import numpy as np
import graph_tool.all as gt 
from cython_mds import standard_mds

def apsp(G,weights=None):
    d = np.array( [v for v in gt.shortest_distance(G,weights=weights)] ,dtype=float)
    return d

if __name__ == "__main__":
    G = gt.lattice((100,5))
    d = apsp(G)

    import time 
    start = time.perf_counter()
    X = standard_mds(d)
    print(f"Optimization took {time.perf_counter()-start}s")

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)

    gt.graph_draw(G,pos=pos)