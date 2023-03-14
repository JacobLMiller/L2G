#!/usr/bin/env python3

if __name__ == '__main__':
    #Driver script adapted from https://github.com/HanKruiger/tsNET
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with L2G.')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--k', '-k', type=int, help='Number of most-connected neighbors to find')
    parser.add_argument('--tau', '-t', type=float, default=0.6, help='Strength of the repulsive force. Should be a nonnegative value between 0 and 1, 1 being equally as strong as the MDS term.')
    parser.add_argument("--alpha", "-a", type=int, default = 5, help="Number of powers to consider in adjacency matrix")
    parser.add_argument('--epsilon', '-e', type=float, default=1e-7, help='Threshold for convergence.')
    parser.add_argument('--max_iter', '-m', type=int, default=200, help='Maximum number of iterations.')
    parser.add_argument('--learning_rate', '-l', type=str, default="convergent", help='Learning rate (hyper)parameter for optimization.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Save layout to the specified file.')

    args = parser.parse_args()

    #Import needed libraries
    import os
    import time
    import graph_tool.all as gt
    import numpy as np

    #Check for valid input
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]

    #Global hyperparameters
    max_iter = args.max_iter
    eps = args.epsilon
    lr = args.learning_rate
    k = args.k
    alpha = args.alpha
    tau = args.tau

    print(f"Reading input graph: {graph_name}")
    G = gt.load_graph(args.input_graph)
    print(f"{graph_name}: |V|={G.num_vertices()}, |E|={G.num_edges()}")

    start = time.perf_counter()
    from modules.graph_metrics import apsp 
    d = apsp(G)

    from modules.L2G import find_neighbors 
    w = find_neighbors(G,k=k,a=alpha)

    print("Starting optimization")
    from modules.cython_l2g import L2G_opt
    X = L2G_opt(d,w,n_iter=max_iter,eps=eps)

    print(f"Complete! Layout took {time.perf_counter()-start :6.4f} seconds")

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    gt.graph_draw(G,pos=pos,output=args.output)