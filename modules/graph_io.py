import graph_tool.all as gt 
import numpy as np 
import matplotlib.pyplot as plt 

def std_draw(G,X,output=None):
    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    gt.graph_draw(G,pos,output=output)

def draw_tsnet_like(G,X,output=None):
    edge_lengths = np.array( [np.linalg.norm(X[i]-X[j]) for i,j in G.iter_edges()] )
    edge_lengths -= np.min(edge_lengths)
    edge_lengths /= np.max(edge_lengths)
    jet_map = plt.get_cmap("jet_r")
    e_clrs = G.new_ep("vector<float>",vals=[jet_map(dist) for dist in edge_lengths])

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)

    gt.graph_draw(G,pos=pos,vertex_fill_color="white",vertex_size=0.00001,edge_color=e_clrs,edge_pen_width=1,output=output)

def plot_data(X,y,alpha=0.6,output=None,title=""):
    fig,ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1],c=y,alpha=alpha)
    ax.axis("off")
    fig.suptitle(title)

    if output:
        fig.savefig(output)
    else:
        plt.show()
    return fig,ax

def read_edgelist(fpath,header=False,delimiter=" "):
    E = np.loadtxt(fpath,delimiter=delimiter,skiprows=1 if header else 0)
    G = gt.Graph(directed=False)
    G.add_edge_list([(u,v) for u,v in E])
    return G

def write_edgelist(G,fpath,delimiter=" "):
    E = np.array([(u,v,1) for u,v in G.iter_edges()])
    header = f"{G.num_vertices()} {G.num_edges()}"
    np.savetxt(fpath,E,delimiter=delimiter,fmt="%d",header=header,comments="")

def read_cids(Gname):
    import json 
    with open(f'cluster_ids/{Gname}.json') as json_file:
        data = json.load(json_file)
    json_file.close()
    return [set(d) for d in data]

def get_table_graphs():
    import os
    path = 'table_graphs/'
    graph_paths = os.listdir(path)

    graphs = [(gt.load_graph(f"{path+graph}"),graph) for graph in graph_paths]
    graphs = sorted(graphs,key=lambda x: x[0].num_vertices())
    return graphs
