import numpy as np 
import pylab
from sklearn.metrics import pairwise_distances
from modules.metrics import get_stress, chen_neighborhood, cluster_distance, cluster_preservation, cluster_preservation2, mahalonobis_metric
from hd_data.load_data import load_data

from sklearn.neighbors import KNeighborsClassifier

data_name = "har"

def knn_accuracy(X,y,n=5):
    return KNeighborsClassifier(n_neighbors=n).fit(X,y).score(X,y)


def print_metrics(X,d,H,c_ids):
    stress = get_stress(X,d)
    NH = 1-chen_neighborhood(d,X,k=7)
    cd = cluster_distance(H,X,c_ids)
    print(f"Stress is {stress}; NH is {NH}; CD is {cd}")
    return stress, NH, cd

c1 = list()
for i in range(5):
    for j in range(5):
        for k in range(5):
            c1.append([i,j,k])

grid = np.array(c1)


gen_cluster = lambda s: np.random.normal(scale=s, size=[np.random.randint(50,150),3])

stretch = 3

c1 = (grid*0.1) + np.array([0,0,0]) * stretch
c2 = gen_cluster(0.1) + np.array([0,0,1]) * stretch
c3 = (grid*0.25) + np.array([0,1,0]) * stretch
c4 = gen_cluster(0.1) + np.array([0,1,1]) * stretch
c5 = gen_cluster(0.25) + np.array([1,0,0]) * stretch
c6 = gen_cluster(0.5) + np.array([1,0,1]) * stretch
c7 = (grid*0.25) + np.array([1,1,0]) * stretch
c8 = (grid*0.5) + np.array([1,1,1]) * stretch

c9 = (grid*0.25) + (np.array([4,4,4]) / 8) * stretch

"""
Arrange 3 clusters on a triangle, take two other clusters raise one higher, lower the other
Look carefully that no 4 are aligned in any projection


Imagine a 3-d cube, create 8 clusters on the 8 corners of the cube, +1 in the center
Then, replace some clusters with 3d shapes

"""

clusters = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
# clusters = [c*10 for c in clusters]

data = np.concatenate(clusters,axis=0)
data.shape

def label_clusters(sizes):
    return sum([[i] * size for i,size in enumerate(sizes)], [])
sizes = [c.shape[0] for c in clusters]
C = label_clusters(sizes)
# data_name = "3d"
c_ids = C


# data = np.loadtxt("hd_data/fashion-mnist_test.csv",skiprows=1,delimiter=",")
# # data = data[np.logical_or(data[:,0] == 0, np.logical_or(data[:,0] == 1, data[:,0] == 7))]
# mask = {0,1,7,9,4}
# data = data[ [True if data[i,0] in mask else False for i in range(data.shape[0])] ]

# choice = np.random.choice(data.shape[0],size=500,replace=False)
# data = data[choice]
# data.shape
# C = data[:,0]
# c_ids = C.astype(int)
# data = data[:,1:]
# data /= 255
# data_name = "fashion-mnist"

data, C = load_data(data_name)
print(data.shape)
# choice = np.random.choice(data.shape[0],size=1000,replace=False)
# data = data[choice]
# C = C[choice]
c_ids = C.astype(int)


d = pairwise_distances(data)
tmp = [[] for _ in np.unique(c_ids)]
unq,inv = np.unique(c_ids,return_inverse=True)
[tmp[inv[i]].append(i) for i,c in enumerate(c_ids)]
c_ids = tmp
print([len(c) for c in c_ids])

# data = list()
# for _ in range(1000):
#     vec = np.random.normal(0,1,size=(3))
#     vec /= np.linalg.norm(vec)
#     data.append(vec)
    
# data = np.array(data)

# fig = pylab.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data[:,0],data[:,1],data[:,2],c=C)
# # fig.savefig("figures/3d-og.png")
# pylab.show()
# pylab.clf()


d = pairwise_distances(data)
from modules.cython_l2g import L2G_opt, standard_mds


from sklearn.manifold import TSNE

fig, ax = pylab.subplots()
X = TSNE(perplexity=20,init="pca",learning_rate="auto").fit_transform(data)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

legend1 = ax.legend(*scatter.legend_elements(),title="Classes", loc="lower right")
ax.add_artist(legend1)

s,n,c = print_metrics(X,d, data,c_ids)
pylab.suptitle(f"t-SNE {data_name}")
pylab.savefig(f"figures/{data_name}-tsne.png")
t_scores = (s,n,c)
pylab.clf()

print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160]])

from sklearn.manifold import MDS
fig, ax = pylab.subplots()

X = MDS(dissimilarity="precomputed",n_init=1).fit_transform(d)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

s,n,c = print_metrics(X,d, data,c_ids)
pylab.suptitle(f"MDS")
pylab.savefig(f"figures/{data_name}-mds.png")
m_scores = (s,n,c)
pylab.clf()

print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160]])


import umap
fig, ax = pylab.subplots()

X = umap.UMAP().fit_transform(data)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

s,n,c = print_metrics(X,d, data,c_ids)
pylab.suptitle(f"UMAP")
pylab.savefig(f"figures/{data_name}-umap.png")
u_scores = (s,n,c)
pylab.clf()

print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160]])


def k_nearest(d,k=7):
    
    w = np.zeros_like(d)
    for i in range(w.shape[0]):
        ind = set( np.argsort(d[i])[1:k+1] )
        for j in ind: 
            w[i][j] = 1
            w[j][i] = 1 
    
    return w

def diffusion_weights(d,a=5, k = 20, sigma=1):
    #Transform distance matrix
    diff = np.exp( -(d**2) / (sigma **2) )
    diff /= np.sum(diff,axis=0)

    #Sum powers from 1 to a
    mp = np.linalg.matrix_power
    A = sum( pow(0.05,i) * mp(diff,i) for i in range(1,a+1) )

    #Find k largest points for each row 
    Neighbors = set()
    for i in range(diff.shape[0]):
        args = np.argsort(A[i])[::-1][1:k+1]
        for j in args:
            Neighbors.add( (int(i),int(j)) )

    #Set pairs to 1
    w = np.zeros_like(diff)
    for i,j in Neighbors:
        w[i,j] = 1
        w[j,i] = 1
    return w


import graph_tool.all as gt
from modules.graph_metrics import apsp
from modules.L2G import L2G

stress, nh,cd = list(), list(), list()
K = [2,4,8,16,32,64,100,110,120,130,140,150,200,300,500]
for k in K:
    fig, ax = pylab.subplots()

    print(k)
    w = k_nearest(d,k=k)
    w = w.astype(np.int16)

    n = w.shape[0]
    G = gt.Graph(directed=False)
    G.add_vertex(n)
    for i in range(n):
        for j in range(i):
            if w[i,j] == 1: G.add_edge(i,j)
    wd = apsp(G)
    # Xw = TSNE(metric="precomputed").fit_transform(wd)

    G,dists = gt.generate_knn(data,k=7)
    arr_dists = dists.a 
    dists.a = np.exp( -(arr_dists**2) / (1 **2) )
    X = L2G(G,k,alpha=0.1)


    # X = L2G_opt(wd,w,n_iter=200,alpha=0.6)
    # X = Xw
    print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160, 250]])

    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    gt.graph_draw(G,pos,output=f"figures/{data_name}_graph_{k}.png")

    s,n,c = print_metrics(X,d,data,c_ids)
    stress.append(s)
    nh.append(n)
    cd.append(c)
    scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # legend1 = ax.legend(*scatter.legend_elements(),title="Classes", location="lower right")
    # ax.add_artist(legend1)
    pylab.suptitle(f"L2G({k})")
    pylab.savefig(f"figures/{data_name}_l2g_{k}.png")

    pylab.clf()




x = K
fig, axes = pylab.subplots(1,3)
for i,(ax, metric) in enumerate(zip(axes,[stress,nh,cd])):
    ax.plot(x,metric, 'o-', label="L2G w/knn stress")
    ax.plot(x,[m_scores[i]] * len(x), '--', label="MDS_stress")
    ax.plot(x,[t_scores[i]] * len(x), '--', label="tsne_stress")
    ax.plot(x,[u_scores[i]] * len(x), '--', label="umap_stress")
    fig.suptitle("Stress scores of DR algs")
    ax.legend()
    fig.savefig(f"figures/{data_name}_compare.png")

fig.show()
