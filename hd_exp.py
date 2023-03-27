import numpy as np 
import pylab as plt
from sklearn.metrics import pairwise_distances
from modules.metrics import get_stress, chen_neighborhood, cluster_distance
from hd_data.load_data import load_data

from sklearn.neighbors import KNeighborsClassifier

data_name = "imdb"

def knn_accuracy(X,y,k=7):
    return KNeighborsClassifier(n_neighbors=k).fit(X,y).score(X,y)


data, C = load_data(data_name)
print(f"Data shape is {data.shape}")
choice = np.random.choice(data.shape[0],size=500,replace=False)
data = data[choice]
C = C[choice]
c_ids = C.astype(int)

data /= np.max(data,axis=0)

from sklearn.decomposition import PCA

data = PCA(50).fit_transform(data)


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

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(data[:,0],data[:,1],data[:,2],c=C)
# # fig.savefig("figures/3d-og.png")
# plt.show()
# plt.clf()


d = pairwise_distances(data)
from modules.cython_l2g import L2G_opt, standard_mds


from sklearn.manifold import TSNE

fig, ax = plt.subplots()
X = TSNE(perplexity=20,init="pca",learning_rate="auto").fit_transform(data)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

legend1 = ax.legend(*scatter.legend_elements(),title="Classes", loc="lower right")
ax.add_artist(legend1)

s,n,c = print_metrics(X,d, data,c_ids)
plt.suptitle(f"t-SNE {data_name}")
plt.savefig(f"figures/{data_name}-tsne.png")
t_scores = (s,n,c)
np.savetxt(f"data/high_dim/tsne_{data_name}.txt",np.array((s,n,c)))

plt.clf()

print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160]])

from sklearn.manifold import MDS
fig, ax = plt.subplots()

X = MDS(dissimilarity="precomputed",n_init=1).fit_transform(d)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

s,n,c = print_metrics(X,d, data,c_ids)
plt.suptitle(f"MDS")
plt.savefig(f"figures/{data_name}-mds.png")
m_scores = (s,n,c)
np.savetxt(f"data/high_dim/mds_{data_name}.txt",np.array((s,n,c)))

plt.clf()

print([knn_accuracy(X,C,n=n) for n in [10,20,40,80,160]])


import umap
fig, ax = plt.subplots()

X = umap.UMAP().fit_transform(data)
scatter = ax.scatter(X[:,0],X[:,1],c=C,alpha=0.6)
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

s,n,c = print_metrics(X,d, data,c_ids)
plt.suptitle(f"UMAP")
plt.savefig(f"figures/{data_name}-umap.png")
u_scores = (s,n,c)
np.savetxt(f"data/high_dim/umap_{data_name}.txt",np.array((s,n,c)))
plt.clf()

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
from sklearn.manifold import TSNE


stress, nh,cd = list(), list(), list()
K = [2,4,8,16,32,64,100,110,120,130,140,150,200,300,500]
record = np.zeros()
for k in K:
    fig, ax = plt.subplots()

    print(k)
    w = diffusion_weights(d,k=k)
    w = w.astype(np.int16)

    n = w.shape[0]
    G = gt.Graph(directed=False)
    G.add_vertex(n)
    for i in range(n):
        for j in range(i):
            if w[i,j] == 1: G.add_edge(i,j)
    wd = apsp(G)

    G,dists = gt.generate_knn(data,k=2)

    # X = L2G(G,k,a=5)
    d = d.astype("double")
    X = L2G_opt(d,w)
    # X = Xw

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
    plt.suptitle(f"L2G({k})")
    plt.savefig(f"figures/{data_name}_l2g_{k}.png")

    plt.clf()




x = K
fig, axes = plt.subplots(1,3)
for i,(ax, metric) in enumerate(zip(axes,[stress,nh,cd])):
    ax.plot(x,metric, 'o-', label="L2G w/knn stress")
    ax.plot(x,[m_scores[i]] * len(x), '--', label="MDS_stress")
    ax.plot(x,[t_scores[i]] * len(x), '--', label="tsne_stress")
    ax.plot(x,[u_scores[i]] * len(x), '--', label="umap_stress")
    fig.suptitle("Stress scores of DR algs")
    ax.legend()
    fig.savefig(f"figures/{data_name}_compare.png")

fig.show()
