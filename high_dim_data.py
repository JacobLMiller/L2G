import numpy as np 
import pylab
from sklearn.metrics import pairwise_distances
from modules.metrics import get_stress, chen_neighborhood, cluster_distance, cluster_preservation, cluster_preservation2, mahalonobis_metric


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
data_name = "3d"
c_ids = C


# data = np.loadtxt("hd_data/fashion-mnist_test.csv",skiprows=1,delimiter=",")
# # data = data[np.logical_or(data[:,0] == 0, np.logical_or(data[:,0] == 5, data[:,0] == 8))]

# choice = np.random.choice(data.shape[0],size=1000,replace=False)
# data = data[choice]
# data.shape
# C = data[:,0]
# c_ids = C.astype(int)
# data = data[:,1:]
# data /= 255
# data_name = "fashion-mnist"



d = pairwise_distances(data)
tmp = [[] for _ in np.unique(c_ids)]
unq,inv = np.unique(c_ids,return_inverse=True)
print(inv)
[tmp[inv[i]].append(i) for i,c in enumerate(c_ids)]
c_ids = tmp

# data = list()
# for _ in range(1000):
#     vec = np.random.normal(0,1,size=(3))
#     vec /= np.linalg.norm(vec)
#     data.append(vec)
    
# data = np.array(data)

fig = pylab.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2],c=C)
# fig.savefig("figures/3d-og.png")
pylab.show()
pylab.clf()


d = pairwise_distances(data)
from modules.cython_l2g import L2G_opt, standard_mds


from sklearn.manifold import TSNE
X = TSNE(perplexity=20,init="pca",learning_rate="auto").fit_transform(data)
pylab.scatter(X[:,0],X[:,1],c=C,alpha=0.5)
s,n,c = print_metrics(X,d, data,c_ids)
pylab.suptitle(f"t-sne\n stress: {s}\nNH: {n}")
pylab.savefig(f"figures/{data_name}-tsne.png")
t_scores = (s,n,c)
pylab.clf()


from sklearn.manifold import MDS
X = MDS(dissimilarity="precomputed",n_init=1).fit_transform(d)
pylab.scatter(X[:,0],X[:,1],c=C,alpha=0.5)
s,n,c = print_metrics(X,d,data,c_ids)
pylab.suptitle(f"mds\n stress: {s}\nNH: {n}")
pylab.savefig(f"figures/{data_name}-mds.png")
m_scores = (s,n,c)
pylab.clf()

import umap
X = umap.UMAP().fit_transform(data)
pylab.scatter(X[:,0],X[:,1],c=C,alpha=0.5)
s,n,c = print_metrics(X,d,data,c_ids)
pylab.suptitle(f"umap\n stress: {s}\nNH: {n}")
pylab.savefig(f"figures/{data_name}-umap.png")
u_scores = (s,n,c)
pylab.clf()


def k_nearest(d,k=7):
    
    w = np.zeros_like(d)
    for i in range(w.shape[0]):
        ind = set( np.argsort(d[i])[1:k+1] )
        for j in ind: 
            w[i][j] = 1
            w[j][i] = 1 
    
    return w

stress, nh,cd = list(), list(), list()
K = [8,16,32,64,100,110,120,130,140,150,200,300,500]
for k in K:
    print(k)
    w = k_nearest(d,k=k)
    w = w.astype(np.int16)
    X = L2G_opt(d,w)
    pylab.scatter(X[:,0],X[:,1],c=C,alpha=0.5)
    s,n,c = print_metrics(X,d,data,c_ids)
    stress.append(s)
    nh.append(n)
    cd.append(c)
    pylab.suptitle(f"l2g: knn search (naive); k = {k}\n stress: {s}\nNH: {n}")
    pylab.savefig(f"figures/{data_name}-l2gk{k}.png")
    pylab.clf()




x = K
pylab.plot(x,stress,'o-',label="stress")
pylab.plot(x,nh,'o-',label="nh")
pylab.plot(x,cd,'o-',label="cluster_distance")
pylab.suptitle("Stress, nh for l2g w/knn on simple 3d data")
pylab.legend()
pylab.savefig(f"figures/{data_name}_naivecurve.png")
pylab.clf()


pylab.plot(x,stress, 'o-', label="L2G w/knn stress")
pylab.plot(x,[m_scores[0]] * len(x), '--', label="MDS_stress")
pylab.plot(x,[t_scores[0]] * len(x), '--', label="tsne_stress")
pylab.plot(x,[u_scores[0]] * len(x), '--', label="umap_stress")
pylab.suptitle("Stress scores of DR algs")
pylab.legend()
pylab.savefig(f"figures/{data_name}_stress_compare.png")
pylab.clf()


pylab.plot(x,nh, 'o-', label="L2G w/knn NH")
pylab.plot(x,[m_scores[1]] * len(x), '--', label="MDS_NH")
pylab.plot(x,[t_scores[1]] * len(x), '--', label="tsne_NH")
pylab.plot(x,[u_scores[1]] * len(x), '--', label="umap_NH")
pylab.suptitle("NH scores of DR algs")
pylab.legend()
pylab.savefig(f"figures/{data_name}_nh_compare.png")
pylab.clf()

pylab.plot(x,cd, 'o-', label="L2G w/knn cluster distance")
pylab.plot(x,[m_scores[2]] * len(x), '--', label="MDS_cd")
pylab.plot(x,[t_scores[2]] * len(x), '--', label="tsne_cd")
pylab.plot(x,[u_scores[2]] * len(x), '--', label="umap_cd")
pylab.suptitle("cd scores of DR algs")
pylab.legend()
pylab.savefig(f"figures/{data_name}_cd_compare.png")
pylab.clf()
