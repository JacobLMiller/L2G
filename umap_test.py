from umap import UMAP
from hd_data.load_data import load_data 
import numpy as np 
import pylab as plt 
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize_scalar

from sklearn.model_selection import ParameterGrid

grid = {'n_components': [2], 
        'random_state': [42],
        'n_neighbors': [5, 10, 15], 
        'metric': ['euclidean'], 
        'init': ['spectral', 'random'], 
        'min_dist': [0.001, 0.01, 0.1, 0.5], 
        'spread': [1.0], 
        'angular_rp_forest': [False]}

params = ParameterGrid(grid)

data,y = load_data("imdb")


for i,p in enumerate(params):
    print(p)
    X = UMAP(**p).fit_transform(data)
    plt.scatter(X[:,0],X[:,1],c=y,alpha=0.6)
    plt.savefig(f"drawings/umapimdb_{i}.png")
    plt.clf()


# choice = np.random.choice(data.shape[0],size=500,replace=False)
# data = data[choice]
# y = y[choice]

# def embed(a=2):
#     X = UMAP(n_neighbors=2,min_dist=0.1,spread=1).fit_transform(data)
#     return KNeighborsClassifier().fit(X,y).score(X,y)

# res = minimize_scalar(embed,bounds=(2,100),method="Bounded")

# for n in [5,10,15]:
#     for m in [0.001,0.01,0.1,0.5]:
#         print(f"n_neighbors={n}; min_dist={m}")
#         X = UMAP(n,min_dist=m,random_state=42,init="random").fit_transform(data)
#         plt.scatter(X[:,0],X[:,1],c=y,alpha=0.6)
#         plt.savefig(f"drawings/umapimdb_random_{n}_{m}.png")
#         plt.clf()


# X = np.loadtxt("data/proj_X_imdb_UMAP.csv",delimiter=",")

