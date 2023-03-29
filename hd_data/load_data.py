import numpy as np 

def sample_data(X: np.array, y: np.array, n_samples=500):
    choice = np.random.choice(X.shape[0],size=n_samples,replace=False)
    return X[choice], y[choice]

def normalize(X: np.array):
    return X / np.max(X,axis=0)

def load_bank():
    X = np.load("hd_data/bank_data.npy")
    y = np.load("hd_data/bank_labels.npy")
    return X,y

def load_fashion():
    X = np.loadtxt("hd_data/fashion-mnist_test.csv",skiprows=1,delimiter=",")
    y = X[:,0]
    return X[:,1:], y

def load_mnist():
    X = np.loadtxt("hd_data/mnist_test.csv",skiprows=1,delimiter=",")
    y = X[:,0]
    return X[:,1:], y

def load_espadato(dname: str):
    s = f"hd_data/{dname}/"
    X = np.load(s+"X.npy")
    y = np.load(s+'y.npy')
    return X,y

def load_data(dname: str):
    if dname == "fashion-mnist":
        return load_fashion()
    elif dname == "mnist":
        return load_mnist()
    else:
        return load_espadato(dname)