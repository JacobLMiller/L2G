import numpy as np 

def load_bank():
    X = np.load("hd_data/bank_data.npy")
    y = np.load("hd_data/bank_labels.npy")
    return X,y

def load_fashion():
    X = np.loadtxt("hd_data/fashion-mnist_test.csv",skiprows=1,delimiter=",")
    y = X[:,0]
    return X[:,1:], y

def load_espadato(dname: str):
    s = f"hd_data/{dname}/"
    X = np.load(s+"X.npy")
    y = np.load(s+'y.npy')
    return X,y

def load_data(dname: str):
    match dname:
        case "bank":
            return load_bank()
        case "fashion-mnist":
            return load_fashion()
        case _:
            return load_espadato(dname)