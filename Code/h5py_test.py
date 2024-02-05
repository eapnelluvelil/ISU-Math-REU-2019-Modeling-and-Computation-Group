import numpy as np
import h5py
import scipy.sparse.linalg as ssl
import radon_basis_vec as rbv
import radon_transform as rtv1
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    Ns = 10
    Nw = 10
    Nq = 10

    a = 0.2
    b = 0.05

    f = lambda x, y: np.exp(-(x/a)**2 - (y/b)**2)

    fw = 2
    fs = 3

    s = rtv1.clen_curt(Ns)[0]
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)

    f_vec = f(S*np.cos(W), S*np.sin(W))
    f_vec = f_vec.flatten()

    f_hat = rbv.radon(fw, fs, Ns, Nw, Nq)

    #get the matrix
    C1 = rbv.radon(fw, fs, Ns, Nw, Nq)
    #print(C)
    C = C1.flatten()

    #save the matrix using h5py
    hf = h5py.File('data.h5', 'w')
    dataset = hf.create_dataset('oeltjenbrunsduke', data = C, track_order = True, fletcher32 = True, dtype = float)
    hf.close()

    #pull data from file
    D1 = np.fromfile('data.h5', dtype = float)
    D = D1.reshape((-1, 1))

    print(C)
    print(D)
