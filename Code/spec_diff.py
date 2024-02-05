import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

def spectral_diff( N, a=-1, b=1 ):
    [J, I] = np.meshgrid( np.arange(0, N, 1), np.arange(0, N, 1) )
    diags = np.eye(N, dtype=bool)

    cheb = np.cos( np.pi * np.arange(0, N, 1) / (N-1) )
    cheb = a + (b-a)*(cheb+1)/2

    # construct off diagonal
    spec_mat = (-1)**(I + J) / ( cheb[I] - cheb[J] + np.eye(N) )

    # fix 4 edges
    spec_mat[:,0] /= 2
    spec_mat[:,-1] /= 2
    spec_mat[0,:] *= 2
    spec_mat[-1,:] *= 2

    #sets the diagonal values to zero
    spec_mat[diags] = 0
    #then adds up the rest of the row and sets the negative of that as the diagonal
    spec_mat[diags] = -np.sum(spec_mat, axis=1)

    # replace 2 corners
    #spec_mat[0][0] = (2*(N-1)**2 + 1 )/6
    #spec_mat[-1][-1] = -spec_mat[0][0]

    return cheb, spec_mat


if __name__ == "__main__":
    f = lambda x: 1 / ( 1 + 16*x**2 )
    fp = lambda x: -32*x/( 1 + 16*x**2 )**2

    #for different numbers of poitns
    Ns = [4, 8, 16, 32, 64, 128, 256]
    err = np.zeros_like( Ns, dtype=float )
    #loops through our different number of points
    for i in range(len(Ns)):
        cheb, spec_mat = spectral_diff( Ns[i] )
        #determines the errors for each Ns value
        err[i] = np.linalg.norm( fp( cheb ) - (spec_mat @ f( cheb )) )
    
    plt.semilogy( Ns, err )
    plt.show()
