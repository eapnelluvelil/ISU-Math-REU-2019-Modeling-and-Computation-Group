import numpy as np
import h5py
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def clen_curt(N):
    # Make up for the difference from MATLAB code
    #should probably go through each N value and fix instead of doing this
    N = N - 1

    # Do the FFT things described in Trefethen
    theta = np.pi * np.arange(0, N+1, 1) / (N)
    x = np.cos(theta)
    w = np.zeros((N+1,))
    ii = np.arange(1, N)
    v = np.ones((N-1,))

    if N % 2 == 0:
        w[0] = 1/(N**2-1)
        w[N] = w[0]

        for k in range (1, N//2):
            v = v - 2*np.cos(2*k*theta[ii])/(4*k**2-1)
        v = v - np.cos(N*theta[ii])/(N**2-1)
    else:
        w[0] = 1/(N**2)
        w[N] = w[0]

        for k in range (1, (N-1)//2 + 1):
            v = v - 2*np.cos(2*k*theta[ii])/(4*k**2-1)

    w[ii] = 2*v/N

    return x, w

def cc_quad(f, a, b, N):
    # Used in convergence test for Clenshaw-Curtis
    x, w = clen_curt(N)
    s = lambda x: (b+a)/2 + (b-a)/2*x
    approx_int = sum(w*f(s(x)))*(b-a)*0.5

    return approx_int

def radon( f, Ns, Nw, Nd ):
    """
    Returns radon transform of f at points in [0,pi]x[-1,1]
    """
    # Get Chebyshev points for discretization along each angle
    ss = clen_curt( Ns )[0]

    # Get Chebyshev points and weights for quadrature
    s, weights = clen_curt( Nd )
    w = np.linspace(0, np.pi, Nw+1)[:-1]

    # Create meshgrid of angles and radii (for plotting purposes, mostly)
    [S, W] = np.meshgrid( ss, w )

    # Somewhere to put the values for the radon transform
    f_hat = np.zeros_like( S )

    # fig1 = plt.figure(1)
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.plot_wireframe( S*np.cos(W), S*np.sin(W), f(S*np.cos(W), S*np.sin(W)) )

    """ Old code, probably slower
    # Loop over all the points in the grid
    for i in range(Ns):
        for j in range(Nw):
            # compute the new range of integration based on chord length
            # ( only does integration from [-r, r] )
            r = np.sqrt(1 - S[i,j]*S[i,j])

            # use following substitution: z = rt, dz = r dt
            #   so that integration is only from -1 to 1
            f_hat[i,j] = r*np.sum( weights*f( S[i,j]*np.cos(W[i,j]) - r*s*np.sin(W[i,j]),
                                              S[i,j]*np.sin(W[i,j]) + r*s*np.cos(W[i,j]) ) )
    """

    # Loop over all the quadrature points
    r = np.sqrt( 1 - S*S )
    for i in range(Nd):
        f_hat += r*weights[i]*f( S*np.cos(W) - r*s[i]*np.sin(W),
                                 S*np.sin(W) + r*s[i]*np.cos(W) )

    # fig2 = plt.figure(2)
    # ax2 = fig2.add_subplot(111, projection='3d')
    # ax2.plot_wireframe(S, W, f_hat)
    # plt.show()

    return f_hat


if __name__ == "__main__":
    Ns = 200
    Nw = 200
    Nd = 75

    #"""
    #creates a function for a unit square
    def unit_square( x, y ):
        return np.array( np.greater(x, 0) & np.greater(y, 0) & np.less(x, 0.5) &
                        np.less( y, 0.5), dtype=float )

    f = lambda x, y: np.exp( -10*(x*x + y*y) )
    #"""

    N = 100
    x, w = clen_curt(N)
    #f = lambda x: np.cos(x)
    #print(sum(w*f(x)))
    a, b = -np.pi/2, np.pi/2

    f_hat = radon( f, Ns, Nw, Nd )
    #print(f_hat.shape)
    #creates subgroups that function like subfolders
    hf = h5py.File('radon.h5', 'a')
    grp1 = hf.create_group('subgroup')
    #creates and writes to the file
    dset = grp1.create_dataset('test7', data = f_hat, shape = f_hat.shape)
    hf.close()
    #Note: the name of the dataset must change everytime you run the code.
    #aka: don't run the code more than you have to

    exact = 2

    #Ns = [4, 8, 16, 32, 64, 128, 256]
    #errs = []
    #for N in Ns:
    #    approx = cc_quad( f, a, b, N )
    #    print( approx )
    #    errs.append( abs(approx - exact) )

    #print( errs )
    #A = np.zeros( [len(Ns), 2] )
    #A[:,0] = 1
    #A[:,1] = np.log( Ns )

    #[logc, k] = np.linalg.solve( A.T @ A, A.T @ np.log( errs ) )
    #print( errs )
    #plt.loglog( Ns, errs)# label='{:.2f}'.format(k) )
    #plt.show()
    #"""
