import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convergence rates
import radon_transform_v1 as rtv1

"""

"""
def clen_curt(N):
    # Make up for the difference from MATLAB code
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

"""
"""
def radon( f, Ns, Nw, Nq ):
    """
    Returns radon transform of f at points in [0,pi]x[-1,1]
    """
    # Get Chebyshev points for discretization along each angle
    s = clen_curt( Ns )[0]

    # Get Chebyshev points and weights for quadrature
    nodes, weights = clen_curt( Nq )

    # Create equispaced angles
    w = np.linspace( 0, np.pi, Nw )

    # Create meshgrid of S and W (for plotting purposes mostly)
    [S, W] = np.meshgrid( s, w )

    # Somewhere to put the values for the radon transform
    f_hat = np.zeros_like( S ).T

    """
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_wireframe( S*np.cos(W), S*np.sin(W), f(S*np.cos(W), S*np.sin(W)) )
    """

    # Plot the physical space grid i.e.
    # the unit disk discretized at equispaced angles and radii
    # plt.figure(1)
    #plt.plot( S*np.cos(W), S*np.sin(W), 'k.')
    # plt.figure(2)
    #plt.plot( S, W, 'k.' )
    #plt.show()
    # print( "w:", w )
    # print( "s:", s )

    for i in range( Ns ): # loop over s in grid 
        for j in range( Nw ): # loop over omega in grid
            for k in range( Nq ): # loop over quadrature points
                r = np.sqrt( 1 - s[i]*s[i] )
                x0 = s[i]*np.cos(w[j]) - r*nodes[k]*np.sin(w[j])
                y0 = s[i]*np.sin(w[j]) + r*nodes[k]*np.cos(w[j])

                s0 = (1-1e-15)*np.sqrt( x0*x0 + y0*y0 )*np.sign(y0)
                w0 = np.arctan2( y0, x0 ) + np.pi*(y0 < 0)

                # Bin the quadrature point between the correct angles
                # and radii
                si = np.digitize( s0, s, right=True )
                wi = np.digitize( w0, w, right=True )

                xs = []
                ys = []
                for ii in range(2):
                    for jj in range(2):
                        xs.append( s[si-ii]*np.cos(w[wi-jj]) )
                        ys.append( s[si-ii]*np.sin(w[wi-jj]) )

                xs = np.array( [xs] )
                ys = np.array( [ys] )
                fs = f( xs, ys )[0]
                #print( "fs", fs )
                #plt.plot( xs, ys, 'r.' )
                #plt.plot( x0, y0, 'b*')

                # Perform radial basis function interpolation
                A = np.sqrt( (xs-xs.T)**2 + (ys-ys.T)**2 )
                sigma = 0.5
                phi = lambda x: -np.exp( x / sigma**2 )
                A = phi( A )

                rbf_weights = np.linalg.solve( A, fs )
                #print( rbf_weights )
                f_hat[i,j] += r*weights[k]*( np.dot( rbf_weights, phi( np.sqrt(
                    (xs[0]-x0)**2 + (ys[0]-y0)**2 ) ) ) )
                #print( np.dot( rbf_weights, phi( np.sqrt(
                #    (xs[0]-x0)**2 + (ys[0]-y0)**2 ) ) ) )
                # Plot the quadrature point in physical space
                # and in s-omega space
                # plt.figure(1)
                # plt.plot(s0*np.cos(w0), s0*np.sin(w0), 'b.')
                # plt.figure(2)
                # plt.plot( s0, w0, 'b.' )

    #plt.show()
    # Plot the wireframe of the approximate Radon transform
    # (mostly for testing purposes)
    # plt.show()
    #fig2 = plt.figure(2)
    #ax2 = fig2.add_subplot(111, projection='3d')
    #ax2.plot_wireframe(S, W, f_hat.T)   # Tranpose to get the right wireframe
    # plt.show()
    #ax2 = fig2.add_subplot(111)
    #ax2.contourf(S, W, f_hat.T)
    #print( f_hat )
    #plt.show()
    return f_hat.T

if __name__ == "__main__":
    def unit_square( x, y ):
        return np.array( np.greater(x, 0) & np.greater(y, 0) & np.less(x, 0.5) &
                        np.less( y, 0.5), dtype=float )

    # Testing convergence rates
    Ns_vals = [50, 100, 200]
    Nw = 50
    Nd = 100

    alpha = 40
    f = lambda x, y: np.exp( -alpha*(x*x + y*y) )
    #f_hat_v2 = radon( f, Ns_vals[0], Nw, Nd )
    errs = []


    for Ns in Ns_vals:
        # Create dummy solution to get the right shape
        f_hat_dummy = radon( f, Ns, Nw, 3 )

        # Create grid
        ss = clen_curt( Ns )[0]
        w = np.linspace( 0, np.pi, Nw )

        # Create meshgrid of S and W (for plotting purposes mostly)
        [S, W] = np.meshgrid( ss, w )

        # Create exact solution
        f_hat_exact = np.zeros_like( f_hat_dummy )
        f_hat_exact[:, :] = np.sqrt(np.pi)/np.sqrt(alpha) * np.exp(-alpha*(ss*ss).T)

        # f_hat_v1 = rtv1.radon( f, Ns, Nw, 300 )
        f_hat_v2 = radon( f, Ns, Nw, Nd )

        #print( f_hat_exact[:5,:5] )
        #print( f_hat_v2[:5,:5] )


        plt.figure(1)
        plt.contourf(S, W, f_hat_exact)
        plt.colorbar()

        plt.figure(2)
        plt.contourf(S, W, f_hat_v2)
        plt.colorbar()

        plt.figure(3)
        plt.contourf(S, W, np.absolute(f_hat_exact-f_hat_v2))
        plt.colorbar()

        plt.show()

        print(np.max(np.absolute(f_hat_exact - f_hat_v2)))
        errs.append(np.max(np.absolute(f_hat_exact - f_hat_v2)))

    print(errs)
    """

    # s, weights = clen_curt( Ns )
    # w = np.linspace(0, np.pi, Nw)
    """
    """
    N = 100
    x, w = clen_curt(N)
    f = lambda x: x*x
    #print(sum(w*f(x)))
    a, b = 1, 3

    exact = 9 - 1/3

    Ns = [5, 10, 20, 40, 80, 160]
    errs = []
    for N in Ns:
        approx = cc_quad( f, a, b, N )
        print( approx )
        errs.append( abs(approx - exact) )

    A = np.zeros( [len(Ns), 2] )
    A[:,0] = 1
    A[:,1] = np.log( Ns )

    #[logc, k] = np.linalg.solve( A.T @ A, A.T @ np.log( errs ) )
    #print( errs )
    plt.loglog( Ns, errs)# label='{:.2f}'.format(k) )
    plt.show()
    """


