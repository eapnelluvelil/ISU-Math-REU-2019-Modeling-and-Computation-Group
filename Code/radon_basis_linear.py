import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import radon_transform as rtv1

def radon(fw, fs, Ns, Nw, Nq):
    # Get radii discretization
    s = rtv1.clen_curt(Ns)[0]

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])
    Np = Ns*Nw
    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = rtv1.clen_curt(Nq)

    ff = 1e-15

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w)

    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    ws = w[np.repeat(np.arange(0, Nw, dtype=int), Ns)].reshape((-1, 1))
    ss = s[np.tile(np.arange(0, Ns, dtype=int), Nw)].reshape((-1, 1))

    rs = np.sqrt( 1 - ss*ss ).reshape((-1, 1))*(1 - ff)

    x_quads = ss*np.cos(ws) - rs*nodes.reshape((1, -1))*np.sin(ws)
    y_quads = ss*np.sin(ws) + rs*nodes.reshape((1, -1))*np.cos(ws)

    x_quads = x_quads.reshape((-1, ))
    y_quads = y_quads.reshape((-1, ))

    w_quads = np.arctan2( y_quads, x_quads )
    s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

    s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
    w_quads = np.mod( w_quads, 2*np.pi)

    w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

    #w_indexes[ np.isclose(y_quads, 0, atol=ff) ] = Nw
    #w_quads[ np.isclose(y_quads, 0, atol=ff) & (x_quads < 0) ] = np.pi

    #plt.plot( S*np.cos(W), S*np.sin(W), 'k.')
    #plt.plot( x_quads[np.isclose(y_quads, 0, atol=ff)], y_quads[np.isclose(y_quads, 0, atol=ff)], 'g.' )
    #plt.plot( x_quads[25311], y_quads[25311], 'y*' )
    #plt.show()
    #print( np.argwhere(w_indexes == 990) )
    f_interp = np.zeros( (Nq*Np, ) )

    nz = ( 1 + fw - (w_indexes % Nw) ) % Nw - 1

    s_quads = np.abs( s_quads )

    s_quads[((fw + Nw    ) % (2*Nw) == w_indexes) |
            ((fw + Nw - 1) % (2*Nw) == w_indexes) |
            ((fw + Nw - 2) % (2*Nw) == w_indexes) |
            ((fw + Nw + 1) % (2*Nw) == w_indexes)] *= -1

    #theta = w[w_indexes % Nw] + dw/2

    #f_interp = poly_weights[fs] / ( s_quads - s[fs] + ff ) / \
    #           np.sum(poly_weights / ( s_quads.reshape((-1,1)) - s.reshape((1,-1)) + ff ), axis=1)

    s_inds = np.digitize( s_quads, s )
    f_interp = ( (s_inds - 1 == fs) * (s[s_inds] - s_quads) + (s_inds == fs) * (s_quads - s[s_inds - 1]) ) / \
               ( s[s_inds] - s[s_inds - 1] )

    #print( w_quads[17] )
    #print( w[w_indexes[17] % Nw] + dw/2 )
    #print( w_quads[17] - ( w[w_indexes[17] % Nw] + dw/2 ) )
    #print( np.mod(w_quads[17], np.pi) - (w[w_indexes[17] % Nw] + dw/2) )

    #w_quads = np.mod( w_quads, np.pi ) - (w[w_indexes % Nw] + dw/2)
    #w_quads = np.minimum( w_quads - (w[w_indexes%Nw]+dw/2),
    #                      w_quads - (w[w_indexes%Nw]+dw/2) + np.pi )
    dist = np.mod(w_quads, np.pi) - (w[w_indexes % Nw] + dw/2)
    dist[ dist > dw ] -= np.pi
    dist[ dist < -dw ] += np.pi

    f_interp[nz == -1] *= (-1/16) + (1/24/dw)*( dist[nz==-1] ) + \
                                ( 1/4/dw**2 )*( dist[nz==-1] )**2 + \
                                (-1/6/dw**3 )*( dist[nz==-1] )**3

    f_interp[nz ==  0] *= ( 9/16) - (9/8/dw)*( dist[nz==0] ) + \
                               (-1/4/dw**2 )*( dist[nz==0] )**2 + \
                               ( 1/2/dw**3 )*( dist[nz==0] )**3

    f_interp[nz ==  1] *= ( 9/16) + (9/8/dw)*( dist[nz==1] ) + \
                               (-1/4/dw**2 )*( dist[nz==1] )**2 + \
                               (-1/2/dw**3 )*( dist[nz==1] )**3

    f_interp[nz ==  2] *= (-1/16) - (1/24/dw)*( dist[nz==2] ) + \
                                ( 1/4/dw**2 )*( dist[nz==2] )**2 + \
                                 (1/6/dw**3 )*( dist[nz==2] )**3

    f_interp[(nz < -1) | (nz > 2)] = 0
    """
    #np.savetxt('f_interp.txt', f_interp )
    for p in range( 2603, 2606 ):
        plt.plot( S*np.cos(W), S*np.sin(W), 'k.')
        plt.plot( s[fs]*np.cos(w[fw]), s[fs]*np.sin(w[fw]), 'b*' )
        plt.plot( x_quads[p], y_quads[p], 'r.' )
        plt.title( "{} {:.4f} {:.4f} {:.4f}".format( p, theta[p], np.mod(w_quads[p], np.pi), f_interp[p] ) )
        plt.show()
    #"""
    #print( "p, w_indexes[p], s_quads[p], w_quads[p], theta[p], f_interp[p]" )
    #print( "w[w_indexes[p] % Nw], dw, dw/2" )
    """
    for p in range( Np*Nq ):
        if f_interp[p] > 2:
            plt.plot( S*np.cos(W), S*np.sin(W), 'k.')
            plt.plot( s[fs]*np.cos(w[fw]), s[fs]*np.sin(w[fw]), 'b*' )
            plt.plot( x_quads[p], y_quads[p], 'ro' )
            print( p )
            print('w_indexes:', w_indexes[p])
            print('w[w_indexes % Nw]:', w[w_indexes[p] % Nw] )
            print('w_quads:', w_quads[p])
            print('f_interp:', f_interp[p] )
            print()
            plt.show()
    #plt.show()
    #print( "f_interp, w_quads, s_quads, theta" )
    #print( f_interp[67], w_quads[67], s_quads[67], theta[67] )
    #"""
    f_interp = f_interp.reshape((Np, Nq))

    f_hat = rs.reshape((1, -1))*np.sum( weights.reshape((1, -1))*f_interp, axis=1 )

    return f_hat.flatten()




if __name__ == "__main__":
    Ns = 4
    Nw = 5#[25, 50, 100, 200, 400]
    Nq = 4
    alpha = 40
    b = 0.05
    a = 0.2
    # f = lambda x, y: np.exp(-alpha*(x*x+y*y))
    f = lambda x, y: np.exp(-(x/a)**2 - (y/b)**2)
    radon(2, 3, Ns, Nw, Nq)

    ggg
    # Create radii discretization
    s = rtv1.clen_curt(Ns)[0]

    # Perform convergence study as we refine
    # the angular discretization
    errs = []

    # for Nw in N_info
    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]

    # Create s-omega meshgrid and plot it
    [S, W] = np.meshgrid(s, w)

    # plt.figure(1)
    # plt.plot(S*np.cos(W), S*np.sin(W), "k.")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("s-omega grid")
    # plt.show()

    # Evaluate f on the s-omega meshgrid
    f_vec = f(S*np.cos(W), S*np.sin(W))

    # Compute exact solution on the s-omega meshgrid
    f_hat_exact = np.zeros_like(f_vec)
    # f_hat_exact[:, :] = np.sqrt(np.pi)/np.sqrt(alpha) * np.exp(-alpha*(s*s).T)
    denom = (a*np.cos(W))**2 + (b*np.sin(W))**2
    f_hat_exact[:, :] = (a*b*np.sqrt(np.pi))/(np.sqrt(denom))*np.exp(-(s*s)/(denom))

    f_hat_exact = f_hat_exact.flatten()

    # Reshape f_vec into 1D array
    # and plot it to see if we can plot
    # f as a vector
    f_vec = f_vec.flatten()

    # Compute the approximate Radon transform of f
    # using information only at the discretization points
    f_hat = radon(f_vec, Ns, Nw, Nq)

    err = np.max(np.absolute(f_hat_exact - f_hat))
    print("Nw = {}, max error = {:.16f}".format(Nw, err))
    errs.append(err)
    """
    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nws+ 1)[:-1]

    # Create s-omega meshgrid and plot it
    [S, W] = np.meshgrid(s, w)

    f_vec = f(S*np.cos(W), S*np.sin(W))

    f_hat = radon(f_vec, Ns, Nws, Nq)
    """
    # Reshape f_vec into 1D array
    # and plot it to see if we can plot
    # f as a vector
    f_vec = f_vec.flatten()

    # Compute the approximate Radon transform of f
    # using information only at the discretization points

    # Plot original function
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_wireframe(S*np.cos(W), S*np.sin(W), f(S*np.cos(W), S*np.sin(W)))
    plt.title("Original function")


    # Plot the exact Radon transform
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_wireframe(S, W, f_hat_exact.reshape((Nw, Ns)))
    plt.title("Exact Radon Transform")

    # Plot the approximate Radon transform
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_wireframe(S, W, f_hat.reshape((Nw, Ns)))
    plt.title("Approximate Radon Transform")

    # Plot the contour of the difference between the
    # exact Radon transform of f and the approximate
    # Radon transform
    plt.figure(4)
    plt.contourf(S, W, np.absolute(f_hat_exact-f_hat).reshape((Nw, Ns)))
    plt.colorbar()

    plt.show()


    # Check how NumPy is flattening f_vec
    # print(f(S*np.cos(W), S*np.sin(W))[0, 0:5])
    # print(f_vec[0:5])
    # p = 842
    # i = p // Nw
    # j = p % Ns
    # print(f_vec[p])
    # print(f(S*np.cos(W), S*np.sin(W))[i, j])
    # print(np.allclose(f_vec[p], f(S*np.cos(W), S*np.sin(W))[i, j]))
