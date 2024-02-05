import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import radon_transform_v1 as rtv1

def radon(f_vec, Ns, Nw, Nq):
    # Get radii discretization
    s = rtv1.clen_curt(Ns)[0]

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = w[1] - w[0]

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = rtv1.clen_curt(Nq)

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros_like(f_vec)

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    # Go through angles
    for i in range(Nw):
        # Go through radii
        for j in range(Ns)[1:-1]:
            # Get linear index
            p = i*Ns + j

            # Get current radius and angle
            w_i = w[i]
            s_j = s[j]

            # Compute half the length of the chord perpendicular
            # to the line determined by w_i and s_j
            r = np.sqrt(1 - s_j*s_j)

            # Discretize the chord perpendicular to the line
            # determined by w_i and s_j
            x_quads = s_j*np.cos(w_i) - r*nodes*np.sin(w_i)
            y_quads = s_j*np.sin(w_i) + r*nodes*np.cos(w_i)

            # Get points in polar
            w_quads = np.arctan2( y_quads, x_quads )
            s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

            # Make sure angles are valid
            s_quads[ ~((0 <= w_quads) & (w_quads <= np.pi))] *= -1
            w_quads = np.mod( w_quads, np.pi )

            # Get indexes for p1
            w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

            # set up array for p1, p2
            w_values = np.empty( (Ns, 2) )
            s_values = np.empty( (Ns, 2) )
            f_values = np.empty( (Ns, 2, Nq) )
            p_values = np.empty( (Ns, 2) )
            i_weights = np.empty( (Ns, 2) )

            w_values[:, 0] = w[w_indexes % Nw]
            w_values[:, 1] = w[(w_indexes + 1) % Nw]

            s_values[:, 0] = s_quads
            #print( "s_values:", s_values.shape )
            #print( "w_indexes:", w_indexes.shape )
            #print( "s_quads:", s_quads.shape )
            s_values[w_indexes % Nw == w_indexes, 0] *= -1

            s_values[:, 1] = s_quads
            s_values[(w_indexes+1) % Nw == (w_indexes+1), 1] *= -1

            f_indexes_0 = (w_indexes%Nw)*Ns
            f_indexes_1 = ((w_indexes+1)%Nw)*Ns
            for k in range( Nq ):
                f_values[k, 0, :] = f_vec[f_indexes_0[k]:f_indexes_0[k] + Ns]
                f_values[k, 1, :] = f_vec[f_indexes_1[k]:f_indexes_1[k] + Ns]

            #f_values[:, 0, :] = f_vec[(w_indexes%Nw)*Ns:(w_indexes%Nw)*Ns + Ns]
            #f_values[:, 1, :] = f_vec[((w_indexes+1)%Nw)*Ns:((w_indexes+1)%Nw)*Ns + Ns]

            i_weights[:,0] = w_quads - w_values[:,0]
            i_weights[:,1] = w_values[:,1] - w_quads
            i_weights /= dw

            conds = (0 <= i_weights[:,1]) & (i_weights[:,1] <= 1)
            i_weights[conds, 0] = 1 - i_weights[conds,1]
            i_weights[~conds, 1] = 1 - i_weights[~conds,0]

            ff = 1e-15

            # Approximate f on the lower line at the lower point
            for k in range( Nq ):
                p_values[k,0] = np.dot(poly_weights, f_values[k,0,:]/(s_values[k,0] - s + ff)) / \
                                np.dot(poly_weights, 1/(s_values[k,0] - s + ff))

                p_values[k,1] = np.dot(poly_weights, f_values[k,1,:]/(s_values[k,1] - s + ff)) / \
                                np.dot(poly_weights, 1/(s_values[k,1] - s + ff))

            # Update the line integral approximation
            f_hat[p] = r*np.dot(weights, p_values[:,0]*i_weights[:,0] + \
                                         p_values[:,1]*i_weights[:,1])

    return f_hat

if __name__ == "__main__":
    Ns = 300
    Nws = [25, 50, 100, 200, 400]
    Nq = 100
    alpha = 40
    b = 0.05
    a = 0.2
    # f = lambda x, y: np.exp(-alpha*(x*x+y*y))
    f = lambda x, y: np.exp(-(x/a)**2 - (y/b)**2)

    # Create radii discretization
    s = rtv1.clen_curt(Ns)[0]
    
    # Perform convergence study as we refine
    # the angular discretization
    errs = []
    for Nw in Nws:
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

    # Reshape f_vec into 1D array
    # and plot it to see if we can plot
    # f as a vector
    f_vec = f_vec.flatten()

    # Compute the approximate Radon transform of f
    # using information only at the discretization points
    """
    # Plot original function
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_wireframe(S*np.cos(W), S*np.sin(W), f(S*np.cos(W), S*np.sin(W)))

    
    # Plot the exact Radon transform
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_wireframe(S, W, f_hat_exact.reshape((Nw, Ns)))

    # Plot the approximate Radon transform
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_wireframe(S, W, f_hat.reshape((Nw, Ns)))

    # Plot the contour of the difference between the
    # exact Radon transform of f and the approximate
    # Radon transform
    plt.figure(4)
    plt.contourf(S, W, np.absolute(f_hat_exact-f_hat).reshape((Nw, Ns)))
    plt.colorbar()
    
    plt.show()
    """

    # Check how NumPy is flattening f_vec
    # print(f(S*np.cos(W), S*np.sin(W))[0, 0:5])
    # print(f_vec[0:5])
    # p = 842
    # i = p // Nw
    # j = p % Ns
    # print(f_vec[p])
    # print(f(S*np.cos(W), S*np.sin(W))[i, j])
    # print(np.allclose(f_vec[p], f(S*np.cos(W), S*np.sin(W))[i, j]))
