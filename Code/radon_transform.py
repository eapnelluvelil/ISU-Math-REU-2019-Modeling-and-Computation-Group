import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import LinearOperator

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


def radon_operator(Ns, Nw, Nq):
    def rad_wrapper( f_vec ):
        return radon( f_vec, Ns, Nw, Nq )
    return LinearOperator((Ns*Nw, Ns*Nw), matvec=rad_wrapper)

def radon(f_vec, Ns, Nw, Nq):
    f_vec = f_vec.flatten()
    # Get radii discretization
    s = clen_curt(Ns)[0]

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = clen_curt(Nq)

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros_like(f_vec)
    f_vec = f_vec.reshape((Nw, Ns))

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    # Go through angles
    ff = 1e-15
    for i in range(Nw):
        # Go through radii
        for j in range(Ns):
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

            w_quads = np.arctan2( y_quads, x_quads )
            s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

            s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
            w_quads = np.mod( w_quads, np.pi )

            w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

            s_values = np.empty( (Nq, 4) )
            f_values = np.empty( (Nq, 4, Ns) )
            p_values = np.empty( (Nq, 4) )


            s_values[:,0] = s_values[:,1] = s_values[:,2] = s_values[:,3] = s_quads[:]
            s_values[(w_indexes-1) % Nw != w_indexes-1, 0] *= -1
            s_values[ w_indexes    % Nw != w_indexes  , 1] *= -1
            s_values[(w_indexes+1) % Nw != w_indexes+1, 2] *= -1
            s_values[(w_indexes+2) % Nw != w_indexes+2, 3] *= -1

            f_values[:, 0, :] = f_vec[(w_indexes-1) % Nw, :]
            f_values[:, 1, :] = f_vec[(w_indexes  ) % Nw, :]
            f_values[:, 2, :] = f_vec[(w_indexes+1) % Nw, :]
            f_values[:, 3, :] = f_vec[(w_indexes+2) % Nw, :]

            #print( x_quads, y_quads )
            #print( f_values[:, 0, :] )

                #plt.cla()
            theta = w[w_indexes % Nw] + dw/2

            p_values[:, 0] = np.sum(poly_weights * f_values[:,0,:]/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 1] = np.sum(poly_weights * f_values[:,1,:]/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 2] = np.sum(poly_weights * f_values[:,2,:]/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 3] = np.sum(poly_weights * f_values[:,3,:]/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)

            angle_dist = np.mod(w_quads, np.pi) - theta
            angle_dist[ angle_dist > dw ] -= np.pi
            angle_dist[ angle_dist < -dw ] += np.pi

            ptheta = (-1/16      )*(p_values[:,0] -  9*p_values[:,1] -  9*p_values[:,2] + p_values[:,3] ) + \
                     ( 1/24/dw   )*(p_values[:,0] - 27*p_values[:,1] + 27*p_values[:,2] - p_values[:,3] )*( angle_dist ) + \
                     ( 1/4/dw**2 )*(p_values[:,0] -    p_values[:,1] -    p_values[:,2] + p_values[:,3] )*( angle_dist )**2 - \
                     ( 1/6/dw**3 )*(p_values[:,0] -  3*p_values[:,1] +  3*p_values[:,2] - p_values[:,3] )*( angle_dist )**3
            """
            for k in range(Nq):
                if w_quads[k] - theta[k] != angle_dist[k]:
                    #if np.allclose( ptheta[k], 1 ):
                    #if np.isclose(s_quads[k], 0):
                    plt.plot( S*np.cos(W), S*np.sin(W), 'k.' )
                    plt.plot( s[0]*np.cos(w[0]), s[0]*np.sin(w[0]), 'b*' )
                    plt.plot( x_quads[k], y_quads[k], 'g*' )
                    plt.plot( s_values[k,0]*np.cos(w[(w_indexes[k]-1)%Nw]), s_values[k,0]*np.sin(w[(w_indexes[k]-1)%Nw]), 'y.', label=str(p_values[k,0]) )
                    plt.plot( s_values[k,1]*np.cos(w[(w_indexes[k]  )%Nw]), s_values[k,1]*np.sin(w[(w_indexes[k]  )%Nw]), 'y.', label=str(p_values[k,1]) )
                    plt.plot( s_values[k,2]*np.cos(w[(w_indexes[k]+1)%Nw]), s_values[k,2]*np.sin(w[(w_indexes[k]+1)%Nw]), 'y.', label=str(p_values[k,2]) )
                    plt.plot( s_values[k,3]*np.cos(w[(w_indexes[k]+2)%Nw]), s_values[k,3]*np.sin(w[(w_indexes[k]+2)%Nw]), 'y.', label=str(p_values[k,3]) )
                    plt.title( "{} {:.4f} {:.4f}".format(ptheta[k], w_quads[k] - theta[k], angle_dist[k]) )
                    plt.legend()
                    plt.show()
            #"""

            f_hat[p] = r*np.dot( weights, ptheta )

    return f_hat



def radon_inverse(f_hat, Ns, Nw, Nq, tol=1e-5):
    low_Nq = int( np.sqrt( Nq ) )
    s = clen_curt(Ns)[0]
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)

    x = np.zeros_like(f_hat)
    Ax = radon( x, Ns, Nw, Nq )
    rk_1 = low_res_inverse(f_hat, Ns, Nw, low_Nq) - low_res_inverse( Ax, Ns, Nw, low_Nq )
    p = np.copy( rk_1 )
    r_hat = np.copy( rk_1 )

    print( r_hat )
    while np.linalg.norm( rk_1, 2 ) > tol:
        Ap = radon( p, Ns, Nw, Nq )
        PAp = low_res_inverse( Ap, Ns, Nw, low_Nq )

        a = np.dot( rk_1, r_hat )/np.dot( PAp, r_hat )

        x = x + a*p

        rk = rk_1 - a*PAp

        Ark = radon( rk, Ns, Nw, Nq )
        PArk = low_res_inverse( Ark, Ns, Nw, low_Nq )

        w = np.dot( rk, PArk )/np.dot( PArk, PArk )

        x = x + w*rk

        rk = rk - w*PArk

        b = a/w * np.dot( rk, r_hat ) / np.dot( rk_1, r_hat )

        p = rk + b*(p - w*PAp)

        rk_1 = rk

        print( np.linalg.norm( rk_1, np.inf ) )

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot( 111, projection='3d' )
        ax3.plot_wireframe(S*np.cos(W), S*np.sin(W), x.reshape((Nw, Ns)),
                            color='blue')
        ax3.set_title("Un-Radon De-Transformed")
        plt.pause(1)

    return x

def low_res_inverse(f_hat, Ns, Nw, Nq, tol=1e-5):
    s = clen_curt(Ns)[0]
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)

    x = np.zeros_like(f_hat)
    rk_1 = f_hat - radon( x, Ns, Nw, Nq )
    p = np.copy( rk_1 )
    r_hat = np.copy( rk_1 )

    while np.linalg.norm( rk_1, 2 ) > tol:
        Ap = radon( p, Ns, Nw, Nq )

        a = np.dot( rk_1, r_hat )/np.dot( Ap, r_hat )

        x = x + a*p

        rk = rk_1 - a*Ap

        Ark = radon( rk, Ns, Nw, Nq )

        w = np.dot( rk, Ark )/np.dot( Ark, Ark )

        x = x + w*rk

        rk = rk - w*Ark

        b = a/w * np.dot( rk, r_hat ) / np.dot( rk_1, r_hat )

        p = rk + b*(p - w*Ap)

        rk_1 = rk

    return x

if __name__ == "__main__":
    Ns = 25
    Nws = 25#[25, 50, 100, 200, 400]
    Nq = 50
    alpha = 40
    b = 0.05
    a = 0.2
    # f = lambda x, y: np.exp(-alpha*(x*x+y*y))
    f = lambda x, y: np.exp(-(x/a)**2 - (y/b)**2)

    # Create radii discretization
    s = clen_curt(Ns)[0]

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

