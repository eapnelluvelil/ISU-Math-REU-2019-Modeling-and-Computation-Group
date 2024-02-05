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
    dw = np.absolute(w[1]-w[0])
    Np = Ns*Nw

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = rtv1.clen_curt(Nq)

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

    s = s.reshape((-1, 1))
    rs = np.sqrt( 1 - s*s ).reshape((-1, 1))
    for i in range( Nw ):
        x_quads = s*np.cos(w[i]) - rs*nodes.reshape((1, -1))*np.sin(w[i])
        y_quads = s*np.sin(w[i]) + rs*nodes.reshape((1, -1))*np.cos(w[i])

        x_quads = x_quads.reshape((-1, ))
        y_quads = y_quads.reshape((-1, ))

        w_quads = np.arctan2( y_quads, x_quads )
        s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

        s_quads[ ~((0 <= w_quads) & (w_quads <= np.pi)) ] *= -1
        w_quads = np.mod( w_quads, np.pi )

        w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

        w_values = np.empty( (Nq*Ns, 4) )
        s_values = np.empty( (Nq*Ns, 4) )
        f_values = np.empty( (Nq*Ns, 4, Ns) )
        p_values = np.empty( (Nq*Ns, 4) )

        w_values[:,0] = w[(w_indexes-1) % Nw]
        w_values[:,1] = w[w_indexes % Nw]
        w_values[:,2] = w[(w_indexes+1) % Nw]
        w_values[:,3] = w[(w_indexes+2) % Nw]

        s_values[:,0] = s_values[:,1] = s_values[:,2] = s_values[:,3] = s_quads
        s_values[(w_indexes-1) % Nw == w_indexes-1, 0] *= -1
        s_values[ w_indexes    % Nw == w_indexes  , 1] *= -1
        s_values[(w_indexes+1) % Nw == w_indexes+1, 2] *= -1
        s_values[(w_indexes+2) % Nw == w_indexes+2, 3] *= -1

        f_values[:, 0, :] = f_vec[(w_indexes-1) % Nw, :]
        f_values[:, 1, :] = f_vec[(w_indexes  ) % Nw, :]
        f_values[:, 2, :] = f_vec[(w_indexes+1) % Nw, :]
        f_values[:, 3, :] = f_vec[(w_indexes+2) % Nw, :]

        ff = 1e-15

        theta = w_values[:, 1] + dw/2

        p_values[:, 0] = np.sum(poly_weights * f_values[:,0,:]/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                         np.sum(poly_weights * 1/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
        p_values[:, 1] = np.sum(poly_weights * f_values[:,1,:]/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                         np.sum(poly_weights * 1/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
        p_values[:, 2] = np.sum(poly_weights * f_values[:,2,:]/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                         np.sum(poly_weights * 1/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
        p_values[:, 3] = np.sum(poly_weights * f_values[:,3,:]/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                         np.sum(poly_weights * 1/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)

        ptheta = (-1/16      )*(p_values[:,0] -  9*p_values[:,1] -  9*p_values[:,2] + p_values[:,3] ) + \
                 ( 1/24/dw   )*(p_values[:,0] - 27*p_values[:,1] + 27*p_values[:,2] - p_values[:,3] )*( w_quads - theta ) + \
                 ( 1/4/dw**2 )*(p_values[:,0] -    p_values[:,1] -    p_values[:,2] + p_values[:,3] )*( w_quads - theta )**2 - \
                 ( 1/6/dw**3 )*(p_values[:,0] -  3*p_values[:,1] +  3*p_values[:,2] - p_values[:,3] )*( w_quads - theta )**3

        ptheta = ptheta.reshape((Ns, Nq))

        f_hat[i*Ns:(i+1)*Ns] = rs.reshape((1, -1))*np.sum( weights.reshape((1, -1))*ptheta, axis=1 )

    return f_hat


def radon_inverse(f_hat, Ns, Nw, Nq, tol=1e-5):
    s = rtv1.clen_curt(Ns)[0]
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

        print( np.linalg.norm( rk_1, np.inf ) )

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot( 111, projection='3d' )
        ax3.plot_wireframe(S*np.cos(W), S*np.sin(W), x.reshape((Nw, Ns)),
                            color='blue')
        ax3.set_title("Un-Radon De-Transformed")
        plt.pause(1)

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
