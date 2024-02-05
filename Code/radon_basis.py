import numpy as np
import radon_transform as rt
import matplotlib
import matplotlib.pyplot as plt

def radon(fw, fs, Ns, Nw, Nq):
    # Get radii discretization
    s = rt.clen_curt(Ns)[0]

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = rt.clen_curt(Nq)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros((Ns*Nw, ))

    #R = np.zeros((Ns*Nw, Ns*Nw))

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    ps = np.array([[1, np.sin(-3*dw), np.sin(-3*dw)**2, np.sin(-3*dw)**3],
                   [1, np.sin(-dw),   np.sin(-dw)**2,   np.sin(-dw)**3  ],
                   [1, np.sin(dw),    np.sin(dw)**2,    np.sin(dw)**3   ],
                   [1, np.sin(3*dw),  np.sin(3*dw)**2,  np.sin(dw)**3   ]])
    ps = np.linalg.inv(ps)
    [S, W] = np.meshgrid(s, w)

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
            for k in range(Nq):
                x_q = s_j*np.cos(w_i) - r*nodes[k]*np.sin(w_i)
                y_q = s_j*np.sin(w_i) + r*nodes[k]*np.cos(w_i)
                
                # Convert the quadrature point from x-y space
                # into s-omega space
                w_q = np.arctan2(y_q, x_q)
                s_q = np.sqrt(x_q*x_q + y_q*y_q)

                if ~(0 <= w_q < np.pi):
                    s_q = -s_q
                    
                w_q = np.mod( w_q, np.pi )

                # plt.plot(x_q, y_q, "bo")
                # plt.plot(s_q*np.cos(w_q), s_q*np.sin(w_q), "r.")

                w_ind = int(np.floor(w_q/(w[1]-w[0])))

                # if fw < (w_ind - 1) or fw > (w_ind + 2):
                #     continue

                w1 = w[(w_ind - 1) % Nw]
                w2 = w[w_ind % Nw]
                w3 = w[(w_ind + 1) % Nw]
                w4 = w[(w_ind + 2) % Nw]

                if w[fw] == w1:
                    col = 0
                    pos = w_ind - 1
                    w_f = w1
                elif w[fw] == w2:
                    col = 1
                    pos = w_ind
                    w_f = w2
                elif w[fw] == w3:
                    col = 2
                    pos = w_ind + 1
                    w_f = w3
                elif w[fw] == w4:
                    col = 3
                    pos = w_ind + 2
                    w_f = w4
                else:
                    continue

                to_negate = 1 * (1 if pos % Nw == pos else -1)

                s_f = to_negate * s_q
                
                fudge_factor = 1e-15

                theta1 = w2 + (dw / 2)
                pval_test = poly_weights[fs]/(s_f - s[fs] + fudge_factor) / np.dot(poly_weights, 1/(s_f - s + fudge_factor))

               
                f_vals = np.zeros((Ns, ))
                f_vals[fs] = 1
                pval = np.zeros((Nq, ))
                pval[col] = np.dot(poly_weights, f_vals/(s_f - s + fudge_factor)) / np.dot(poly_weights, 1/(s_f - s + fudge_factor))
              
                ptheta = pval_test*np.sum( ps[:, col] * \
                         np.array([1, np.sin(w_q - theta1), np.sin(w_q - theta1)**2, np.sin(w_q - theta1)**3]))

                #ptheta = ( -1/16 )*( pval[0] - 9*pval[1] - 9*pval[2] + pval[3] ) \
                #       + ( 1/(24*dw) )*( pval[0] - 27*pval[1] + 27*pval[2] - pval[3] )*( w_q - theta1 ) \
                #       + ( 1/(4*dw**2) )*( pval[0] - pval[1] - pval[2] + pval[3] )*( w_q - theta1 )**2 \
                #       - ( 1/(6*dw**3) )*( pval[0] - 3*pval[1] + 3*pval[2] - pval[3] )*( w_q - theta1 )**3

                # print("np.allclose(pval_test-pval[col]): {:d}".format(np.allclose(pval_test, pval[col])))
                # print("np.allclose(ptheta_test, ptheta): {:d}".format(np.allclose(ptheta_test, ptheta)))
                # print(ptheta_test)
                # print(ptheta)g
                # print("")
                
                # plt.figure(1)
                # plt.plot(S*np.cos(W), S*np.sin(W), "k.")
                # plt.plot(s_f*np.cos(w_f), s_f*np.sin(w_f), "ro")

                # Update the line integral approximation
                f_hat[p] += r*weights[k]*ptheta
                
    # plt.show()
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
        """
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot( 111, projection='3d' )
        ax3.plot_wireframe(S*np.cos(W), S*np.sin(W), x.reshape((Nw, Ns)),
                            color='blue')
        ax3.set_title("Un-Radon De-Transformed")
        plt.pause(1)
        """
    return x


if __name__ == "__main__":
    Ns = 25
    Nw = 25#[25, 50, 100, 200, 400]
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
    """
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
