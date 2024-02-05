import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import radon_transform_v1 as rtv1

def radon(f_vec, Ns, Nw, Nq):
    # Get radii discretization
    s = rtv1.clen_curt(Ns)[0]

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = rtv1.clen_curt(Nq)

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros_like(f_vec)

    #R = np.zeros((Ns*Nw, Ns*Nw))

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
            for k in range(Nq):
                x_q = s_j*np.cos(w_i) - r*nodes[k]*np.sin(w_i)
                y_q = s_j*np.sin(w_i) + r*nodes[k]*np.cos(w_i)

                # Convert the quadrature point from x-y space
                # into s-omega space
                w_q = np.arctan2(y_q, x_q)
                s_q = np.sqrt(x_q*x_q + y_q*y_q)

                # If angle of the quadrature point is pi
                # make the angle 0 and use the negative radius
                if np.allclose(np.pi, w_q, rtol=1e-8):
                    w_q = w_q - np.pi
                    s_q = -s_q
                # If angle is negative, make the angle positive
                # and use the negative radius
                elif w_q < 0:
                    w_q = w_q + np.pi
                    s_q = -s_q

                # Find which two angles the quadrature point
                # lies between
                w_ind = int(np.floor(w_q/(w[1]-w[0])))
                w2 = w[w_ind % Nw]
                w3 = w[(w_ind + 1) % Nw]

                # Find another two angles that the quadrature
                # point lies between
                w1 = w[(w_ind - 1) % Nw]
                w4 = w[(w_ind + 2) % Nw]

                # Find the radii corresponding to the above four angles
                # We also test if we "wrapped around" the disk
                s2  = s_q if (w_ind % Nw == w_ind) else -s_q
                s3 = s_q if ((w_ind + 1) % Nw == (w_ind + 1)) else -s_q
                s1 = s_q if ((w_ind - 1) % Nw == (w_ind - 1)) else -s_q
                s4 = s_q if ((w_ind + 2) % Nw == (w_ind + 2)) else -s_q

                # Find the function values along the lower line
                lower_start_ind = (w_ind % Nw) * Ns
                f2 = f_vec[lower_start_ind:(lower_start_ind + Ns)]

                # Find the function values along the upper line
                upper_start_ind = ((w_ind + 1) % Nw) * Ns
                f3 = f_vec[upper_start_ind:(upper_start_ind + Ns)]

                # Find the function values along the lower2 linear
                lower2_start_ind = ((w_ind - 1) % Nw) * Ns
                f1 = f_vec[lower2_start_ind:(lower2_start_ind + Ns)]

                # Find the function values along the upper 2 line
                upper2_start_ind = ((w_ind + 2) % Nw) * Ns
                f4 = f_vec[upper2_start_ind:(upper2_start_ind + Ns)]



                # We compute the function value at the quadrature point
                # by taking a convex combination of the function values
                # at the lower and upper points
                # We compute the weights associated with the lower and upper
                # function values, respectively, by "measuring" the distance
                # between the quadrature point and the lower and upper points,
                # respectively
                # upper_weight = (w_q - w_lower)/(w[1]-w[0])
                # lower_weight = (w_upper - w_q)/(w[1]-w[0])

                fudge_factor = 1e-15

                theta1 = w2 + (dw / 2)
                p1 = np.dot(poly_weights, f1/(s1 - s + fudge_factor)) / np.dot(poly_weights, 1/(s1 - s + fudge_factor))
                p2 = np.dot(poly_weights, f2/(s2 - s + fudge_factor)) / np.dot(poly_weights, 1/(s2 - s + fudge_factor))
                p3 = np.dot(poly_weights, f3/(s3 - s + fudge_factor)) / np.dot(poly_weights, 1/(s3- s + fudge_factor))
                p4 = np.dot(poly_weights, f4/(s4 - s + fudge_factor)) / np.dot(poly_weights, 1/(s4 - s + fudge_factor))

                ptheta = ( -1/16 )*( p1 - 9*p2 - 9*p3 + p4 ) + ( 1/(24*dw) )*( p1 - 27*p2 + 27*p3 - p4 )*( w_q - theta1 ) \
                       + ( 1/(4*dw**2) )*( p1 - p2 - p3 + p4 )*( w_q - theta1 )**2 - ( 1/(6*dw**3) )*( p1 - 3*p2 + 3*p3 - p4 )*( w_q - theta1 )**3

                # if 0 <= upper_weight <= 1:
                #     lower_weight = 1 - upper_weight
                # else:
                #     upper_weight = 1 - lower_weight

                # If the upper or lower points are identical to one of the radii discretization
                # points, we use a fudge factor to prevent division by zero in the
                # barycentric interpolation

                # Approximate f on the lower line at the lower point
                # p_lower = np.dot(poly_weights, f_lower/(s_lower - s + fudge_factor)) / np.dot(poly_weights, 1/(s_lower - s + fudge_factor))

                # Approximate f on the upper line at the lower point
                # p_upper = np.dot(poly_weights, f_upper/(s_upper - s + fudge_factor)) / np.dot(poly_weights, 1/(s_upper - s + fudge_factor))

                # Update the line integral approximation
                f_hat[p] += r*weights[k]*ptheta

                # Plot the x-y mesh (mostly for testing purposes)
                # plt.figure(1)
                # plt.plot(S*np.cos(W), S*np.sin(W), "k.")
                # plt.plot(x_q, y_q, "ro")
                # plt.plot(s2*np.cos(w2), s2*np.sin(w2), "g.")
                # plt.plot(s3*np.cos(w3), s3*np.sin(w3), "g.")
                # plt.plot(s1*np.cos(w1), s1*np.sin(w1), "g.")
                # plt.plot(s4*np.cos(w4), s4*np.sin(w4), "g.")

                # Plot the s-omega mesh
                # plt.figure(2)
                # plt.plot(S, W, "k.")
                # plt.plot(s_q, w_q, "bo")
                # plt.plot(s_q, w2, "g.")
                # plt.plot(s_q, w3, "g.")
                # plt.plot(s_q, w1, "g.")
                # plt.plot(s_q, w4, "g.")

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
