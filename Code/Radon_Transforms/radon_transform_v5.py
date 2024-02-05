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
                w_lower = w[w_ind % Nw]
                w_upper = w[(w_ind + 1) % Nw]

                # Find the radii corresponding to the above two angles
                # We also test if we "wrapped around" the disk
                s_lower  = s_q if (w_ind % Nw == w_ind) else -s_q
                s_upper = s_q if ((w_ind + 1) % Nw == (w_ind + 1)) else -s_q

                # Find the function values along the lower line
                lower_start_ind = (w_ind % Nw) * Ns
                f_lower = f_vec[lower_start_ind:(lower_start_ind + Ns)]

                # Find the function values along the upper line
                upper_start_ind = ((w_ind + 1) % Nw) * Ns
                f_upper = f_vec[upper_start_ind:(upper_start_ind + Ns)]

                # We compute the function value at the quadrature point
                # by taking a convex combination of the function values
                # at the lower and upper points
                # We compute the weights associated with the lower and upper
                # function values, respectively, by "measuring" the distance
                # between the quadrature point and the lower and upper points,
                # respectively
                upper_weight = (w_q - w_lower)/(w[1]-w[0])
                lower_weight = (w_upper - w_q)/(w[1]-w[0])

                if 0 <= upper_weight <= 1:
                    lower_weight = 1 - upper_weight
                else:
                    upper_weight = 1 - lower_weight

                # If the upper or lower points are identical to one of the radii discretization
                # points, we use a fudge factor to prevent division by zero in the
                # barycentric interpolation
                fudge_factor = 1e-15

                # Approximate f on the lower line at the lower point
                p_lower = np.dot(poly_weights, f_lower/(s_lower - s + fudge_factor)) / np.dot(poly_weights, 1/(s_lower - s + fudge_factor))

                # Approximate f on the upper line at the lower point
                p_upper = np.dot(poly_weights, f_upper/(s_upper - s + fudge_factor)) / np.dot(poly_weights, 1/(s_upper - s + fudge_factor))

                # Update the line integral approximation
                f_hat[p] += r*weights[k]*(p_lower*lower_weight + p_upper*upper_weight)

                # Plot the x-y mesh (mostly for testing purposes)
                plt.figure(1)
                plt.plot(S*np.cos(W), S*np.sin(W), "k.")
                plt.plot(x_q, y_q, "ro")
                plt.plot(s_lower*np.cos(w_lower), s_lower*np.sin(w_lower), "g.")
                plt.plot(s_upper*np.cos(w_upper), s_upper*np.sin(w_upper), "g.")

                # Plot the s-omega mesh
                # plt.figure(2)
                # plt.plot(S, W, "k.")
                # plt.plot(s_q, w_q, "bo")
                # plt.plot(s_q, w_lower, "g.")
                # plt.plot(s_q, w_upper, "g.")

                plt.show()

    return f_hat

if __name__ == "__main__":
    Ns = 200
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
