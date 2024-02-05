"""
numerical_utilities.py

Contains various numerical methods used in the package
    chebyshev   := Returns chebyshev points
    clen_curt   := Returns nodes and weights used for quadrature
    diff_matrix := Returns differentiation matrix
    pn_coeffs   := Returns matrix coefficients for Pn
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def chebyshev( N, a=-1, b=1 ):
    cheb = np.cos( np.pi * np.arange( 0, N, 1 ) / (N - 1) )
    return a + (b-a)*(cheb+1)/2

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

def diff_matrix( N, a=-1, b=1 ):
    [J, I] = np.meshgrid( np.arange(0, N, 1), np.arange(0, N, 1) )
    diags = np.eye(N, dtype=bool)

    cheb = chebyshev( N, a, b )

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

    return spec_mat

"""
Inputs:
Ns      - Number of discretization points in s-direction (coarse mesh)
Nw      - Number of discretization points in w-direction (coarse mesh)
Ns_fine - Number of interpolation points in s-direction
Nw_fine - Number of interpolation points in w-direction
vec     - Function values on coarser mesh; will be <= (Ns_fine * Nw_fine)
ff      - Small value (near zero) to prevent division by zero runtime errors
          Default value of 1e-15
plot    - Boolean that indicates if the interpolated function values should be plotted
rt_space - Boolean that indicates if the interpolation function values should be plotted
           in physical or Radon transform space

Outputs:
S_fine, W_fine - Fine meshgrid used to plot vec_fine_s_w
vec_fine_s_w - Function values on finer mesh
"""
def cfp(Ns, Nw, Ns_fine, Nw_fine, vec, ff=1e-15, plot=False, rt_space=False):
    # Interpolation in the s-direction first
    s = chebyshev(Ns)
    w = np.linspace(0, np.pi, Nw)
    
    s_fine = chebyshev(Ns_fine)
    
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = 0.5
    poly_weights[-1] = (0.5)*poly_weights[-1]

    vec_fine_s = np.zeros((Nw, Ns_fine))

    for row in np.arange(Nw, dtype=int):
        for col in np.arange(Ns_fine, dtype=int):
            vec_fine_s[row, col] = np.sum((poly_weights * vec[row, :]) / (s_fine[col] - s + ff)) / \
                                   np.sum(poly_weights * 1 / (s_fine[col] - s + ff))
    
    # Interpolate in w-direction
    w_fine = np.linspace(0, np.pi, Nw_fine)

    w_bins = np.digitize(w_fine, w)

    vec_fine_s_w = np.zeros((Nw_fine, Ns_fine))

    for col in np.arange(Ns_fine, dtype=int):
        for row in np.arange(Nw_fine, dtype=int):
            idx_1 = w_bins[row] - 1
            idx_2 = w_bins[row] if w_bins[row] < Nw else (idx_1 + 0)

            w_1 = w[idx_1]
            w_2 = w[idx_2]

            vec_fine_s_w[row, col] = vec_fine_s[idx_1, col] * (1 - (w_fine[row] - w_1) / (w_2 - w_1 + ff)) + \
                                     vec_fine_s[idx_2, col] * ((w_fine[row] - w_1) / (w_2 - w_1 + ff))

    [S_fine_s_w, W_fine_s_w] = np.meshgrid(s_fine, w_fine)
    X_fine_s_w = S_fine_s_w * np.cos(W_fine_s_w)
    Y_fine_s_w = S_fine_s_w * np.sin(W_fine_s_w)

    if plot:
        plt.figure(10)
        plt.pcolor(S_fine_s_w*rt_space + X_fine_s_w*(1-rt_space), \
                   W_fine_s_w*rt_space + Y_fine_s_w*(1-rt_space), \
                   vec_fine_s_w)
        plt.colorbar()
        plt.show()
        
    return (S_fine_s_w, W_fine_s_w, vec_fine_s_w)

def pn_coeffs( N, St = 0, Ss = 0 ):
    # Spherical harmonic constants from paper
    A = lambda m, l: np.sqrt( (l-m+1)*(l+m+1)/(2*l+3)/(2*l+1) )
    B = lambda m, l: np.sqrt( (l-m  )*(l+m  )/(2*l+1)/(2*l-1) )
    C = lambda m, l: np.sqrt( (l+m+1)*(l+m+2)/(2*l+3)/(2*l+1) )
    D = lambda m, l: np.sqrt( (l-m  )*(l-m-1)/(2*l+1)/(2*l-1) )
    E = lambda m, l: np.sqrt( (l-m+1)*(l-m+2)/(2*l+3)/(2*l+1) )
    F = lambda m, l: np.sqrt( (l+m  )*(l+m-1)/(2*l+1)/(2*l-1) )

    # Total number of equations
    p_max = (N+1)*(N+2)//2

    # Matrices
    Az = np.zeros((p_max, p_max))
    Ax = np.copy(Az)

    Ac = -np.eye(p_max)*St
    Ac[0,0] = Ac[0,0] + Ss

    p = 0 #p = l(l+1)/2 + 1 + m
    #from functions that Dr. Rossmanith wrote on the board to have one variable instead of two
    for l in range(N+1):
        # Handle m == 0 case
        if 0 <= p-l+1 < p_max:
            Ax[p, p-l+1] = E(1, l-1)
        if 0 <= p+l+2 < p_max:
            Ax[p, p+l+2] = -F(1, l+1)

        if 0 <= p-l < p_max:
            Az[p, p-l] = A(0, l-1)
        if 0 <= p+l+1 < p_max:
            Az[p, p+l+1] = B(0, p+l+1)

        p += 1
        #for all other m values
        for m in range(1, l+1):
            if 0 <= p-l-1 < p_max:
                Ax[p,p-l-1] = -0.5*C(m-1, l-1)
            if 0 <= p+l < p_max:
                Ax[p,p+l]   =  0.5*D(m-1, l+1)
            if 0 <= p-l+1 < p_max:
                Ax[p,p-l+1] =  0.5*E(m+1, l-1)
            if 0 <= p+l+2 < p_max:
                Ax[p,p+l+2] = -0.5*F(m+1, l+1)


            if 0 <= p-l < p_max:
                Az[p,p-l]   = A(m, l-1)
            if 0 <= p+l+1 < p_max:
                Az[p,p+l+1] = B(m, l+1)

            p += 1

    return Ax, Az, Ac

