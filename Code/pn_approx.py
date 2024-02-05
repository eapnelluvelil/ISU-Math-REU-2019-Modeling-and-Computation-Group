import numpy as np
import sys
#makes matrices print nice
np.set_printoptions(formatter={'float': lambda x: "{0: 0.3f}".format(x)})
"""
psi order:

l = 0   1
l = 1   2 3
l = 2   4 5 6
l = 3   7 8 9 10
etc.
"""

# Spherical harmonic constants from paper
A = lambda m, l: np.sqrt( (l-m+1)*(l+m+1)/(2*l+3)/(2*l+1) )
B = lambda m, l: np.sqrt( (l-m  )*(l+m  )/(2*l+1)/(2*l-1) )
C = lambda m, l: np.sqrt( (l+m+1)*(l+m+2)/(2*l+3)/(2*l+1) )
D = lambda m, l: np.sqrt( (l-m  )*(l-m-1)/(2*l+1)/(2*l-1) )
E = lambda m, l: np.sqrt( (l-m+1)*(l-m+2)/(2*l+3)/(2*l+1) )
F = lambda m, l: np.sqrt( (l+m  )*(l+m-1)/(2*l+1)/(2*l-1) )

def pn_matrices( N, St = 0, Ss = 0 ):
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


if __name__ == "__main__":
    N = 3
    #calls function with N value set in the terminal
    Ax, Az, Ac = pn_matrices(int(sys.argv[1]), 1, 1)

    #prints matrix Ax and Az
    print( sorted(np.linalg.eig( Ax )[0]) )
    print( sorted(np.linalg.eig( Az )[0]) )
    #prints matrix Ac
    print( Ac )
