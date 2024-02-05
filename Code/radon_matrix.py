import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import radon_transform as rt
import radon_transform_exact as rte
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import LinearOperator
import radon_basis as rb
import radon_basis_unif as rbu
import radon_basis_vec as rbv
import radon_basis_quad as rbq
import radon_basis_linear as rbl
import radon_basis_bilinear as rbbl
np.set_printoptions(formatter={'float': lambda x: "{0: 0.3f}".format(x)})

def radon_matrix( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            if j <= pad - 1 or j == Ns-pad:
                sR[p,p] = 1.0
            else:
                f_vec[p] = 1
                sR[:, p] = rbv.radon( i, j, Ns, Nw, Nq ).reshape((-1, 1))
                f_vec[p] = 0
        print("Column", p)
    return sR.tocsc()

def radon_matrix_long( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            if j <= pad - 1 or j >= Ns-pad:
                sR[p,p] = 1.0
            else:
                f_vec[p] = 1
                sR[:, p] = rbq.radon( i, j, Ns, Nw, Nq ).reshape((-1, 1))
                f_vec[p] = 0
        #print("Column", p)
    return sR.tocsc()

def radon_matrix_linear( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            if j <= pad - 1 or j >= Ns-pad:
                sR[p,p] = 1.0
            else:
                f_vec[p] = 1
                sR[:, p] = rbl.radon( i, j, Ns, Nw, Nq ).reshape((-1, 1))
                f_vec[p] = 0
        #print("Column", p)
    return sR.tocsc()

def radon_matrix_bilinear( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            if j <= pad - 1 or j >= Ns-pad:
                sR[p,p] = 1.0
            else:
                f_vec[p] = 1
                sR[:, p] = rbbl.radon( i, j, Ns, Nw, Nq ).reshape((-1, 1))
                f_vec[p] = 0
        #print("Column", p)
    return sR.tocsc()


def radon_matrix_unif( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            if j <= pad - 1 or j >= Ns-pad:
                sR[p,p] = 1.0
            else:
                f_vec[p] = 1
                sR[:, p] = rbu.radon( i, j, Ns, Nw, Nq ).reshape((-1, 1))
                f_vec[p] = 0
        #print("Column", p)
    return sR.tocsc()

def radon_matrix_slow( Ns, Nw, Nq ):
    Np = Ns*Nw
    sR = lil_matrix( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            f_vec[p] = 1
            sR[:, p] = rt.radon( f_vec, Ns, Nw, Nq ).reshape((-1, 1))
            f_vec[p] = 0
            if j == 0 or j == Ns-1:
                sR[p,p] = 1.0
        #print("Column", p)
    return sR.tocsc()

def preconditioner( M, drop_tol=1e-7, fill_factor=40 ):
    return LinearOperator( M.shape, spilu(M, drop_tol=drop_tol, fill_factor=fill_factor).solve )

if __name__ == "__main__":
    Ns = 25
    Nw = 24
    Nq = 5

    a = 0.3
    b = 0.05
    #f = lambda x, y: np.exp(-(x/a)**2 - (y/b)**2)
    f = lambda x, y: (1 - x*x - y*y)*(x-0.5)*(y+0.5)
    def unit_square( x, y ):
        return np.array( np.greater(x, 0) & np.greater(y, 0) & np.less(x, 0.5) &
                        np.less( y, 0.5), dtype=float )
    #f = unit_square
    Np = Ns*Nw

    # Test creation of Radon transform matrices
    f_vec = np.zeros((Np, ))
    R1 = np.zeros((Np, Np))
    R2 = np.zeros((Np, Np))

    for i in range(Nw):
        for j in range(Ns):
            p = i*Ns + j
            f_vec[p] = 1

            R1[:, p] = rt.radon(f_vec, Ns, Nw, Nq)
            R2[:, p] = rbv.radon(i, j, Ns, Nw, Nq)

            #for k in range(Np):
            #    print( "{: .8f} {: .8f}".format(R1[k,p], R2[k,p]) )
            #print(p, np.allclose(np.sort(R1[:,p]), np.sort(R2[:,p])))
            if not np.allclose(R1[:,p], R2[:,p]):
                print(i, j)
            #input()
            #fgg
            if j == 0 or j == (Ns - 1):
                R1[p, p] = 1
                R2[p, p] = 1

            f_vec[p] = 0

    print( np.allclose( R1, R2 ) )
    s = rt.clen_curt(Ns)[0]
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)
    f_vec = f(S*np.cos(W), S*np.sin(W)).flatten()

    #f_hat_exact = np.zeros_like(f_vec).reshape((Nw, Ns))
    #denom = (a*np.cos(W))**2 + (b*np.sin(W))**2
    #f_hat_exact[:, :] = (a*b*np.sqrt(np.pi))/(np.sqrt(denom))*np.exp(-(s*s)/(denom))
    #f_hat_exact = f_hat_exact.flatten()

    f_hat_exact = rt.radon( f_vec, Ns, Nw, Nq )

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot( 111, projection='3d' )
    ax1.plot_wireframe( S*np.cos(W), S*np.sin(W), f_vec.reshape((Nw, Ns)), color='blue' )
    ax1.set_title( 'original function' )

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot( 111, projection='3d' )
    ax2.plot_wireframe( S, W, f_hat_exact.reshape((Nw, Ns)), color='green', label='direct operator' )
    ax2.plot_wireframe( S, W, (R1 @ f_vec).reshape((Nw, Ns)), color='red', label='sin interpolate' )
    ax2.plot_wireframe( S, W, (R2 @ f_vec).reshape((Nw, Ns)), color='blue', label='reg interpolate' )
    ax2.legend()

    plt.show()
