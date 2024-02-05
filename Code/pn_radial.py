import numpy as np
import radon.utilities as util
import radon.transform as rt
import radon.advection as ra
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as sla
#import scipy.linalg as sla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla
import radial_radon_transform as rrt
import radon.plotting as rp
from pathlib import Path
import os
import inspect

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def pull_radial_matrix( Ns, Nq ):
    save_fname = f"Radial_R_{Ns}_{Nq}"

    load_fname = save_fname + ".npy"

    matrix_file = Path(load_fname)

    if matrix_file.is_file():
        print("Matrix exists")

        R = np.load(load_fname)
    else:
        print("Matrix DOES NOT exist")

        R = rt.radial_matrix( Ns, Nq )
        np.save(save_fname, R)

    return R

if __name__ == "__main__":
    Ns = 201
    Nw = 201
    Nq = 75

    N = 5

    a = 1.5
    alpha = 0.03

    dt = 0.001
    tf = 1.0/a
    order=1

    siga = 0.0
    sigs = 1.0
    sigt = siga + sigs

    def pn_init(x,y):
        return (1/(4*np.pi*alpha**2))*np.exp(-(((a*x)**2+(a*y)**2)/(4*alpha**2)))

    def spn_init(x, y):
        return pn_init(x-0.2, y-0.2) + pn_init(x+0.2, y+0.2)

    def g(x,y):
        return 0*x

    print(f"N = {N}")

    folder = f"p{N}_plots"
    os.system( "mkdir " + folder + '/' )

    Ax, Az, Ac = util.pn_coeffs( N=N, St=sigt, Ss=sigs)

    A = Ax
    B = Az
    C = Ac*a

    s = util.chebyshev(Ns)
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)
    X, Y = S*np.cos(W), S*np.sin(W)

    m = A.shape[0]

    f0s = [pn_init] + [g]*(m-1)

    f_vec_t0 = pn_init( X, Y )
    f_hat_t0s = rt.radon_system_exact( f0s, Ns, Nw, Nq )

    rp.plot_physical_space( f_vec_t0.reshape((Nw, Ns)), title=f"$P_{{{N}}}$ at $t=0$",
                                                        filename=(folder+f"/physical_initial_p{N}.pdf"),
                                                        scale=a, plot_slice=True )

    rp.plot_radon_space( f_hat_t0s[0].reshape((Nw, Ns)), title=f"$P_{{{N}}}$ at $t=0$ in Radon space",
                                                         filename=(folder+f"/radon_initial_p{N}.pdf"),
                                                         scale=a, plot_slice=True )

    pn = ra.pn_approx( tf, Ns, Nw, dt, A, B, C, f_hat_t0s, order=order, radial=True )
    #pn = np.load(folder + "/pn.npy" )
    np.save(folder + "/pn.npy", pn )

    #""" Solve with Matrix
    R = pull_radial_matrix( Ns, Nq )
    f_vec_tf = np.linalg.solve( R.T @ R, R.T @ pn[0][0,:].flatten() ).reshape((1, Ns))
    f_vec_tf = f_vec_tf * np.ones((Nw, 1))
    #"""

    """ Solve with BICGSTAB
    x0 = np.zeros((Nw*Ns,))
    b_vec = rt.backprojection( pn[0].flatten(), Ns, Nw, Nq )
    b_norm = np.linalg.norm( b_vec, 2 )
    counter = 0
    def callback(xk):
        global counter, X, Y, Nw, Ns, b_norm
        np.save( folder + f'/f_vec_tf_{counter}.npy', xk )
        plt.figure(6)
        plt.contourf( X, Y, xk.reshape((Nw, Ns)), levels=50 )
        plt.contourf( X, Y, xk.reshape((Nw, Ns)), levels=50 )
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig( folder + f'/f_vec_tf_{counter}.pdf' )
        plt.clf()

        plt.figure(7)
        plt.plot( s, xk.reshape((Nw, Ns))[0,:] )
        plt.savefig( folder + f'/f_vec_tf_slice_{counter}.pdf' )
        counter += 1
        frame = inspect.currentframe().f_back
        print( '\t', counter, '\t', frame.f_locals['resid']/b_norm )

    #f_vec_tf, info = sla.bicgstab(rt.backprojection_operator(Ns, Nw, Nq), b_vec, \
    #                                     x0=x0, tol=0.000001, callback=callback, atol=0.000001)
    f_vec_tf = np.load( folder + "/f_vec_tf_29.npy" )
    #"""

    rp.plot_physical_space( f_vec_tf.reshape((Nw, Ns)), title=f"$P_{{{N}}}$ at $t=1.0$",
                                                        filename=(folder+f"/physical_final_p{N}.pdf"),
                                                        scale=a, plot_slice=True )

    rp.plot_radon_space( pn[0].reshape((Nw, Ns)), title=f"$P_{{{N}}}$ at $t=1.0$ in Radon space",
                                                  filename=(folder+f"/radon_final_p{N}.pdf"),
                                                  scale=a, plot_slice=True )


