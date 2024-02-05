import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm
from radon import utilities as util
from radon import transform

scale = 1

sig = 1

def plot_stuff( Ns ):
    Nw = 5
    Nq = 75
    a = -1.0
    b = 1.0

    scale = 1

    sig = 1
    # Get rid of scattering and absorption
    # sig = 0

    r = util.chebyshev(Ns, a=a, b=b)

    w = np.linspace(0, np.pi, Nw)

    [S, W] = np.meshgrid(scale*r, w)

    # Differentiation matrices
    D1 = util.diff_matrix(Ns, a=a, b=b)
    D2 = np.copy(D1)

    # Get rid of transport
    # D1 = np.zeros((Ns, Ns))
    # D2 = np.zeros((Ns, Ns))

    # Zero out first and last rows, first and last columns
    # to enforce inflow condition
    D1[0, :] = D1[:, 0] = 0
    D2[-1, :] = D2[:, -1] = 0

    ff = 0

    # F11 = lambda r: (1.0/(2 * np.sqrt(3) * r + ff) - scale * sig/2.0)

    def F11(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (1.0 / (2 * np.sqrt(3) * r[i]) - scale * sig/2.0)
            else:
                result[i] = -scale * sig/2.0

        return result
    
    # F12 = lambda r: (-1.0/(2 * np.sqrt(3) * r + ff) + scale * sig/2.0)

    def F12(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (-1.0 / (2 * np.sqrt(3) * r[i]) + scale * sig/2.0)
            else:
                result[i] = scale * sig/2.0

        return result
    
    # F21 = lambda r: (1.0/(2 * np.sqrt(3) * r + ff) + scale * sig/2.0)

    def F21(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (1.0 / (2 * np.sqrt(3) * r[i]) + scale * sig/2.0)
            else:
                result[i] = scale * sig/2.0

        return result
    
    # F22 = lambda r: (-1.0/(2 * np.sqrt(3) * r + ff) - scale * sig/2.0)

    def F22(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (-1.0 / (2 * np.sqrt(3) * r[i]) - scale * sig/2.0)
            else:
                result[i] = -scale * sig/2.0

        return result

    # Introduce damping
    # F11 = lambda r: (0*r - scale * sig/2.0 - scale/2.0)
        
    # F12 = lambda r: (0*r + scale * sig/2.0 - scale/2.0)
    
    # F21 = lambda r: (0*r + scale * sig/2.0 - scale/2.0)
    
    # F22 = lambda r: (0*r - scale * sig/2.0 - scale/2.0)

    # Matrix on RHS
    A = np.zeros((2*Ns, 2*Ns))
    
    A[0:Ns, 0:Ns] = (1.0/np.sqrt(3)) * D1 + np.diag(F11(r))

    A[0:Ns, Ns:(2*Ns)] = np.diag(F12(r))
    
    A[Ns:(2*Ns), 0:Ns] = np.diag(F21(r))

    A[Ns:(2*Ns), Ns:(2*Ns)] = (-1.0/np.sqrt(3)) * D2 + np.diag(F22(r))

    
    # Initial conditions   
    alpha = 0.03

    tf = 1.0/scale
    
    def f(r):
        return (1/(4*np.pi*alpha**2))*np.exp(-((scale*r)**2)/(4*alpha**2))

    p_t0 = f(r)
    ur_t0 = np.zeros_like(p_t0)

    w1_t0 = (1.0/2.0) * (p_t0 - ur_t0)
    w2_t0 = (1.0/2.0) * (p_t0 + ur_t0)

    w_vec_t0 = np.zeros((2*Ns, ))
    w_vec_t0[0:Ns] = np.copy(w1_t0)
    w_vec_t0[Ns:(2*Ns)] = np.copy(w2_t0)

    w_vec_tf = expm(tf*A) @ w_vec_t0

    w1_tf = np.copy(w_vec_tf[0:Ns])
    w2_tf = np.copy(w_vec_tf[Ns:(2*Ns)])

    p_tf = w1_tf + w2_tf

    ###########################################################################
    # Fine resolution slices
    ###########################################################################
    plt.figure(1)
    plt.plot(scale*r, p_t0, label=str(Ns))
    plt.title(r"$P_{1}$ slice at time $t = 0$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p$")

    plt.figure(2)
    plt.plot(scale*r, p_tf, label=str(Ns))
    plt.title(r"$P_{1}$ slice at time $t = " + str(scale * tf) + "$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p$")

if __name__ == "__main__":
    #for Ns in [50, 100, 150, 200, 250, 300, 350, 400]:
    #    plot_stuff( Ns )
    #plt.legend()
    #plt.show()

    Ns = int(sys.argv[1])
    Nw = int(sys.argv[2])
    Nq = int(sys.argv[3])

    a = -1.0
    b = 1.0

    scale = 1.5

    sig = 1
    # Get rid of scattering and absorption
    # sig = 0

    r = util.chebyshev(Ns, a=a, b=b)

    w = np.linspace(0, np.pi, Nw)

    [S, W] = np.meshgrid(scale*r, w)

    # Differentiation matrices
    D1 = util.diff_matrix(Ns, a=a, b=b)
    D2 = np.copy(D1)

    # Get rid of transport
    # D1 = np.zeros((Ns, Ns))
    # D2 = np.zeros((Ns, Ns))

    # Zero out first and last rows, first and last columns
    # to enforce inflow condition
    D1[0, :] = D1[:, 0] = 0
    D2[-1, :] = D2[:, -1] = 0

    ff = 0

    # F11 = lambda r: (1.0/(2 * np.sqrt(3) * r + ff) - scale * sig/2.0)

    def F11(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (1.0 / (2 * np.sqrt(3) * r[i]) - scale * sig/2.0)
            else:
                result[i] = -scale * sig/2.0

        return result
    
    # F12 = lambda r: (-1.0/(2 * np.sqrt(3) * r + ff) + scale * sig/2.0)

    def F12(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (-1.0 / (2 * np.sqrt(3) * r[i]) + scale * sig/2.0)
            else:
                result[i] = scale * sig/2.0

        return result
    
    # F21 = lambda r: (1.0/(2 * np.sqrt(3) * r + ff) + scale * sig/2.0)

    def F21(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (1.0 / (2 * np.sqrt(3) * r[i]) + scale * sig/2.0)
            else:
                result[i] = scale * sig/2.0

        return result
    
    # F22 = lambda r: (-1.0/(2 * np.sqrt(3) * r + ff) - scale * sig/2.0)

    def F22(r):
        global scale, sig
        result = np.zeros_like(r)
        for i in np.arange(np.max(r.shape), dtype=int):
            if r[i] != 0:
                result[i] = (-1.0 / (2 * np.sqrt(3) * r[i]) - scale * sig/2.0)
            else:
                result[i] = -scale * sig/2.0

        return result

    # Introduce damping
    # F11 = lambda r: (0*r - scale * sig/2.0 - scale/2.0)
        
    # F12 = lambda r: (0*r + scale * sig/2.0 - scale/2.0)
    
    # F21 = lambda r: (0*r + scale * sig/2.0 - scale/2.0)
    
    # F22 = lambda r: (0*r - scale * sig/2.0 - scale/2.0)

    # Matrix on RHS
    A = np.zeros((2*Ns, 2*Ns))
    
    A[0:Ns, 0:Ns] = (1.0/np.sqrt(3)) * D1 + np.diag(F11(r))

    A[0:Ns, Ns:(2*Ns)] = np.diag(F12(r))
    
    A[Ns:(2*Ns), 0:Ns] = np.diag(F21(r))

    A[Ns:(2*Ns), Ns:(2*Ns)] = (-1.0/np.sqrt(3)) * D2 + np.diag(F22(r))

    
    # Initial conditions   
    alpha = 0.03

    tf = 1.0/scale
    
    def f(r):
        return (1/(4*np.pi*alpha**2))*np.exp(-((scale*r)**2)/(4*alpha**2))

    p_t0 = f(r)
    ur_t0 = np.zeros_like(p_t0)

    w1_t0 = (1.0/2.0) * (p_t0 - ur_t0)
    w2_t0 = (1.0/2.0) * (p_t0 + ur_t0)

    w_vec_t0 = np.zeros((2*Ns, ))
    w_vec_t0[0:Ns] = np.copy(w1_t0)
    w_vec_t0[Ns:(2*Ns)] = np.copy(w2_t0)

    w_vec_tf1 = expm(tf*A) @ w_vec_t0
    w_vec_tf2 = np.copy(w_vec_t0)
    for i in range(100):
        w_vec_tf2 = expm( tf/100 * A ) @ w_vec_tf2

    print( np.linalg.norm( w_vec_tf1 - w_vec_tf2, np.inf ) )

    ggg
    w1_tf = np.copy(w_vec_tf[0:Ns])
    w2_tf = np.copy(w_vec_tf[Ns:(2*Ns)])

    p_tf = w1_tf + w2_tf

    ###########################################################################
    # Fine resolution slices
    ###########################################################################
    plt.figure(1)
    plt.plot(r, p_t0, "k")
    plt.title(r"$P_{1}$ slice at time $t = 0$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p$")

    np.save( f'p1_physical_slice_{Ns}.npy', p_tf )
    plt.figure(2)
    plt.plot(r, p_tf, "b")
    plt.title(r"$P_{1}$ slice at time $t = " + str(scale * tf) + "$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$p$")
    #"""
    ###########################################################################
    # Fine resolution contour plots
    ###########################################################################
    levels = 50
    #"""
    # Fine resolution contour plot of initial conditions
    # in physical space
    p_t0_contour = np.zeros((Nw, Ns))
    p_t0_contour[:, :] = np.copy(p_t0)
    #"""
    plt.figure(3)
    plt.contourf(S*np.cos(W), S*np.sin(W), p_t0_contour, levels=levels)
    plt.colorbar()
    plt.title(r"$P_{1}$ in physical space at time $t = 0$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    # Fine resolution contour plot of initial conditions
    # in RT space
    #"""
    p_t0_frt = transform.radon(p_t0_contour.flatten(), Ns, Nw, Nq)
    p_t0_frt = p_t0_frt.reshape((Nw, Ns))
    #"""
    plt.figure(4)
    plt.contourf(S, W, p_t0_frt, levels=levels)
    plt.colorbar()
    plt.title(r"$P_{1}$ in RT space at time $t = 0$")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\omega$")
    #"""
    # Fine resolution contour plot of final solution
    # in physical space
    p_tf_contour = np.zeros((Nw, Ns))
    p_tf_contour[:, :] = np.copy(p_tf)
    #"""
    plt.figure(5)
    plt.contourf(S*np.cos(W), S*np.sin(W), p_tf_contour, levels=levels)
    plt.colorbar()
    plt.title(r"$P_{1}$ in physical space at time $t = " + str(scale * tf) + "$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    #"""
    # Fine resolution contour plot of final solution
    # in RT space
    p_tf_frt = transform.radial(p_tf_contour.flatten(), Ns, Nw, Nq)
    p_tf_frt = p_tf_frt.reshape((Nw, Ns))
    #"""
    plt.figure(6)
    plt.contourf(S, W, p_tf_frt, levels=levels)
    plt.colorbar()
    plt.title(r"$P_{1}$ in RT space at time $t = " + str(scale * tf) + "$")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$\omega$")
    #"""
    plt.figure(7)
    plt.plot(scale*r, p_t0_frt[0, :], "r")
    plt.plot(scale*r, p_tf_frt[0, :], "b")
    np.save( f'p1_radon_slice_{Ns}.npy', p_tf_frt[0,  :] )
    plt.title(r"Slices of $P_{1}$ in RT space at time $t = 0$ and $t = " + str(scale*tf) + "$, respectively")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$p$")
    plt.show()
    #"""
    plt.show()
    #"""
