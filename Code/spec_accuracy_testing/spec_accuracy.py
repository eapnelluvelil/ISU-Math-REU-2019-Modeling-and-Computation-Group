import sys
sys.path.append("/home/esn2/Projects/isu_reu_2019/Code/")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import radon.transform as rt
import radon.advection as ra
import radon.utilities as util

def test_spec_diff_accuracy():
    alpha = 0.03
    init_cond_slice = lambda s: (1.0/(4*np.pi*alpha**2))*np.exp(-(s**2) * (1.0/(4*alpha**2)))
    init_cond_slice_deriv = lambda s: (-1.0) * s*np.exp(-(s**2) * (1.0/(4*alpha**2))) * (1.0/(8*np.pi*alpha**4))

    # Ns_vals = np.array([25, 50, 100, 200])
    # Ns_vals = np.arange(25, 201, dtype=int)
    Ns_vals = 2**np.arange(5, 11, dtype=int)

    rel_errs = []

    for Ns in Ns_vals:
        s = util.chebyshev(Ns)
        D = util.diff_matrix(Ns)

        init_cond_slice_evals = init_cond_slice(s)
        init_cond_slice_approx_deriv = D @ init_cond_slice_evals
        init_cond_slice_actual_deriv = init_cond_slice_deriv(s)

        rel_errs.append(np.linalg.norm(init_cond_slice_approx_deriv - init_cond_slice_actual_deriv, np.inf) / np.linalg.norm(init_cond_slice_actual_deriv, np.inf))

    plt.figure(1)
    plt.semilogy(Ns_vals, np.array(rel_errs))
    plt.show()

def test_pn_spec_accuracy():
    Ns = int(sys.argv[1])
    Nw = int(sys.argv[2])
    Nq = int(sys.argv[3])
    N = int(sys.argv[4])

    # Initial conditions
    scale = 1.5
    
    alpha = 0.03
    
    def f(x,y):
        return (1/(4*np.pi*alpha**2))*np.exp(-(((scale*x)**2+(scale*y)**2)/(4*alpha**2)))

    def g(x,y):
        return 0*x
    
    dt = 0.001
    
    tf = 1.0/scale

    siga = 0.0
    sigs = 1.0
    sigt = siga + sigs

    Ax, Az, Ac = util.pn_coeffs( N=N, St=sigt, Ss=sigs)

    A = Ax
    B = Az
    C = Ac*scale

    s = util.chebyshev(Ns)
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    [S, W] = np.meshgrid(s, w)
    X = S*np.cos(W)
    Y = S*np.sin(W)

    m = A.shape[0]

    f0s = [f] + [g]*(m-1)
    
    f_hat_t0s = rt.radon_system_exact( f0s, Ns, Nw, Nq )

    domain = [-1.5/scale, 1.5/scale]

    order = 1

    pn = ra.pn_approx_expm(tf, Ns, Nw, A, B, C, f_hat_t0s, domain=domain, radial=True)

    # Plotting parameters
    levels = 100

    # Initial condition in Radon space
    plt.figure(1)
    plt.contourf(S, W, f_hat_t0s[0].reshape((Nw, Ns)))
    plt.colorbar()
    
    # Final solution in Radon space
    plt.figure(2)
    plt.contourf(S, W, pn[0], levels=levels)
    plt.colorbar()

    # Slice of final solution in Radon space
    plt.figure(3)
    plt.plot(S[0, :], pn[0][0, :])
    
    plt.show()

if __name__ == "__main__":
    # test_spec_diff_accuracy()
    test_pn_spec_accuracy()
