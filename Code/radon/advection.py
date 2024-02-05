"""
radon_advection.py

Contains methods to perform various advection
    wave        := Does wave_system for each omega
    wave_system := Calculates 1D wave equation solution
    pn_approx   := Does pn_approx for each omega
    pn_system   := Calculates 1D Pn equation solution
"""
import numpy as np
import radon.utilities as util
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as sla

def wave( tf, Ns, Nw, dt, A, B, f_hat_t0s, domain=[-1, 1], order=2 ):
    w = np.linspace( 0, np.pi, Nw+1 )[:-1]
    m = A.shape[0]
    f_hat_tfs = [np.zeros(( Nw, Ns )) for i in range(m)]

    for i in range( Nw ):
        q0s = np.zeros(( Ns, m ))
        qfs = np.zeros(( Ns, m ))

        for p in range( m ):
            q0s[:,p] = f_hat_t0s[p][i,:]

        qfs = wave_system( tf = tf,
                           N = Ns,
                           dt = dt,
                           A = np.cos(w[i])*A + np.sin(w[i])*B,
                           q0s = q0s,
                           domain = domain,
                           order = order )

        for p in range( m ):
            f_hat_tfs[p][i,:] = qfs[:,p]

    return f_hat_tfs

def wave_system( tf, N, dt, A, q0s, domain=[-1, 1], order=2 ):
    [a, b] = domain

    Nt = int(tf // dt) #number of time steps
    dt = tf / Nt #reassigning time step

    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )

    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    D0 = util.diff_matrix( N, a, b )

    # Create list of time-stepping operators
    # that correspond to w_1, w_2, ..., w_m
    Operators = []
    for i in range(A.shape[0]):
        #the difference matrix scaled by the lambda value
        D = dt*Lam[i]*D0

        # Impose boundary conditions
        if Lam[i] < 0 and order in [1, 2, 4]:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        elif order in [1, 2, 4]:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        # For the TR-BDF2 method, the above rules are flipped
        if Lam[i] > 0 and order not in [1, 2, 4]:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        elif order not in [1, 2, 4]:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        # Backward Euler rule
        if order == 1:
            Operators.append( np.linalg.inv( np.eye(N) + D ) )
        # Trapezoidal rule
        elif order == 2:
            Operators.append( np.linalg.inv(np.eye(N) + D/2) @ ( np.eye(N) - D/2 ) )
        # 4th order extension of trapezoidal rule
        elif order == 4:
            Operators.append( np.linalg.inv(np.eye(N) + D/2 + D@D/12) \
                            @ ( np.eye(N) - D/2 + D@D/12 ) )
        # TR-BDF2 method
        else:
            Operators.append( np.linalg.inv(3 * np.eye(N) - D) \
                              @ (4 * np.linalg.inv(np.eye(N) - D/4) @ (np.eye(N) + D/4) - np.eye(N) ) )

    t = 0

    # Array of initial conditions
    W0 = q0s @ np.linalg.inv(R).T
    for i in range( int(Nt) ):
        for k in range(A.shape[0]):
            W0[:, k] = Operators[k] @ W0[:, k]

        t += dt

    return W0 @ R.T


def pn_approx( tf, Ns, Nw, dt, A, B, C, f_hat_t0s, domain=[-1, 1], order=3, radial=False ):
    w = np.linspace( 0, np.pi, Nw+1 )[:-1]
    m = A.shape[0]
    f_hat_tfs = [np.zeros(( Nw, Ns )) for i in range(m)]

    if radial == True:
        loops = 1
    else:
        loops = Nw

    for i in range( loops ):
        print( i )
        q0s = np.zeros(( Ns, m ))
        qfs = np.zeros(( Ns, m ))

        for p in range(m):
            q0s[:,p] = f_hat_t0s[p][i,:]

        qfs = pn_system( tf = tf,
                         N = Ns,
                         dt = dt,
                         A = np.cos(w[i])*A + np.sin(w[i])*B,
                         C = C,
                         q0s = q0s,
                         domain = domain,
                         order=order)
        np.save( f"slices/qfs_{i}.npy", qfs )
        s = util.chebyshev( Ns )
        #plt.plot( s, q0s[:,0], 'r' )
        #plt.plot( s, qfs[:,0], 'b' )
        #plt.show()
        #np.save( 'p11_200_slice.npy', qfs[:,0] )
        for p in range( m ):
            f_hat_tfs[p][i,:] = qfs[:,p] #complex value error happens here when i = 1

    if radial == True:
        for i in range( 1, Nw ):
            for p in range( m ):
                f_hat_tfs[p][i,:] = qfs[:,p]

    return f_hat_tfs

def pn_approx_expm( tf, Ns, Nw, A, B, C, f_hat_t0s, domain=[-1, 1], radial=False ):
    w = np.linspace( 0, np.pi, Nw+1 )[:-1]
    m = A.shape[0]
    f_hat_tfs = [np.zeros(( Nw, Ns )) for i in range(m)]

    if radial == True:
        loops = 1
    else:
        loops = Nw

    for i in range( loops ):
        print( i )
        q0s = np.zeros(( Ns, m ))
        qfs = np.zeros(( Ns, m ))

        for p in range(m):
            q0s[:,p] = f_hat_t0s[p][i,:]

        qfs = pn_system_expm( tf=tf,
                         N=Ns,
                         A=(np.cos(w[i])*A + np.sin(w[i])*B),
                         C=C,
                         q0s=q0s,
                         domain=domain)

        for p in range( m ):
            f_hat_tfs[p][i,:] = qfs[:,p] #complex value error happens here when i = 1

    if radial == True:
        for i in range( 1, Nw ):
            for p in range( m ):
                f_hat_tfs[p][i,:] = qfs[:,p]

    return f_hat_tfs


def pn_system( tf, N, dt, A, C, q0s, domain=[-1, 1], order=3 ):
    [a, b] = domain

    al = 0.24169426078821
    be = 0.06042356519705
    et = 0.1291528696059

    Ac = np.array( [[al ,    0,        0,  0],
                    [-al,   al,        0,  0],
                    [  0, 1-al,       al,  0],
                    [ be,   et, .5-be-et, al]] )
    wc = np.array([0, 1/6, 1/6, 2/3])

    At = np.array( [[0,   0   ,0, 0],
                    [0,   0,   0, 0],
                    [0,   1,   0, 0],
                    [0, .25, .25, 0]] )
    wt = np.array([0, 1/6, 1/6, 2/3])

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step

    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )

    F = np.linalg.inv( R ) @ C @ R

    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    D0 = util.diff_matrix( N, a, b )
    s = util.chebyshev( N )

    # Create list of time-stepping operators
    # that correspond to w_1, w_2, ..., w_m
    Diff_Ops = []
    Diff_Inv = []
    for i in range(A.shape[0]):
        #the difference matrix scaled by the lambda value
        D = np.copy(Lam[i]*D0)

        # Impose boundary conditions
        if Lam[i] < 0:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        else:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        Diff_Ops.append( D )
        if order == 1:
            Diff_Inv.append( np.linalg.inv( np.eye( N ) + dt*D ) )
        if order == 3:
            Diff_Inv.append( np.linalg.inv( np.eye( N ) + dt*al*D ) )

    # Array of initial conditions
    WN = q0s @ np.linalg.inv(R).T
    W1 = np.zeros_like( WN )
    W2 = np.zeros_like( WN )
    W3 = np.zeros_like( WN )
    W4 = np.zeros_like( WN )

    t = 0

    M = A.shape[0]
    """
    Q = WN @ R.T

    fig1, axs = plt.subplots(2, 2, sharex='col')
    axs[0,0].plot([0],[0], 'k', label='$t_0$')
    axs[0,0].plot([0],[0], 'k--', label='$t_f$')
    axs[0,0].plot( s, Q[:,0], 'r' )
    axs[1,0].plot( s, Q[:,1], 'g' )

    axs[0,1].plot( s, WN[:,0], 'c' )
    axs[0,1].set_ylim([-0.5, 5])
    axs[1,1].plot( s, WN[:,1], 'y' )
    axs[1,1].set_ylim([-5, 0.5])
    #"""
    if order == 3:
        for i in range( int(Nt) ):
            for p in range(M):
                W1[:, p] = Diff_Inv[p] @ WN[:, p]

            for p in range(M):
                W2[:, p] = Diff_Inv[p] @ ( WN[:, p] - dt*Ac[1][0]*Diff_Ops[p] @ W1[:, p] )

            for p in range(M):
                W3[:, p] = Diff_Inv[p] @ ( WN[:, p] \
                                          + dt*At[2][1]*np.sum( F[p, :] * W2[:,:], axis=1 ) \
                                          - dt*Ac[2][0]*Diff_Ops[p] @ W1[:, p] \
                                          - dt*Ac[2][1]*Diff_Ops[p] @ W2[:, p] )

            for p in range(M):
                W4[:, p] = Diff_Inv[p] @ ( WN[:, p] \
                                          + dt*At[3][1]*np.sum( F[p, :] * W2[:,:], axis=1 ) \
                                          + dt*At[3][2]*np.sum( F[p, :] * W3[:,:], axis=1 ) \
                                          - dt*Ac[3][0]*Diff_Ops[p] @ W1[:, p] \
                                          - dt*Ac[3][1]*Diff_Ops[p] @ W2[:, p] \
                                          - dt*Ac[3][2]*Diff_Ops[p] @ W3[:, p] )
            for p in range(M):
                WN[:, p] = WN[:, p] + dt*( wt[1]*np.sum(F[p,:] * W2[:,:], axis=1) + \
                                           wt[2]*np.sum(F[p,:] * W3[:,:], axis=1) + \
                                           wt[3]*np.sum(F[p,:] * W4[:,:], axis=1) ) \
                                    - dt*( wc[1]*Diff_Ops[p] @ W2[:,p] + \
                                           wc[2]*Diff_Ops[p] @ W3[:,p] + \
                                           wc[3]*Diff_Ops[p] @ W4[:,p] )
            #print( t )
            t += dt
    else:
        for i in range( int(Nt) ):
            for p in range(M):
                W1[:, p] = Diff_Inv[p] @ ( WN[:, p] + dt*np.sum( F[p,:] * WN[:,:], axis=1 ) )
            WN = W1[:,:]

            #print( t )
            t += dt
    """
    Q = WN @ R.T
    axs[0,0].plot( s, Q[:,0], 'r--' )
    axs[1,0].plot( s, Q[:,1], 'g--' )
    #axs[2,0].plot( s, Q[:,2], 'b--' )
    axs[0,0].set_title( "Physical Solution" )
    axs[0,0].legend(loc=2, fontsize=15)

    axs[0,1].plot( s, WN[:,0], 'c--' )
    axs[1,1].plot( s, WN[:,1], 'y--' )
    #axs[2,1].plot( s, WN[:,2], 'm--' )
    axs[0,1].set_title( "Characteristic Solution" )

    fig1.subplots_adjust(hspace=0, wspace=0.15)
    plt.show()
    #"""
    return WN @ R.T

def pn_system_expm( tf, N, A, C, q0s, domain=[-1, 1]):
    [a, b] = domain

    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )

    F = np.linalg.inv( R ) @ C @ R

    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    D0 = util.diff_matrix( N, a, b )

    # Create list of time-stepping operators
    # that correspond to w_1, w_2, ..., w_m
    Diff_Ops = []
    
    for i in range(A.shape[0]):
        #the difference matrix scaled by the lambda value
        D = np.copy(Lam[i]*D0)

        # Impose boundary conditions
        if Lam[i] < 0:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        else:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        Diff_Ops.append( D )

    # Array of initial conditions
    WN = q0s @ np.linalg.inv(R).T

    M = A.shape[0]
    
    big_mat = np.zeros((M*N, M*N))
    
    for i in np.arange(M, dtype=int):
        for j in np.arange(M, dtype=int):
            if i == j:
                big_mat[i*N:(i+1)*N, i*N:(i+1)*N] = Diff_Ops[i] - F[i, i] * np.eye(N)
            else:
                big_mat[i*N:(i+1)*N, j*N:(j+1)*N] = -F[i, j] * np.eye(N)
    
    WN = sla.expm(-tf*big_mat) @ WN.flatten("F")
    WN = WN.reshape((N, M), order="F")
    return WN @ R.T
