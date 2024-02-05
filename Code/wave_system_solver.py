import numpy as np
import scipy.linalg as sla
import matplotlib
import matplotlib.pyplot as plt

from spec_diff import spectral_diff
#makes matrices print nice
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

"""

"""
def wave_system( tf, N, dt, A, q0s,  domain=[-1, 1], order=2, plot=True ):
    [a, b] = domain
    frames = 15

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step

    t_per_frame = tf / frames #amount of time per frame
    Nt_per_frame = (tf / frames) // dt #number of time steps per frame
    dt = (tf / frames) / Nt_per_frame #reassigning time step again
    Nt = Nt_per_frame * frames #reassigning number of time steps

    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )

    # Lambda that creates array of initial conditions
    init_conds = lambda x: np.array( [q(x) for q in q0s] ).T
    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    s, D0 = spectral_diff( N, a, b )

    # Create list of time-stepping operators
    # that correspond to w_1, w_2, ..., w_m
    # Create operator
    Operators = []
    for i in range(A.shape[0]):
        #the difference matrix scaled by the lambda value
        D = dt*Lam[i]*D0

        # Impose boundary conditions
        if Lam[i] < 0:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
            # For right traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        else:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        #Backwords Euler equation
        if order == 1:
            Operators.append( np.linalg.inv( np.eye(N) + D ) )
        #trapezoidal equation order 2
        elif order == 2:
            Operators.append( np.linalg.inv(np.eye(N) + D/2) @ ( np.eye(N) - D/2 ) )
        #trapezoidal equation order 4
        else:
            Operators.append( np.linalg.inv(np.eye(N) + D/2 + D@D/12) \
                            @ ( np.eye(N) - D/2 + D@D/12 ) )


    t = 0

    # Create array of initial conditions
    Q0 = init_conds(s)
    W0 = Q0 @ np.linalg.inv(R).T
    print( Q0.shape )

    for i in range( int(frames) ):
        if plot == True:
            #plots the solution
            plt.figure(1)
            plt.clf()
            Q0 = W0 @ R.T
            plt.plot( s, Q0 )
            plt.ylim([-1,1])
            plt.title("Function at {:.2f}, time-step {:.5f}, order {}".format(t, dt, order))

            """
            plt.figure(2)
            plt.clf()
            plt.plot( s, s - t, 'r--')
            plt.plot( s, s + t, 'b--' )
            plt.ylim([-3,3])
            plt.title("Solution at {:.2f}, time-step {:.5f}, order {}".format(t, dt, order))
            """
            plt.pause(0.1)

        for j in range( int(Nt_per_frame) ):
            #does math between plot times
            for k in range(A.shape[0]):
                W0[:, k] = Operators[k] @ W0[:, k]
            t += dt

"""
Inputs:
tf
N
dt
A
q0s - Array of initial conditions
domain
order
plot

Outputs:
W0

"""
def radon_advection( tf, N, dt, A, q0s, angle, domain=[-1, 1], order=2, plot=False):
    [a, b] = domain
    frames = 15

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step

    Nt_per_frame = (tf / frames) // dt
    dt = (tf / frames)/ Nt_per_frame

    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )

    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    s, D0 = spectral_diff( N, a, b )

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

    for j in np.arange(int(frames), dtype=int):
        if plot == True:
            plt.figure(6)
            plt.clf()
            # Q0 = (W0 @ R.T)[:, 0]
            Q0 = W0[:, 0]
            plt.plot(s, Q0)
            plt.ylim([-1, 1])
            plt.title(str(angle))
            plt.pause(0.1)
            
        for i in range( int(Nt_per_frame)  ):
            for k in range(A.shape[0]):
                W0[:, k] = Operators[k] @ W0[:, k]
            
            t += dt

    return W0 @ R.T

def radon_advection_exact( tf, N, dt, A, q0s, angle, domain=[-1, 1], plot=False ):
    [a, b] = domain

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step
    
    # Perform eigendecomposition of A
    Lam, R = np.linalg.eig( A )
    # print(Lam)

    # Create Chebyshev points and corresponding spectral
    # differentiation matrix
    s, D0 = spectral_diff( N, a, b )
   
    # Create list of time-stepping operators
    # that correspond to w_1, w_2, ..., w_m
    Operators = []
    for i in range(A.shape[0]):
        #the difference matrix scaled by the lambda value
        D = dt*Lam[i]*D0

        # Impose boundary conditions
        if Lam[i] < 0:
            # For left traveling waves
            D[ 0,: ] = D[ :,0 ] = 0
        else:
            # For right traveling waves
            D[ -1,: ] = D[ :,-1 ] = 0

        Operators.append(sla.expm(-D))

    t = 0

    # Convert to characteristic variables
    W0 = q0s @ np.linalg.inv(R).T
    
    for i in range( int(Nt)  ):
        for k in range(A.shape[0]):
            W0[:, k] = Operators[k] @ W0[:, k]
            
        t += dt
            
    return W0 @ R.T

if __name__ == "__main__":
    N = 256 #number of chebyshev points
    tf = 2.0 #amount of time to plot
    domain = [-1,1]
    #determines our inital conditions
    q10 = lambda x:  np.exp(-25*(x)**2)
    q20 = lambda x:  0.0*x #-np.exp(-25*(x)**2)
    q0s = [q10] + [q20]
    # B = np.random.rand(5, 5)
    # C = B.T @ B
    # w, v = np.linalg.eig(C)
    # max_e = max(w)
    # min_e = min(w)
    # A = C - 0.5*(max_e + min_e) * np.eye(5)
    #determines out inital matrix
    A = np.array([[0, 1], [1, 0]])

    #exact = lambda s, t: q0(s - t)

    #for multiple time steps
    dts = 2.0**-np.arange(4, 10)
    #loops through the orders that we want it to
    for order in [4]:#range(2, 5, 2):
        # errs = []
        # plt.clf()
        for dt in dts:
            #calls the function
            wave_system( tf, N, dt, A, q0s, domain=domain, order=order,
                              plot=True )
            # errs.append( np.linalg.norm( qf1 - (s - tf) + qf2 - (s + tf) ) )


        # A = np.zeros( [len(dts), 2] )
        # A[:,0] = 1
        # A[:,1] = np.log( dts )

        # [logc, k] = np.linalg.solve( A.T @ A, A.T @ np.log( errs ) )
        #print( errs )
        # plt.loglog( dts, errs, label='{}, {:.2f}'.format(order, k) )

    # plt.legend()
    # plt.show()
