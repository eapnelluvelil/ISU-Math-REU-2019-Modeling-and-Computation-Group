import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from spec_diff import spectral_diff
#makes matrices print nice
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def advection( tf, N, dt, q0, domain=[-1, 1], order=2, plot=True ):
    """
    Note q0 must be a function
    """
    [a, b] = domain
    frames = 10

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step

    t_per_frame = tf / frames #amount of time per frame
    Nt_per_frame = (tf / frames) // dt #number of time steps per frame
    dt = (tf / frames) / Nt_per_frame #reassigning time step again
    Nt = Nt_per_frame * frames #reassigning number of time steps

    s, D = spectral_diff( N, a, b )
    #sets the last row and column to 0
    D[-1,:] = D[:,-1] = 0 # Scenario 1

    # Scenario 2
    #D = D[:-1,:-1]
    #s = s[:-1]
    #N = N - 1

    # Does matrix multiplication for later use, can cut out if wanted
    D2 = np.linalg.matrix_power( dt*D, 2 )
    #D3 = np.linalg.matrix_power( dt*D, 3 )
    #D4 = np.linalg.matrix_power( dt*D, 4 )
    #D5 = np.linalg.matrix_power( dt*D, 5 )
    #D6 = np.linalg.matrix_power( dt*D, 6 )

    """ # Scenario 3
    D[-1,:] = 0
    D[:,-1] = 0

    D2[-1,:] = 0
    D2[:,-1] = 0

    D3[-1,:] = 0
    D3[:,-1] = 0

    D4[-1,:] = 0
    D4[:,-1] = 0

    D5[-1,:] = 0
    D5[:,-1] = 0

    D6[-1,:] = 0
    D6[:,-1] = 0
    """

    #Backwords Euler equation
    if order == 1:
        op = np.linalg.inv( np.eye(N) + dt*D )
    #trapezoidal equation order 2
    elif order == 2:
        op = np.linalg.inv(np.eye(N) + dt*D/2) @ ( np.eye(N) - dt*D/2 )
        #op = np.linalg.inv( np.eye(N) + dt*D + D2/2 )
    #elif order == 3:
    #    op = np.linalg.inv( np.eye(N) + dt*D + D2/2 + D3/6 )
    #trapezoidal equation order 4
    else:
        op = np.linalg.inv(np.eye(N) + dt*D/2 + D2/12) @ ( np.eye(N) - dt*D/2 + D2/12)
        #op = np.linalg.inv( np.eye(N) + dt*D + D2/2 + D3/6 + D4/24 )
    #elif order == 5:
    #    op = np.linalg.inv( np.eye(N) + dt*D + D2/2 + D3/6 + D4/24 + D5/120 )
    #elif order == 6:
    #    op = np.linalg.inv( np.eye(N) + dt*D + D2/2 + D3/6 + D4/24 + D5/120 + D6/720 )
    # print( "dt = {}, order = {}, max_eig = {}".format(dt, order,
    # np.max(np.abs(np.linalg.eigvals(op)))) )

    # Scenario 4
    #op[-1,:] = 0
    #op[:,-1] = 0

    t = 0
    #inital condition
    q = q0( s )
    for i in range( int(frames) ):
        if plot == True:
            #plots the created line with the actual line
            plt.figure(1)
            plt.clf()
            plt.plot( s, q, 'r' )
            plt.plot( s, q0( s - t ), 'b--' )
            plt.ylim([0,1.5])
            plt.title("Function at {}, order {}".format(t, order))

            #plt.figure(2)
            #plt.clf()
            #plt.plot( s, D @ q, 'b' )
            #plt.title("Derivative at {}".format(t))
            plt.pause(0.0001)

        for j in range( int(Nt_per_frame) ):
            #does time steps between plot times
            q = op @ q
            t += dt

    return q, s

if __name__ == "__main__":
    N = 256 #chebyshev points
    tf = 3.0 #amount of time
    domain = [-5,5]
    q0 = lambda x: np.exp(-25*x**2)
    #exact = lambda s, t: q0(s - t)

    #creates our different time steps
    dts = 2.0**-np.arange(4, 11)
    #loops through the orders we want it to
    for order in [1, 2, 4]:
        errs = []
        for dt in dts:
            #calls our function
            qf, s = advection( tf, N, dt, q0, domain=domain, order=order,
                              plot=False )
            #adds errors
            errs.append( np.linalg.norm( qf - q0(s - tf) ) )


        #create matrices for linear regression
        A = np.zeros( [len(dts), 2] ) #size
        #sets the columns to what we need them to be
        A[:,0] = 1
        A[:,1] = np.log( dts )

        #solves the system
        [logc, k] = np.linalg.solve( A.T @ A, A.T @ np.log( errs ) )
        #print( errs )
        #plots the solution
        plt.loglog( dts, errs, label='{}, {:.2f}'.format(order, k) )

    plt.legend()
    plt.show()
