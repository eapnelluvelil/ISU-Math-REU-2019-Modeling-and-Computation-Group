import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from spec_diff import spectral_diff
#makes matrices print nice
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def wave( tf, N, dt, q10, q20,  domain=[-1, 1], order=2, plot=True ):
    """
    p0: initial position of wave equation
    pt0: initial velocity of wave equation
    """
    [a, b] = domain
    frames = 15

    Nt = tf // dt #number of time steps
    dt = tf / Nt #reassigning time step

    t_per_frame = tf / frames #amount of time per frame
    Nt_per_frame = (tf / frames) // dt #number of time steps per frame
    dt = (tf / frames) / Nt_per_frame #reassigning time step again
    Nt = Nt_per_frame * frames #reassigning number of time steps

    s, D = spectral_diff( N, a, b )
    #make copies of D to have separate matrices for each direction of wave
    D1 = np.copy(D)
    D2 = -np.copy(D)

    # Impose boundary conditions
    D2[0,:] = D2[:,0] = 0 # Scenario 1
    D1[-1,:] = D1[:,-1] = 0 # Scenario 1

    #Backwords Euler equation
    if order == 1:
        op1 = np.linalg.inv( np.eye(N) + dt*D1 )
        op2 = np.linalg.inv( np.eye(N) + dt*D2 )
    #trapezoidal equation order 2
    elif order == 2:
        op1 = np.linalg.inv(np.eye(N) + dt*D1/2) @ ( np.eye(N) - dt*D1/2 )
        op2 = np.linalg.inv(np.eye(N) + dt*D2/2) @ ( np.eye(N) - dt*D2/2 )
    #trapezoidal equation order 4
    else:
        op1 = np.linalg.inv(np.eye(N) + dt*D1/2 + ((dt*D1)@(dt*D1))/12) \
                        @ ( np.eye(N) - dt*D1/2 + ((dt*D1)@(dt*D1))/12)
        op2 = np.linalg.inv(np.eye(N) + dt*D2/2 + ((dt*D2)@(dt*D2))/12) \
                        @ ( np.eye(N) - dt*D2/2 + ((dt*D2)@(dt*D2))/12)

    t = 0
    #inital conditions
    q1 = q10( s )
    q2 = q20( s )

    #determine w values in terms of q values
    w2 = 0.5*(q1-q2)
    w1 = 0.5*(q1+q2)

    #plt.clf()
    #plt.plot( s, w1 + w2, 'g' )
    #plt.plot( s, w2, 'b' )
    #plt.show()


    for i in range( int(frames) ):
        if plot == True:
            #get q values from w values
            q1 = w1 - w2
            q2 = w2 + w1
            #plot the waves
            plt.figure(1)
            plt.clf()
            plt.plot( s, w1, 'r' )
            plt.plot( s, w2, 'b' )
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
            #does math in between plot times
            w1 = op1 @ w1
            w2 = op2 @ w2
            t += dt
    #finds q in terms of w
    q1 = w1 - w2
    q2 = w2 + w1
    return q1, q2, s

if __name__ == "__main__":
    N = 256 #number of chebyshev points
    tf = 2.0 #amount of time for the plot to run
    domain = [-1,1]
    q10 = lambda x:  np.exp(-25*(x)**2)
    q20 = lambda x:  0#-np.exp(-25*(x)**2)
    #exact = lambda s, t: q0(s - t)

    #for multiple time steps
    dts = 2.0**-np.arange(4, 10)
    #loops through the different orders that we want it to
    for order in [1, 2, 4]:#range(2, 5, 2):
        errs = []
        #clears the plot
        plt.clf()
        for dt in dts:
            #calls the function
            qf1, qf2, s = wave( tf, N, dt, q10, q20, domain=domain, order=order,
                              plot=True )
            #adds the errors
            errs.append( np.linalg.norm( qf1 - (s - tf) + qf2 - (s + tf) ) )


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
