"""
radon_transform.py

Contains various forward radon transforms:
    radon              := Discretized, Barycentric Interpolation
    radon_system       := Compute radon transform for system
    radon_system_exact := Comupute exact transform for system
    radon_exact        := No interpolation used in quadrature
    radon_basis        := Assumes basis vector as input
    radon_matrix       := Uses radon_basis to create matrix operator
    radon_operator     := Creates linear operator using radon
"""
import numpy as np
import radon.utilities as util
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

matplotlib.rcParams.update({'font.size': 20, 'font.family': 'sans-serif', 'text.usetex': 'true'})

def radon(f_vec, Ns, Nw, Nq):
    f_vec = f_vec.flatten()
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])

    s = np.linspace(-1, 1, Ns)
    [S, W] = np.meshgrid(s, w)
    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    # Create s-omega meshgrid

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros_like(f_vec)
    f_vec = f_vec.reshape((Nw, Ns))

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    # Go through angles
    ff = 1e-15
    for i in range(Nw):
        # Go through radii
        for j in range(Ns)[1:-1]:
            # Get linear index
            p = i*Ns + j

            # Get current radius and angle
            w_i = w[i]
            s_j = s[j]

            # Compute half the length of the chord perpendicular
            # to the line determined by w_i and s_j
            r = np.sqrt(1 - s_j*s_j)

            # Discretize the chord perpendicular to the line
            # determined by w_i and s_j
            x_quads = s_j*np.cos(w_i) - r*nodes*np.sin(w_i)
            y_quads = s_j*np.sin(w_i) + r*nodes*np.cos(w_i)

            x_quads[0] = 0.4
            y_quads[0] = 0.5
            """
            if True:#np.isclose( x_quads, x_p ).any() and np.isclose( y_quads, y_p ).any():
                #plt.scatter( s[j]*np.cos(w[i]), s[j]*np.sin(w[i]), color='c', marker='*', s=400 )

                plt.text( s_j*np.cos(w_i)+0.06, s_j*np.sin(w_i)-0.07, f"$(s_{j}, \omega)$", fontsize=10 )
                #plt.scatter( S*np.cos(W), S*np.sin(W), color='k', marker='.', s=64 )
                #plt.scatter( S[i,:]*np.cos(W[i,:]), S[i,:]*np.sin(W[i,:]), color='k', marker='.', s=64 )

                #plt.scatter( x_quads, y_quads, color='r', marker='.', s=64 )
                plt.plot( x_quads, y_quads, 'r-' )
                plt.plot( s[j]*np.cos(w[i]), s[j]*np.sin(w[i]), 'c*', markersize=15 )
            #"""

            w_quads = np.arctan2( y_quads, x_quads )
            s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

            s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
            w_quads = np.mod( w_quads, np.pi )

            w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

            s_values = np.empty( (Nq, 4) )
            f_values = np.empty( (Nq, 4, Ns) )
            p_values = np.empty( (Nq, 4) )


            s_values[:,0] = s_values[:,1] = s_values[:,2] = s_values[:,3] = s_quads[:]
            s_values[(w_indexes-1) % Nw != w_indexes-1, 0] *= -1
            s_values[ w_indexes    % Nw != w_indexes  , 1] *= -1
            s_values[(w_indexes+1) % Nw != w_indexes+1, 2] *= -1
            s_values[(w_indexes+2) % Nw != w_indexes+2, 3] *= -1

            """
            for k in range( Nq ):
                plt.scatter( x_quads[k], y_quads[k], color='red', marker='.', s=200)
                i1, i2, i3, i4 = (w_indexes[k]-1)%Nw, (w_indexes[k])%Nw, (w_indexes[k]+1)%Nw, (w_indexes[k]+2)%Nw
                plt.plot( S[i1,:]*np.cos(W[i1,:]), S[i1,:]*np.sin(W[i1,:]), 'c-', linewidth=1)
                plt.plot( S[i2,:]*np.cos(W[i2,:]), S[i2,:]*np.sin(W[i2,:]), 'c-', linewidth=1)
                plt.plot( S[i3,:]*np.cos(W[i3,:]), S[i3,:]*np.sin(W[i3,:]), 'c-', linewidth=1)
                plt.plot( S[i4,:]*np.cos(W[i4,:]), S[i4,:]*np.sin(W[i4,:]), 'c-', linewidth=1)
                plt.plot( S*np.cos(W), S*np.sin(W), 'k.', linewidth=7 )#, color='k', marker='.', s=49 )
                t = np.linspace(w[(w_indexes[k]-1)%Nw], w[(w_indexes[k]+2)%Nw], 100)
                l = np.sqrt( x_quads[k]**2 + y_quads[k]**2 )
                plt.plot( l*np.cos(t), l*np.sin(t), 'r', linewidth=1 )
                plt.plot( s_values[k, 0]*np.cos(w[(w_indexes[k]-1)%Nw]),
                             s_values[k, 0]*np.sin(w[(w_indexes[k]-1)%Nw]), 'c.', markersize=10)#, c='c', marker='.', s=64 )
                plt.plot( s_values[k, 1]*np.cos(w[(w_indexes[k]  )%Nw]),
                             s_values[k, 1]*np.sin(w[(w_indexes[k]  )%Nw]), 'c.', markersize=10)# c='c', marker='.', s=64 )
                plt.plot( s_values[k, 2]*np.cos(w[(w_indexes[k]+1)%Nw]),
                             s_values[k, 2]*np.sin(w[(w_indexes[k]+1)%Nw]), 'c.', markersize=10)# c='c', marker='.', s=64 )
                plt.plot( s_values[k, 3]*np.cos(w[(w_indexes[k]+2)%Nw]),
                             s_values[k, 3]*np.sin(w[(w_indexes[k]+2)%Nw]), 'c.', markersize=10)# c='c', marker='.', s=64 )

                plt.ylim([0, 1])
                plt.xlim([0, 1])
                plt.gca().set_aspect('equal', 'box')
                plt.gca().set_xticks([0, 0.5, 1])
                plt.gca().set_xlabel('$x$', fontsize=15)
                plt.gca().set_yticks([0, 0.5, 1])
                plt.gca().set_ylabel('$y$', fontsize=15)
                plt.gca().axhline(y=0, color='k', lw=1)
                plt.gca().axvline(x=0, color='k', lw=1)
                #plt.text(0.2, 0.03, "$\omega$", fontsize=15)
                plt.tight_layout()
                plt.show()
            #"""

            f_values[:, 0, :] = f_vec[(w_indexes-1) % Nw, :]
            f_values[:, 1, :] = f_vec[(w_indexes  ) % Nw, :]
            f_values[:, 2, :] = f_vec[(w_indexes+1) % Nw, :]
            f_values[:, 3, :] = f_vec[(w_indexes+2) % Nw, :]

            #print( x_quads, y_quads )
            #print( f_values[:, 0, :] )

                #plt.cla()
            theta = w[w_indexes % Nw] + dw/2

            p_values[:, 0] = np.sum(poly_weights * f_values[:,0,:]/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 1] = np.sum(poly_weights * f_values[:,1,:]/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 2] = np.sum(poly_weights * f_values[:,2,:]/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 3] = np.sum(poly_weights * f_values[:,3,:]/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)

            angle_dist = np.mod(w_quads, np.pi) - theta
            angle_dist[ angle_dist > dw ] -= np.pi
            angle_dist[ angle_dist < -dw ] += np.pi

            ptheta = (-1/16      )*(p_values[:,0] -  9*p_values[:,1] -  9*p_values[:,2] + p_values[:,3] ) + \
                     ( 1/24/dw   )*(p_values[:,0] - 27*p_values[:,1] + 27*p_values[:,2] - p_values[:,3] )*( angle_dist ) + \
                     ( 1/4/dw**2 )*(p_values[:,0] -    p_values[:,1] -    p_values[:,2] + p_values[:,3] )*( angle_dist )**2 - \
                     ( 1/6/dw**3 )*(p_values[:,0] -  3*p_values[:,1] +  3*p_values[:,2] - p_values[:,3] )*( angle_dist )**3

            f_hat[p] = r*np.dot( weights, ptheta )
            """
            plt.ylim([-1.1, 1.1])
            plt.xlim([-1.1, 1.1])
            plt.gca().set_aspect('equal', 'box')
            plt.gca().set_xticks([-1, 0, 1])
            plt.gca().set_yticks([-1, 0, 1])
            #plt.text(0.2, 0.03, "$\omega$", fontsize=15)
            plt.axis("off")
            plt.gca().set_xlabel('$x$', fontsize=15)
            plt.gca().set_ylabel('$y$', fontsize=15)
            plt.tight_layout()
            #plt.show()
            #"""
        """
        plt.text( 0.14, 0.02, "$\omega$", fontsize=15 )
        #t = np.linspace(0, 2*np.pi, 200)
        #plt.plot( np.cos(t), np.sin(t), 'k', linewidth=1 )
        plt.plot( 2*S[i,:]*np.cos(W[i,:]), 2*S[i,:]*np.sin(W[i,:]), 'k--', linewidth=1)
        plt.plot( -2*S[i,:]*np.sin(W[i,:]), 2*S[i,:]*np.cos(W[i,:]), 'k--', linewidth=1)
        plt.text( S[i,-1]*np.cos(W[i,-1])+0.1, S[i,-1]*np.sin(W[i,-1])+0.15, "$s$", fontsize=15)
        plt.text( -S[i,-1]*np.sin(W[i,-1])-0.25, S[i,-1]*np.cos(W[i,-1])+0.15, "$z$", fontsize=15)
        plt.text(0.02, 1, "$y$", fontsize=15)
        plt.text(1, 0.02, "$x$", fontsize=15)
        plt.gca().axhline(y=0, color='k', lw=1)
        plt.gca().axvline(x=0, color='k', lw=1)
        plt.show()
        """
    return f_hat


def radon_system( f_vecs, Ns, Nw, Nq ):
    m = len( f_vecs )
    f_hats = [np.zeros((Nw, Ns)) for i in range(m)]

    for p in range( m ):
        f_hats[p] = radon( f_vecs[p], Ns, Nw, Nq ).reshape((Nw, Ns))

    return f_hats


def radon_system_exact( fs, Ns, Nw, Nq ):
    m = len( fs )
    f_hats = [np.zeros((Nw, Ns)) for i in range(m)]

    for p in range( m ):
        f_hats[p] = radon_exact( fs[p], Ns, Nw, Nq ).reshape((Nw, Ns))

    return f_hats


def radon_exact( f, Ns, Nw, Nd ):
    """
    Returns radon transform of f at points in [0,pi]x[-1,1]
    """
    # Get Chebyshev points for discretization along each angle
    s = util.chebyshev( Ns )

    # Get Chebyshev points and weights for quadrature
    nodes, weights = util.clen_curt( Nd )
    w = np.linspace(0, np.pi, Nw+1)[:-1]

    # Create meshgrid of angles and radii (for plotting purposes, mostly)
    [S, W] = np.meshgrid( s, w )

    # Somewhere to put the values for the radon transform
    f_hat = np.zeros_like( S )

    # Loop over all the quadrature points
    r = np.sqrt( 1 - S*S )
    for i in range(Nd):
        f_hat += r*weights[i]*f( S*np.cos(W) - r*nodes[i]*np.sin(W),
                                 S*np.sin(W) + r*nodes[i]*np.cos(W) )

    return f_hat.flatten()

def radon_basis(fw, fs, Ns, Nw, Nq):
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w = np.linspace(0, np.pi, Nw + 1)[:-1]
    dw = np.absolute(w[1]-w[0])
    Np = Ns*Nw
    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    ff = 1e-15

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w)

    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    ws = w[np.repeat(np.arange(0, Nw, dtype=int), Ns)].reshape((-1, 1))
    ss = s[np.tile(np.arange(0, Ns, dtype=int), Nw)].reshape((-1, 1))

    rs = np.sqrt( 1 - ss*ss ).reshape((-1, 1))

    x_quads = ss*np.cos(ws) - rs*nodes.reshape((1, -1))*np.sin(ws)
    y_quads = ss*np.sin(ws) + rs*nodes.reshape((1, -1))*np.cos(ws)

    x_quads = x_quads.reshape((-1, ))
    y_quads = y_quads.reshape((-1, ))

    w_quads = np.arctan2( y_quads, x_quads )
    s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

    s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
    w_quads = np.mod( w_quads, 2*np.pi)

    w_indexes = np.array( np.floor( w_quads/dw ), dtype=int )

    f_interp = np.zeros( (Nq*Np, ) )

    nz = ( 1 + fw - (w_indexes % Nw) ) % Nw - 1

    s_quads = np.abs( s_quads )

    s_quads[((fw + Nw    ) % (2*Nw) == w_indexes) |
            ((fw + Nw - 1) % (2*Nw) == w_indexes) |
            ((fw + Nw - 2) % (2*Nw) == w_indexes) |
            ((fw + Nw + 1) % (2*Nw) == w_indexes)] *= -1

    #theta = w[w_indexes % Nw] + dw/2

    f_interp = poly_weights[fs] / ( s_quads - s[fs] + ff ) / \
               np.sum(poly_weights / ( s_quads.reshape((-1,1)) - s.reshape((1,-1)) + ff ), axis=1)

    dist = np.mod(w_quads, np.pi) - (w[w_indexes % Nw] + dw/2)
    dist[ dist > dw ] -= np.pi
    dist[ dist < -dw ] += np.pi

    f_interp[nz == -1] *= (-1/16) + (1/24/dw)*( dist[nz==-1] ) + \
                                ( 1/4/dw**2 )*( dist[nz==-1] )**2 + \
                                (-1/6/dw**3 )*( dist[nz==-1] )**3

    f_interp[nz ==  0] *= ( 9/16) - (9/8/dw)*( dist[nz==0] ) + \
                               (-1/4/dw**2 )*( dist[nz==0] )**2 + \
                               ( 1/2/dw**3 )*( dist[nz==0] )**3

    f_interp[nz ==  1] *= ( 9/16) + (9/8/dw)*( dist[nz==1] ) + \
                               (-1/4/dw**2 )*( dist[nz==1] )**2 + \
                               (-1/2/dw**3 )*( dist[nz==1] )**3

    f_interp[nz ==  2] *= (-1/16) - (1/24/dw)*( dist[nz==2] ) + \
                                ( 1/4/dw**2 )*( dist[nz==2] )**2 + \
                                 (1/6/dw**3 )*( dist[nz==2] )**3

    f_interp[(nz < -1) | (nz > 2)] = 0

    f_interp = f_interp.reshape((Np, Nq))

    f_hat = rs.reshape((1, -1))*np.sum( weights.reshape((1, -1))*f_interp, axis=1 )

    return f_hat.flatten()


def radon_matrix( Ns, Nw, Nq, pad=1 ):
    Np = Ns*Nw
    R = np.zeros( (Np, Np) )
    f_vec = np.zeros((Np,))

    for i in range( Nw ):
        for j in range( Ns ):
            p = i*Ns + j
            f_vec[p] = 1
            R[:, p] = radon_basis( i, j, Ns, Nw, Nq ).reshape((-1, ))
            f_vec[p] = 0
            if j <= pad - 1 or j == Ns-pad:
                R[p,p] = 1.0
        #print("Column", p, '/', Np)
    return R

def radial(f_vec, Ns, Nq):
    # Get radii discretization
    s = util.chebyshev( Ns ).reshape((Ns, 1))

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros_like(f_vec)
    f_vec = f_vec.reshape((1, 1, Ns))
    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]
    poly_weights = poly_weights.reshape((1, 1, Ns))

    rs = np.sqrt( 1 - s*s )

    y_quads = rs*nodes.reshape((1, Nq))
    x_quads = s*np.ones_like( y_quads )

    s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

    ff = 1e-15

    f_values = f_vec * np.ones((Ns, Nq, 1))

    if Ns % 2 == 0:
        f_values[ :Ns//2, :, :Ns//2 ] = f_vec[:, :, :Ns//2]
        f_values[ :Ns//2, :, Ns//2: ] = np.flip(f_vec[:, :, :Ns//2])

        f_values[ Ns//2:, :, :Ns//2 ] = np.flip(f_vec[:, :, Ns//2:])
        f_values[ Ns//2:, :, Ns//2: ] = f_vec[:, :, Ns//2:]
    else:
        f_values[ :Ns//2, :, :Ns//2 ] = f_vec[:, :, :Ns//2]
        f_values[ :Ns//2, :, Ns//2+1: ] = np.flip(f_vec[:, :, :Ns//2])

        f_values[ Ns//2+1:, :, :Ns//2 ] = np.flip(f_vec[:, :, Ns//2+1:])
        f_values[ Ns//2+1:, :, Ns//2+1: ] = f_vec[:, :, Ns//2+1:]

    denom = s_quads.reshape((Ns, Nq, 1)) - s.reshape((1, 1, Ns)) + ff
    p_values = np.sum(poly_weights * f_values / denom, axis=2) / \
               np.sum(poly_weights / denom, axis=2)

    f_hat = rs*np.sum( weights.reshape((1, Nq)) * p_values, axis=1 ).reshape((Ns, 1))

    return f_hat.flatten()


def radial_matrix( Ns, Nq ):
    R = np.zeros( (Ns, Ns) )
    f_vec = np.zeros((Ns, ))

    for j in range( Ns ):
        f_vec[j] = 1
        R[:, j] = radial( f_vec, Ns, Nq ).reshape((-1, ))
        f_vec[j] = 0
        if j == 0 or j == Ns - 1:
            R[j,j] = 1.0
        print("Column", j, '/', Ns)
    return R


def fat_radon_basis(fw, fs, Ns, Nw, Nq, skip=2):
    Nwi = skip*Nw
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w_int = np.linspace(0, np.pi, Nwi + 1)[:-1]
    w_ret = np.linspace(0, np.pi, Nw + 1)[:-1]

    dwi = np.absolute(w_int[1]-w_int[0])
    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    ff = 1e-15

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w_int)

    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    ws = w_ret[np.repeat(np.arange(0, Nw, dtype=int), Ns)].reshape((-1, 1))
    ss = s[np.tile(np.arange(0, Ns, dtype=int), Nw)].reshape((-1, 1))

    rs = np.sqrt( 1 - ss*ss ).reshape((-1, 1))

    x_quads = ss*np.cos(ws) - rs*nodes.reshape((1, -1))*np.sin(ws)
    y_quads = ss*np.sin(ws) + rs*nodes.reshape((1, -1))*np.cos(ws)

    x_quads = x_quads.reshape((-1, ))
    y_quads = y_quads.reshape((-1, ))

    w_quads = np.arctan2( y_quads, x_quads )
    s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

    s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
    w_quads = np.mod( w_quads, 2*np.pi )

    w_indexes = np.array( np.floor( w_quads/dwi ), dtype=int )

    f_interp = np.zeros( (Nw*Ns*Nq, ) )

    nz = ( 1 + fw - (w_indexes % Nwi) ) % Nwi - 1

    s_quads = np.abs( s_quads )

    s_quads[((fw + Nwi    ) % (2*Nwi) == w_indexes) |
            ((fw + Nwi - 1) % (2*Nwi) == w_indexes) |
            ((fw + Nwi - 2) % (2*Nwi) == w_indexes) |
            ((fw + Nwi + 1) % (2*Nwi) == w_indexes)] *= -1

    #theta = w[w_indexes % Nw] + dw/2

    f_interp = poly_weights[fs] / ( s_quads - s[fs] + ff ) / \
               np.sum(poly_weights / ( s_quads.reshape((-1,1)) - s.reshape((1,-1)) + ff ), axis=1)

    dist = np.mod(w_quads, np.pi) - (w_int[w_indexes % Nwi] + dwi/2)
    dist[ dist > dwi ] -= np.pi
    dist[ dist < -dwi ] += np.pi

    f_interp[nz == -1] *= (-1/16) + (1/24/dwi)*( dist[nz==-1] ) + \
                                ( 1/4/dwi**2 )*( dist[nz==-1] )**2 + \
                                (-1/6/dwi**3 )*( dist[nz==-1] )**3

    f_interp[nz ==  0] *= ( 9/16) - (9/8/dwi)*( dist[nz==0] ) + \
                               (-1/4/dwi**2 )*( dist[nz==0] )**2 + \
                               ( 1/2/dwi**3 )*( dist[nz==0] )**3

    f_interp[nz ==  1] *= ( 9/16) + (9/8/dwi)*( dist[nz==1] ) + \
                               (-1/4/dwi**2 )*( dist[nz==1] )**2 + \
                               (-1/2/dwi**3 )*( dist[nz==1] )**3

    f_interp[nz ==  2] *= (-1/16) - (1/24/dwi)*( dist[nz==2] ) + \
                                ( 1/4/dwi**2 )*( dist[nz==2] )**2 + \
                                 (1/6/dwi**3 )*( dist[nz==2] )**3

    f_interp[(nz < -1) | (nz > 2)] = 0

    f_interp = f_interp.reshape((Ns*Nw, Nq))

    f_hat = rs.reshape((1, -1))*np.sum( weights.reshape((1, -1))*f_interp, axis=1 )

    return f_hat.flatten()


def fat_radon_matrix( Ns, Nw, Nq, skip=2 ):
    R = np.zeros( (Ns*Nw, Ns*skip*Nw) )
    f_vec = np.zeros((Ns*skip*Nw,))

    p = -1
    for i in range( skip*Nw ):
        for j in range( Ns ):
            p += 1
            f_vec[p] = 1
            R[:, p] = fat_radon_basis( i, j, Ns, Nw, Nq ).reshape((-1, ))
            #if np.allclose( R[:, p], 0 ):
            #    print( i, j, p )
            f_vec[p] = 0
            if ( j == 0 or j == Ns-1 ) and i % skip == 0:
                R[i//skip*Ns+j, p] = 1.0
        #print("Column", p)
    return R


def rect_radon_basis(fw, fs, Ns, Nw, Nq, ret=1, inp=1):
    Nwr = ret*Nw
    Nwi = inp*Nw
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w_int = np.linspace(0, np.pi, Nwi + 1)[:-1]
    w_ret = np.linspace(0, np.pi, Nwr + 1)[:-1]

    dwi = np.absolute(w_int[1]-w_int[0])
    dwr = np.absolute(w_ret[1]-w_ret[0])
    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    ff = 1e-15

    # Create s-omega meshgrid
    #[S, W] = np.meshgrid(s, w)

    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    ws = w_ret[np.repeat(np.arange(0, Nwr, dtype=int), Ns)].reshape((-1, 1))
    ss = s[np.tile(np.arange(0, Ns, dtype=int), Nwr)].reshape((-1, 1))

    rs = np.sqrt( 1 - ss*ss ).reshape((-1, 1))

    x_quads = ss*np.cos(ws) - rs*nodes.reshape((1, -1))*np.sin(ws)
    y_quads = ss*np.sin(ws) + rs*nodes.reshape((1, -1))*np.cos(ws)

    x_quads = x_quads.reshape((-1, ))
    y_quads = y_quads.reshape((-1, ))

    w_quads = np.arctan2( y_quads, x_quads )
    s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

    s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
    w_quads = np.mod( w_quads, 2*np.pi )

    w_indexes = np.array( np.floor( w_quads/dwi ), dtype=int )

    f_interp = np.zeros( (Nwr*Ns*Nq, ) )

    nz = ( 1 + fw - (w_indexes % Nwi) ) % Nwi - 1

    s_quads = np.abs( s_quads )

    s_quads[((fw + Nwi    ) % (2*Nwi) == w_indexes) |
            ((fw + Nwi - 1) % (2*Nwi) == w_indexes) |
            ((fw + Nwi - 2) % (2*Nwi) == w_indexes) |
            ((fw + Nwi + 1) % (2*Nwi) == w_indexes)] *= -1

    #theta = w[w_indexes % Nw] + dw/2

    f_interp = poly_weights[fs] / ( s_quads - s[fs] + ff ) / \
               np.sum(poly_weights / ( s_quads.reshape((-1,1)) - s.reshape((1,-1)) + ff ), axis=1)

    dist = np.mod(w_quads, np.pi) - (w_int[w_indexes % Nwi] + dwi/2)
    dist[ dist > dwi ] -= np.pi
    dist[ dist < -dwi ] += np.pi

    f_interp[nz == -1] *= (-1/16) + (1/24/dwi)*( dist[nz==-1] ) + \
                                ( 1/4/dwi**2 )*( dist[nz==-1] )**2 + \
                                (-1/6/dwi**3 )*( dist[nz==-1] )**3

    f_interp[nz ==  0] *= ( 9/16) - (9/8/dwi)*( dist[nz==0] ) + \
                               (-1/4/dwi**2 )*( dist[nz==0] )**2 + \
                               ( 1/2/dwi**3 )*( dist[nz==0] )**3

    f_interp[nz ==  1] *= ( 9/16) + (9/8/dwi)*( dist[nz==1] ) + \
                               (-1/4/dwi**2 )*( dist[nz==1] )**2 + \
                               (-1/2/dwi**3 )*( dist[nz==1] )**3

    f_interp[nz ==  2] *= (-1/16) - (1/24/dwi)*( dist[nz==2] ) + \
                                ( 1/4/dwi**2 )*( dist[nz==2] )**2 + \
                                 (1/6/dwi**3 )*( dist[nz==2] )**3

    f_interp[(nz < -1) | (nz > 2)] = 0

    f_interp = f_interp.reshape((Ns*Nwr, Nq))

    f_hat = rs.reshape((1, -1))*np.sum( weights.reshape((1, -1))*f_interp, axis=1 )

    return f_hat.flatten()


def rect_radon_matrix( Ns, Nw, Nq, ret=1, inp=1 ):
    R = np.zeros( (Ns*ret*Nw, Ns*inp*Nw) )
    f_vec = np.zeros((Ns*inp*Nw,))

    p = -1
    for i in range( inp*Nw ):
        for j in range( Ns ):
            p += 1
            f_vec[p] = 1
            R[:, p] = rect_radon_basis( i, j, Ns, Nw, Nq, ret, inp ).reshape((-1, ))
            #if np.allclose( R[:, p], 0 ):
            #    print( i, j, p )
            f_vec[p] = 0
            if ( j == 0 or j == Ns-1 ):# and (i % inp == 0):
            #    for k in range(ret):
                 R[i//inp*ret*Ns+j, i*Ns+j] = 1.0
        print("Column", p, '/', Ns*inp*Nw)
    return R

def backprojection(f_hat, Ns, Nw, Nq, ret=1, inp=1):
    Nwi = Nw*inp
    Nwr = Nw*ret

    f_hat = f_hat.flatten()
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w_int = np.linspace(0, np.pi, Nwi + 1)[:-1]
    w_ret = np.linspace(0, np.pi, Nwr + 1)[:-1]

    dwi = np.absolute(w_int[1]-w_int[0])
    dwr = np.absolute(w_ret[1]-w_ret[0])

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    #nodes, weights = util.clen_curt(Nq)
    node_angles = np.linspace(0, np.pi, Nq+1)[:-1]
    w_quads = node_angles

    # Create s-omega meshgrid
    [S, W] = np.meshgrid(s, w_int)

    # Vector to store approximate Radon transform
    # of f
    f_vec = np.zeros((Nwr, Ns)).flatten()
    f_hat = f_hat.reshape((Nwi, Ns))

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    # Go through angles
    ff = 1e-15
    for i in range(Nwr):
        # Go through radii
        for j in range(Ns):
            # Get linear index
            p = i*Ns + j

            # Get current radius and angle
            w_i = w_ret[i]
            s_j = s[j]

            # Compute half the length of the chord perpendicular
            # to the line determined by w_i and s_j
            #r = np.sqrt(1 - s_j*s_j)

            # Discretize the chord perpendicular to the line
            # determined by w_i and s_j
            #x_quads = s_j*np.cos(w_i) - r*nodes*np.sin(w_i)
            #y_quads = s_j*np.sin(w_i) + r*nodes*np.cos(w_i)

            #w_quads = np.arctan2( y_quads, x_quads )
            #s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )
            s_quads = s_j*np.cos(w_i)*np.cos(node_angles) + s_j*np.sin(w_i)*np.sin(node_angles)

            """
            if True:#np.isclose( x_quads, x_p ).any() and np.isclose( y_quads, y_p ).any():
                #plt.scatter( s[j]*np.cos(w[i]), s[j]*np.sin(w[i]), color='c', marker='*', s=400 )

                t = np.linspace(0, 2*np.pi, 200)

                w_quads2 = np.linspace(0, np.pi, 200)
                s_quads2 = s_j*np.cos(w_i)*np.cos(w_quads2) + s_j*np.sin(w_i)*np.sin(w_quads2)

                plt.plot( np.cos(t), np.sin(t), 'k', linewidth=1 )
                plt.text( s_j*np.cos(w_i)+0.09, s_j*np.sin(w_i)-0.09, "$(s, \omega)$", fontsize=20 )

                plt.scatter( S*np.cos(W), S*np.sin(W), color='k', marker='.', s=64 )
                plt.scatter( S[i,:]*np.cos(W[i,:]), S[i,:]*np.sin(W[i,:]), color='k', marker='.', s=64 )

                plt.plot( s_quads2*np.cos(w_quads2), s_quads2*np.sin(w_quads2), 'r' )
                plt.scatter( s_quads*np.cos(w_quads), s_quads*np.sin(w_quads), color='r', marker='.', s=64 )
            """

            s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
            w_quads = np.mod( w_quads, np.pi )

            w_indexes = np.array( np.floor( w_quads/dwi ), dtype=int )

            s_values = np.empty( (Nq, 4) )
            f_values = np.empty( (Nq, 4, Ns) )
            p_values = np.empty( (Nq, 4) )


            s_values[:,0] = s_values[:,1] = s_values[:,2] = s_values[:,3] = s_quads[:]
            s_values[(w_indexes-1) % Nwi != w_indexes-1, 0] *= -1
            s_values[ w_indexes    % Nwi != w_indexes  , 1] *= -1
            s_values[(w_indexes+1) % Nwi != w_indexes+1, 2] *= -1
            s_values[(w_indexes+2) % Nwi != w_indexes+2, 3] *= -1

            f_values[:, 0, :] = f_hat[(w_indexes-1) % Nwi, :]
            f_values[:, 1, :] = f_hat[(w_indexes  ) % Nwi, :]
            f_values[:, 2, :] = f_hat[(w_indexes+1) % Nwi, :]
            f_values[:, 3, :] = f_hat[(w_indexes+2) % Nwi, :]

            #print( x_quads, y_quads )
            #print( f_values[:, 0, :] )

                #plt.cla()
            theta = w_int[w_indexes % Nwi] + dwi/2

            p_values[:, 0] = np.sum(poly_weights * f_values[:,0,:]/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 1] = np.sum(poly_weights * f_values[:,1,:]/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 2] = np.sum(poly_weights * f_values[:,2,:]/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 3] = np.sum(poly_weights * f_values[:,3,:]/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)

            angle_dist = np.mod(w_quads, np.pi) - theta
            angle_dist[ angle_dist > dwi ] -= np.pi
            angle_dist[ angle_dist < -dwi ] += np.pi

            ptheta = (-1/16       )*(p_values[:,0] -  9*p_values[:,1] -  9*p_values[:,2] + p_values[:,3] ) + \
                     ( 1/24/dwi   )*(p_values[:,0] - 27*p_values[:,1] + 27*p_values[:,2] - p_values[:,3] )*( angle_dist ) + \
                     ( 1/4/dwi**2 )*(p_values[:,0] -    p_values[:,1] -    p_values[:,2] + p_values[:,3] )*( angle_dist )**2 - \
                     ( 1/6/dwi**3 )*(p_values[:,0] -  3*p_values[:,1] +  3*p_values[:,2] - p_values[:,3] )*( angle_dist )**3

            f_vec[p] = np.sum( ptheta ) / Nq
            """
            plt.gca().set_aspect('equal', 'box')
            plt.gca().set_xticks([-1, 0, 1])
            plt.gca().set_xlabel('$x$', fontsize=15)
            plt.gca().set_yticks([-1, 0, 1])
            plt.gca().set_ylabel('$y$', fontsize=15)
            plt.gca().axhline(y=0, color='k', lw=1)
            plt.gca().axvline(x=0, color='k', lw=1)
            #plt.text(0.2, 0.03, "$\omega$", fontsize=15)
            plt.tight_layout()
            plt.show()
            #"""

    return f_vec


def backprojection_matrix( Ns, Nw, Nq, ret=1, inp=1 ):
    R = np.zeros( (Ns*ret*Nw, Ns*inp*Nw) )
    f_hat = np.zeros((Ns*inp*Nw,))

    p = -1
    for i in range( inp*Nw ):
        for j in range( Ns ):
            p += 1
            f_hat[p] = 1
            R[:, p] = backprojection( f_hat, Ns, Nw, Nq, ret, inp ).reshape((-1, ))
            #if np.allclose( R[:, p], 0 ):
            #    print( i, j, p )
            f_hat[p] = 0
            if ( j == 0 or j == Ns-1 ):# and (i % inp == 0):
            #    for k in range(ret):
                 R[i//inp*ret*Ns+j, i*Ns+j] = 1.0
        print("Column", p, '/', Ns*inp*Nw)
    return R

def rect_radon(f_vec, Ns, Nw, Nq, ret=1, inp=1):
    Nwi = inp*Nw
    Nwr = ret*Nw

    f_vec = f_vec.flatten()
    # Get radii discretization
    s = util.chebyshev( Ns )

    # Create angle (omega) discretization
    w_int = np.linspace(0, np.pi, Nwi + 1)[:-1]
    w_ret = np.linspace(0, np.pi, Nwr + 1)[:-1]
    dwi = np.absolute(w_int[1]-w_int[0])
    dwr = np.absolute(w_ret[1]-w_ret[0])

    # Get Chebyshev nodes and weights for quadrature
    # along each line that goes through the origin
    nodes, weights = util.clen_curt(Nq)

    # Create s-omega meshgrid
    #[S, W] = np.meshgrid(s, w)

    # Vector to store approximate Radon transform
    # of f
    f_hat = np.zeros((Nwr, Ns)).flatten()
    f_vec = f_vec.reshape((Nwi, Ns))

    # Compute weights for barycentric interpolation
    # along a line that passes through the origin
    poly_weights = (-1.0)**np.arange(Ns)
    poly_weights[0] = (0.5)
    poly_weights[-1] = (0.5)*poly_weights[-1]

    # Go through angles
    ff = 1e-15
    for i in range(Nwr):
        # Go through radii
        for j in range(Ns)[1:-1]:
            # Get linear index
            p = i*Ns + j

            # Get current radius and angle
            w_i = w_ret[i]
            s_j = s[j]

            # Compute half the length of the chord perpendicular
            # to the line determined by w_i and s_j
            r = np.sqrt(1 - s_j*s_j)

            # Discretize the chord perpendicular to the line
            # determined by w_i and s_j
            x_quads = s_j*np.cos(w_i) - r*nodes*np.sin(w_i)
            y_quads = s_j*np.sin(w_i) + r*nodes*np.cos(w_i)

            w_quads = np.arctan2( y_quads, x_quads )
            s_quads = np.sqrt( x_quads*x_quads + y_quads*y_quads )

            s_quads[ ~((0 <= w_quads) & (w_quads < np.pi)) ] *= -1
            w_quads = np.mod( w_quads, np.pi )

            w_indexes = np.array( np.floor( w_quads/dwi ), dtype=int )

            s_values = np.empty( (Nq, 4) )
            f_values = np.empty( (Nq, 4, Ns) )
            p_values = np.empty( (Nq, 4) )


            s_values[:,0] = s_values[:,1] = s_values[:,2] = s_values[:,3] = s_quads[:]
            s_values[(w_indexes-1) % Nwi != w_indexes-1, 0] *= -1
            s_values[ w_indexes    % Nwi != w_indexes  , 1] *= -1
            s_values[(w_indexes+1) % Nwi != w_indexes+1, 2] *= -1
            s_values[(w_indexes+2) % Nwi != w_indexes+2, 3] *= -1

            f_values[:, 0, :] = f_vec[(w_indexes-1) % Nwi, :]
            f_values[:, 1, :] = f_vec[(w_indexes  ) % Nwi, :]
            f_values[:, 2, :] = f_vec[(w_indexes+1) % Nwi, :]
            f_values[:, 3, :] = f_vec[(w_indexes+2) % Nwi, :]

            #print( x_quads, y_quads )
            #print( f_values[:, 0, :] )

                #plt.cla()
            theta = w_int[w_indexes % Nwi] + dwi/2

            p_values[:, 0] = np.sum(poly_weights * f_values[:,0,:]/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,0].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 1] = np.sum(poly_weights * f_values[:,1,:]/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,1].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 2] = np.sum(poly_weights * f_values[:,2,:]/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,2].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)
            p_values[:, 3] = np.sum(poly_weights * f_values[:,3,:]/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1) / \
                             np.sum(poly_weights * 1/(s_values[:,3].reshape((-1,1)) - s.reshape((1,-1)) + ff), axis=1)

            angle_dist = np.mod(w_quads, np.pi) - theta
            angle_dist[ angle_dist > dwi ] -= np.pi
            angle_dist[ angle_dist < -dwi ] += np.pi

            ptheta = (-1/16      )*(p_values[:,0] -  9*p_values[:,1] -  9*p_values[:,2] + p_values[:,3] ) + \
                     ( 1/24/dwi   )*(p_values[:,0] - 27*p_values[:,1] + 27*p_values[:,2] - p_values[:,3] )*( angle_dist ) + \
                     ( 1/4/dwi**2 )*(p_values[:,0] -    p_values[:,1] -    p_values[:,2] + p_values[:,3] )*( angle_dist )**2 - \
                     ( 1/6/dwi**3 )*(p_values[:,0] -  3*p_values[:,1] +  3*p_values[:,2] - p_values[:,3] )*( angle_dist )**3

            f_hat[p] = r*np.dot( weights, ptheta )

    #plt.show()
    return f_hat

def radon_operator(Ns, Nw, Nq):
    def rad_wrapper( f_vec ):
        return radon( f_vec, Ns, Nw, Nq )
    return LinearOperator((Ns*Nw, Ns*Nw), matvec=rad_wrapper)

def backprojection_operator( Ns, Nw, Nq, ret=1, inp=1 ):
    def rad_wrapper( f_vec ):
        return backprojection( rect_radon( f_vec, Ns, Nw, Nq, ret=ret ), Ns, Nw, Nq, inp=ret )
    return LinearOperator((Ns*Nw*inp, Ns*Nw*inp), matvec=rad_wrapper)

"""
Computes the approximate pseudoinverse of R

rcond is a float, rcond >= 0, such that all singular
values less than rcond * |sing_max|, where sing_max
is the largest singular value of R, are set to 0
"""
def radon_inverse(R, f_hat, rcond=1e-2):
    return np.linalg.pinv(R, rcond).dot(f_hat)

def kaczmarz(A, b, max_iter=1, order=2):
    m = A.shape[0]
    n = A.shape[1]

    x_init = np.zeros((n, max_iter))
    unit_vec = np.zeros((m, ))
    rel_errs = np.zeros((max_iter, ))

    for it in np.arange(max_iter, dtype=int):
        if it > 0:
            x_init[:, it] = np.copy(x_init[:, it - 1])
        for i in np.arange(m, dtype=int):
            print("Iter {:d}, row = {:d}/{:d}".format(it, i, m))

            unit_vec[i] = 1.0

            x_init[:, it] -= A[i, :] * (np.dot(A @ x_init[:, it] - b, unit_vec) / np.dot(A[i, :], A[i, :]))

            unit_vec[i] = 0

        rel_errs[it] = np.linalg.norm(A @ x_init[:, it] - b, ord=order) / \
                         np.linalg.norm(b, ord=order)

    return x_init, rel_errs

