import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmaps
import radon.utilities as ru

matplotlib.rcParams.update({'font.size': 20, 'font.family': 'sans-serif', 'text.usetex': 'true'})

wlabels = [r'$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$']
wvalues = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])

latexify = lambda s: '$' + str(s) + '$'

def plot_physical_space(f_vec, title, filename, scale=1, plot_slice=False):
    cvalues = scale*np.array([-1, -0.5, 0, 0.5, 1])
    clabels = list(map( latexify, cvalues.tolist() ) )

    Nw, Ns = f_vec.shape

    w = np.linspace(0, np.pi, Nw+1)
    s = scale*ru.chebyshev( Ns )
    [S, W] = np.meshgrid( s, w )
    X, Y = S*np.cos(W), S*np.sin(W)

    f_vec_long = np.zeros((Nw + 1, Ns))
    f_vec_long[:-1,:] = f_vec[:,:]
    f_vec_long[ -1,:] = np.flip(f_vec[0,:])

    plt.figure(10)
    plt.clf()
    plt.contourf( X, Y, f_vec_long, cmap=cmaps.jet, levels=50 )
    plt.contourf( X, Y, f_vec_long, cmap=cmaps.jet, levels=50 )
    #plt.plot([-0.411, 0.911], [0.911, -0.411], 'g')
    #plt.plot([-0.911, 0.411], [0.411, -0.911], 'y')
    #plt.plot([-0.97, 0.97], [0.25, 0.25], 'm' )

    plt.title( title )
    plt.gca().set_aspect('equal', 'box')
    plt.colorbar()

    plt.gca().set_ylabel( "$y$" )
    plt.gca().set_yticks( cvalues )
    plt.gca().set_yticklabels( clabels )

    plt.gca().set_xlabel( "$x$" )
    plt.gca().set_xticks( cvalues )
    plt.gca().set_xticklabels( clabels )

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

    if plot_slice == True:
        plt.plot( s, f_vec[0,:] )
        plt.gca().set_ylabel( "$f$" )
        plt.gca().set_xlabel( "$x$" )
        plt.gca().set_xticks( cvalues )
        plt.gca().set_xticklabels( clabels )
        plt.title( title + " Horizontal Slice" )
        plt.tight_layout()
        plt.savefig( filename.replace( '.', '_slice.' ) )
        plt.clf()

def plot_radon_space(f_hat, title, filename, scale=1, plot_slice=False):
    cvalues = scale*np.array([-1, -0.5, 0, 0.5, 1])
    clabels = list(map( latexify, cvalues.tolist() ) )

    Nw, Ns = f_hat.shape

    w = np.linspace(0, np.pi, Nw)
    s = scale*ru.chebyshev( Ns )
    [S, W] = np.meshgrid( s, w )

    plt.figure(10)
    plt.clf()
    plt.contourf( S, W, f_hat, cmap=cmaps.plasma, levels=50 )
    plt.contourf( S, W, f_hat, cmap=cmaps.plasma, levels=50 )
    plt.colorbar()
    #plt.scatter( [0.3535533], [np.pi/4], c='g', marker='*', s=100)
    #plt.scatter( [-0.3535533], [np.pi/4], c='y', marker='*', s=100)
    #plt.scatter( [0.25], [np.pi/2], c='m', marker='*', s=100)
    plt.title( title )

    plt.gca().set_ylabel( "$\omega$" )
    plt.gca().set_yticks( wvalues )
    plt.gca().set_yticklabels( wlabels )

    plt.gca().set_xlabel( "$s$" )
    plt.gca().set_xticks( cvalues )
    plt.gca().set_xticklabels( clabels )

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

    if plot_slice == True:
        plt.plot( s, f_hat[0,:] )
        plt.gca().set_ylabel( "$\widehat{f}$" )
        plt.gca().set_xlabel( "$s$" )
        plt.gca().set_xticks( cvalues )
        plt.gca().set_xticklabels( clabels )
        plt.title( title + " Horizontal Slice" )
        plt.tight_layout()
        plt.savefig( filename.replace( '.', '_slice.' ) )
        plt.clf()

