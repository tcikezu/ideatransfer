""" simulate.py
-----------------------
This file creates a community object, defined in test.py, and runs to see how ideas
evolve over some number of iterations."""

import transfer
from community import *
import  matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def simulate(X=community(300,3), gamma = 0.005, T = 80):
    #gamma = 0.005
    #T = 80
    #X = community(300,3)
    x,y,z = [], [], []

    #fn = 'probabilistic merging'
    fn = 'deterministic merging'
    for t in range(T):
        transfer.deterministicMerge(X, gamma)
        #transfer.probabilisticMerge(X,gamma)
        x.append(X.allIdeas[:,0])
        y.append(X.allIdeas[:,1])
        z.append(X.allIdeas[:,2])


    fps = 40 

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #sct, = ax.plot3D([],[],[],"o",markersize=2)
    #sct = ax.scatter3D([],[],[])

    def update(ifrm, xa, ya, za):
        #sct.set_offsets(np.c_[xa[ifrm],ya[ifrm]])
        #sct.set_3d_properties(za[ifrm])
        #sct.set_data(xa[ifrm], ya[ifrm])
        #sct.set_3d_properties(za[ifrm])
        ax.clear()
        plt.autoscale(False)
        ax.set_xticks([-1.0,-0.5,0.0,0.5,1])
        ax.set_yticks([-1.0,-0.5,0.0,0.5,1])
        ax.set_zticks([-1.0,-0.5,0.0,0.5,1])
        ax.scatter3D(xa[ifrm],ya[ifrm],za[ifrm])
        ax.set_xlabel("frame: %d" % (ifrm))
         
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    ani = animation.FuncAnimation(fig, update, T, fargs=(x,y,z),interval = T/fps )


    #ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)

    plt.show()

