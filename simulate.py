""" simulate.py
-----------------------
This file creates a community object, defined in test.py, and runs to see how ideas
evolve over some number of iterations. Also creates a gif to see an animation of the ideas."""

import transfer
from community import *
import  matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
from datetime import datetime

# There is an option to set a file-name -- would highly recommend doing that.
date_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

""" simulate returns an animated 3d scatter plot of the community allIdeas object.
The input parameters are the community, the interaction coefficient gamma, 
and also fn, a string file-name which defaults to date and time."""
def simulate(X=None, gamma = 0.005, T = 80,fn=date_time):
    # seems like the same community is used again and again if I do not specify None case below.
    if X is None:
        X = community(300,3)
    fps = 40
    
    # Data to store X.allIdeas, to then make animation
    dataX = np.ndarray((T*fps,X.numberMembers))
    dataY = np.ndarray((T*fps,X.numberMembers))
    dataZ = np.ndarray((T*fps,X.numberMembers))

    # Iterate the idea transfer throughout community
    for t in range(T):
        dataX[t,:] = X.allIdeas[:,0]
        dataY[t,:] = X.allIdeas[:,1]
        dataZ[t,:] = X.allIdeas[:,2]
        #transfer.deterministicMerge(X, gamma)
        transfer.probabilisticMerge(X, gamma)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(dataX[0], dataY[0], dataZ[0])

    #animation function for animation.FuncAnimation
    def update(ifrm,dataX,dataY,dataZ):
        ax.clear()
        plt.autoscale(False)
        ax.set_xticks([-1.0,-0.5,0.0,0.5,1])
        ax.set_yticks([-1.0,-0.5,0.0,0.5,1])
        ax.set_zticks([-1.0,-0.5,0.0,0.5,1])
        ax.scatter3D(dataX[ifrm], dataY[ifrm], dataZ[ifrm])
        ax.set_xlabel("frame: %d" % (ifrm))
         
    ani = animation.FuncAnimation(fig, update, T, fargs=(dataX,dataY,dataZ),interval = T/fps )
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)
    
    plt.show()
