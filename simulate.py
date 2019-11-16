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
def simulate(c=None, gamma = 0.005, T = 80,fn=date_time):
    # seems like the same community is used again and again if I do not specify None case below.
    if c is None:
        c = community(300,3)
    fps = 40

    # Data to store c.allIdeas, to then make animation
    dataX = np.zeros((T*fps,c.numberMembers))
    dataY = np.zeros((T*fps,c.numberMembers))
    dataZ = np.zeros((T*fps,c.numberMembers))

#     posX = np.zeros((T*fps,c.numberMembers))
#     posY = np.zeros((T*fps,c.numberMembers))

    # Iterate the idea transfer throughout community
    for t in range(0,T+1):
        np.random.seed()
        dataX[t,:] = c.allIdeas[:,0]
        dataY[t,:] = c.allIdeas[:,1]
        dataZ[t,:] = c.allIdeas[:,2]

#         posX[t,:] = c.allPositions[:,0]
#         posY[t,:] = c.allPositions[:,1]
        #transfer.deterministicMerge(c, gamma/(t+1)**0.5)
        transfer.probabilisticMerge(c, gamma,t)

    # Plot results
    fig = plt.figure()
#   fig, (ax1, ax2) = plt.subplots(1,2)
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.scatter3D(dataX[0], dataY[0], dataZ[0])
#     ax2.scatter(posX[0],posY[0])

    ax1Ticks = np.linspace(-c.domainSize,c.domainSize,4)
#     ax2Ticks = np.linspace(-2,2,4)

    #animation function for animation.FuncAnimation
    def update(ifrm,dataX,dataY,dataZ):
        ax1.clear()
        plt.autoscale(False)
        ax1.set_xticks(ax1Ticks)
        ax1.set_yticks(ax1Ticks)
        ax1.set_zticks(ax1Ticks)
        ax1.scatter3D(dataX[ifrm], dataY[ifrm], dataZ[ifrm])
        ax1.set_xlabel("frame: %d" % (ifrm))

#         ax2.clear()
#         ax2.set_xticks(ax2Ticks)
#         ax2.sety_ticks(ax2Ticks)
#         ax2.scatter(posX[ifrm],posY[ifrm])
#         ax2.set_xlabel("frame: %d" % (ifrm))

    ani = animation.FuncAnimation(fig, update, T, fargs=(dataX,dataY,dataZ),interval = T/fps )
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)

    plt.show()
