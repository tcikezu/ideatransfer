import util
from transfer import transfer
from community import community
import  matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
from datetime import datetime
##############################################################

# There is an option to set a file-name -- would highly recommend doing that.
date_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

def plotCommunity(c, data, fps=4, fn=date_time):
    [dataX, dataY, dataZ, posX, posY, ideaDist] = data
    
    T = len(dataX)

    # Plot results
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(131,projection='3d')
    ax1.scatter3D(dataX[0], dataY[0], dataZ[0])

    ax1Ticks = np.linspace(-c.domainSize,c.domainSize,5)
    ax1bounds = (-c.domainSize,c.domainSize)
    
    ax2 = fig.add_subplot(132)
    ax2Ticks = np.linspace(-c.getPositionBounds(),c.getPositionBounds(),4)*2  
    ax2bounds = (-2*c.getPositionBounds(),2*c.getPositionBounds())
    ax2.scatter(posX[0],posY[0],s=1000*c.allRadii**2,alpha=0.08)
    ax2.scatter(posX[0][0], posY[0][0],s=1000*c.allRadii[0]**2,c='r',alpha=0.2)
    
    ax3 = fig.add_subplot(133)
    ifrm = 0
    squareLoss = -1*np.sum(((dataX[ifrm]- dataX[ifrm][0])**2 + (dataY[ifrm] - dataY[ifrm][0])**2 + (dataZ[ifrm] - dataZ[ifrm][0])**2))
#     ax3.scatter(0, squareLoss)
#     ax3.scatter(c.domain,ideaDist[0])
    
    #animation function for animation.FuncAnimation
    def update(ifrm,dataX,dataY,dataZ,posX,posY,ideaDist):
        
        #### special stuff for the agent ###
        squareLoss = -1*np.sum(((dataX[ifrm]- dataX[ifrm][0])**2 + (dataY[ifrm] - dataY[ifrm][0])**2 + (dataZ[ifrm] - dataZ[ifrm][0])**2))
        
        ax1.clear()
        ax1.set_xticks(ax2Ticks)
        ax1.set_yticks(ax2Ticks)
        ax1.set_zticks(ax1Ticks)
        ax1.set(xlim=ax2bounds,ylim=ax2bounds,zlim=ax1bounds)
        ax1.scatter3D(posX[ifrm], posY[ifrm], dataZ[ifrm])
        ax1.set_xlabel("frame: %d" % (ifrm))
        
        ax2.clear()
        ax2.set_xticks(ax2Ticks)
        ax2.set_yticks(ax2Ticks)
        ax2.set(xlim = ax2bounds, ylim = ax2bounds)
        ax2.scatter(posX[ifrm],posY[ifrm],s=1000*c.allRadii**2,alpha=0.08)
        ax2.scatter(posX[ifrm][0], posY[ifrm][0],s=1000*c.allRadii[0]**2,color='r',alpha=0.2)
        ax2.set_xlabel("frame: %d" % (ifrm))
        
#         ax3.clear()
#         ax3.scatter(c.domain,neighborDist[ifrm])
#         ax3.scatter(c.domain,ideaDist[ifrm],color='r')
        ax3.scatter([ifrm],squareLoss,color='b')
#         ax3.set_xlabel("member 0, frame: %d" % (ifrm))

    ani = animation.FuncAnimation(fig, update, T, fargs=(dataX,dataY,dataZ,posX,posY,ideaDist),interval = T/fps )
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)

    #plt.show()


