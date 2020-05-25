import util
from plotCommunity import plotCommunity
from transfer import transfer
from community import community
import numpy as np
#####################################################################################

""" simulate returns an animated 3d scatter plot of the community allIdeas object.
The input parameters are the community, the interaction coefficient gamma, 
and also fn, a string file-name which defaults to date and time."""
def simulate(c=None, T = 80):    
    if c is None:
        c = community(300,3,10)
    Transfer = transfer(c)

    # Data to store c.allIdeas, to then make animation
    dataX = np.zeros((T,c.numberMembers))
    dataY = np.zeros((T,c.numberMembers))
    dataZ = np.zeros((T,c.numberMembers))
    
    posX = np.zeros((T,c.numberMembers))
    posY = np.zeros((T,c.numberMembers))
    
    ideaDist = np.zeros((T,c.domainSize*2+1))

    # Iterate the idea transfer throughout community
    
    for t in range(0,T):
        np.random.seed()
        dataX[t,:] = c.allIdeas[:,0]
        dataY[t,:] = c.allIdeas[:,1]
        dataZ[t,:] = c.allIdeas[:,2]
        
        posX[t,:] = c.allPositions[:,0]
        posY[t,:] = c.allPositions[:,1]
        
        ideaDist[t,:] = c.ideaDistribution[0][0]
        #Transfer.deterministicMerge()
        Transfer.probabilisticMerge()
    
    data = [dataX, dataY, dataZ, posX, posY, ideaDist]
    return data
