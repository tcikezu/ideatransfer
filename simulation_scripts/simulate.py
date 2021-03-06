from transfer import transfer
from mdp import agentMDP
from community import *
import numpy as np

def schedule(c=None, T = 80):
    if c is None:
        c = community(100,3,10)

    Transfer = transfer(c)
    agent = agentMDP()

    # Data to store c.allIdeas, to then make animation
    dataX = np.zeros((T, c.numberMembers))
    dataY = np.zeros((T, c.numberMembers))
    dataZ = np.zeros((T, c.numberMembers))

    posX = np.zeros((T, c.numberMembers))
    posY = np.zeros((T, c.numberMembers))

    # Iterate the idea transfer throughout community
    for t in range(T):
        np.random.seed()
        dataX[t,:] = c.allIdeas[:,0]
        dataY[t,:] = c.allIdeas[:,1]
        dataZ[t,:] = c.allIdeas[:,2]

        posX[t,:] = c.allPositions[:,0]
        posY[t,:] = c.allPositions[:,1]
        Transfer.probabilisticMerge(t)
