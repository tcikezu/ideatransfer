import util
from plotCommunity import plotCommunity
from transfer import transfer
from community import community
import numpy as np
from simulateWithoutAgent import simulate
# from controlSimulation import simulate
import matplotlib.pyplot as plt
import copy, pickle
##############################################################

c = community(400, 4, 10)
parameterRange = [0.4]
distanceMatrices = []
for r in parameterRange:
    c.allRadii = (np.random.normal(r,0.1,c.numberMembers)**2 + np.random.normal(r,0.1, c.numberMembers)**2)**0.5
    data = simulate(c, T=200)
#     .append(c.distanceMatrix) 

#     agentMDP = mdp.agentMDP(community = c, index=0, sampleSize=10)
#     agentMDP.T = 100
#     agentMDP.tau = 10
#     rl = mdp.QLearningAlgorithm(agentMDP.index, agentMDP.tau, agentMDP.c, agentMDP.actions, agentMDP.discount(), mdp.communityFeatureExtractor, explorationProb = 0.2)            
#     data = simulate(rl, numTrials=1, maxIterations=100)
#     distanceMatrices.append(rl.c.distanceMatrix)

    fn = 'data_radius_'+str(r)+'_N_'+str(c.numberMembers)+'.pkl'
    with open(fn, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
