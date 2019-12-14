from community import community
from transfer import transfer
import mdp, util, utilRL, copy, time, random, cloudpickle
import numpy as np

############################# Choose a simulation #############################
from simulateRL import simulate
# from controlSimulation import simulate

######################### Take care of filename (pkl extension added later) ##########################
fn='N=100_NT=10000_samp=10_ag_dist_nomove'

######################## community parameters ##############################
c = community(100,4,10)
c.allRadii = np.random.normal(0.4, 0.1, c.numberMembers)

######################### agent parameters ############################
agentMDP = mdp.agentMDP(community = c, index=0, sampleSize=10)
agentMDP.T = 100
agentMDP.tau = 10

rl = mdp.QLearningAlgorithm(agentMDP.index, agentMDP.tau, agentMDP.c, agentMDP.actions, agentMDP.discount(), mdp.communityFeatureExtractor, explorationProb = 0.2)            

######################## number trials ###############################
data = simulate(rl, numTrials=10000, maxIterations=100)

with open(fn+'.pkl', 'wb') as outfile:
    cloudpickle.dump(data, outfile, cloudpickle.HIGHEST_PROTOCOL)

print('community radius', np.mean(c.allRadii), 'numberMembers', c.numberMembers, 'agent radius', c.allRadii[agentMDP.index], 'agent tau', agentMDP.tau, 'agent samplesize', 10)

