from community import community
from transfer import transfer
import mdp, util, utilRL, copy, time, cloudpickle
import numpy as np

############################# Choose a simulation #############################
from simulateRL import simulate
#from controlSimulate import simulate

startTime = time.time()
######################### Take care of filename (pkl extension added later) ##########################
fn='ag_dist_rad_gre'

######################## community parameters ##############################
c = community(100,4,10)
c.allRadii = np.random.normal(0.4, 0.1, c.numberMembers)

######################### agent parameters ############################
agentMDP = mdp.agentMDP(community = c, index=0, sampleSize=4)
agentMDP.T = 100
agentMDP.tau = 10

rl = mdp.QLearningAlgorithm(agentMDP.index,agentMDP.tau, agentMDP.c, agentMDP.actions, agentMDP.discount(), mdp.communityFeatureExtractor, explorationProb = 0.2, alpha=0.1)

with open(fn+'.pkl', 'rb') as infile:
    data = cloudpickle.load(infile)
rl.weights = data[-2]
print(len(rl.weights))

######################## number trials ###############################
data = simulate(rl, numTrials=10000, maxIterations=agentMDP.T)
data += [rl.weights]
data += [c]
with open(fn+'_10000.pkl', 'wb') as outfile:
    cloudpickle.dump(data, outfile, cloudpickle.DEFAULT_PROTOCOL)

print('community radius', np.mean(c.allRadii), 'numberMembers', c.numberMembers, 'agent radius', c.allRadii[0], 'agent tau', agentMDP.tau, 'agent samplesize', 4)

print('explorationProb', 0.2, 'numberTrials', 10000, 'maxIterations', 100)

print('totaltime spent: ', time.time() - startTime)

