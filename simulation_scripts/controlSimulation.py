from community import community
from transfer import transfer
import mdp, util, utilRL, random
import numpy as np

# Return i in [0, ..., len(probs)-1] with probability probs[i].
def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)
    
def simulate(rl, numTrials=1, maxIterations=100, verbose=False,sort=False):
    T = maxIterations     
    totalRewards = []  # The rewards we get on each trial
    
    dataX = np.zeros((T,rl.c.numberMembers))
    dataY = np.zeros((T,rl.c.numberMembers))
    dataZ = np.zeros((T,rl.c.numberMembers))
    posX = np.zeros((T,rl.c.numberMembers))
    posY = np.zeros((T,rl.c.numberMembers))
    ideaDist = np.zeros((T,rl.c.domainSize*2+1))
    
    for trial in range(numTrials):
        # Form a brand new community
        c = community(rl.c.numberMembers, rl.c.numberIdeas, rl.c.domainSize)
        

        # Form a brand new agent  
        agentMDP = mdp.agentMDP(c)
        agentMDP.tau = rl.tau
        
        # Update QLearning community to new community
        rl.c = agentMDP.c
        rl.index = agentMDP.index
        
        Transfer = transfer(c)

        state = agentMDP.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for t in range(maxIterations):
            action = rl.getAction(state)
            transitions = agentMDP.succAndProbReward(state, action)

            if trial == numTrials - 1:
                dataX[t,:] = c.allIdeas[:,0]
                dataY[t,:] = c.allIdeas[:,1]
                dataZ[t,:] = c.allIdeas[:,2]
                posX[t,:] = c.allPositions[:,0]
                posY[t,:] = c.allPositions[:,1]
                ideaDist[t,:] = c.ideaDistribution[0][0]
            
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            
            for n in range(agentMDP.tau):
                Transfer.probabilisticMerge()
            newState = agentMDP.observeCommunity()

            totalReward += totalDiscount * reward
            totalDiscount *= agentMDP.discount()
            state = newState
        
        if verbose:
            print(("Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)))
        totalRewards.append(totalReward)
    data = [dataX, dataY, dataZ, posX, posY, ideaDist, totalRewards]
    return data
