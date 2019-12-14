"""
File: mdp.py
-------------------
MDP for an agent that is trying to spread its beliefs to the rest of the members of the community. The environment is the community, its members, and the state will comprise the members' beliefs and position. The reward is the community agreement with the member. The set of actions is twofold -- an action comprises a movement in some direction, and also an interaction with some subset of the community.
"""

import util, utilRL, copy, random, math
import numpy as np
import community
from utilRL import ValueIteration
from transfer import transfer
from collections import defaultdict
from datetime import datetime

date_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
######################################################################

# Class : Agent MDP
# --------------------
# A MDP representing an agent ... (how to incorporate 2+ agents? how to ensure that the community is the same for both agents?)

# Plan : there will be an instance of community the agent will take from.


class agentMDP(utilRL.MDP):
    # I suppose that if I let index be a setting here, thne I can allow multiple
    # agents to be a part of the same community. Still need to address rest of the code
    # that assumes agent to be member 0
    def __init__(self, community, index=0, sampleSize=10):
        self.index = index
        np.random.seed()
        community.ideaDistribution[self.index] *= 0 

        # modify agent beliefs to be delta fcns; believes only one thing 
        for i in range(community.numberIdeas):
            community.ideaDistribution[self.index][i][np.random.randint(2*community.domainSize+1)] = 1
        
        # normalize the idea distribution -- although this is already normalized 
        community.ideaDistribution = util.normalizeDistribution(community.ideaDistribution)
        community.allRadii[self.index] = 0.5
        
        # make sure agent can't change beliefs --> so agent can learn to change member beliefs
        # and not its own beliefs.
        community.allGamma[self.index] = 0
        self.c = community
        # Only issue I have with this is that tau isn't a shared variable, so different agents may have different tau, and different agents may continue on for different periods of time
        self.T = 100 # number of time steps we let the simulation run
        self.tau = 10 # number of time steps we let the community run between each step of agent.
        self.sampleSize = sampleSize

    # function : Observe Community --- state definition
    # ------------------------------
    # This function retrieves a list of member tuples: (relative position, member index).
    # Some # of members is sampled randomly from community.
    # The rest are all neighbors to the agent. 
    # Note that positions and agreement values are rounded to nearest tenth.
    def observeCommunity(self, community = None):
        if community is None:
            community = self.c

        # retrieve neighbor and sample member indices
        neighbors = community.getNeighbors(self.index)
        sampleMembers = np.random.choice(np.arange(community.numberMembers), size=self.sampleSize, replace=False).tolist()

        # Retrieve (relative position, member index) tuples
        neighborState = tuple(zip([tuple(x) for x in np.round(community.allPositions[neighbors] - community.allPositions[self.index],1).tolist()], neighbors))
        sampleState = tuple(zip([tuple(x) for x in np.round(community.allPositions[sampleMembers] - community.allPositions[self.index],1).tolist()], sampleMembers))
        
        return tuple([neighborState, sampleState])

    # function : Get Reward
    # ---------------------
    # Attempt one: reward is given at the end of an episode, and is the sum
    # of all agreements to agent.
    # Attempt two: reward is -1*(sum of squared distances in idea space)
    # Awarded *every* step? 
    def getReward(self, community = None):
        if community is None:
            community = self.c
        return np.sum(community.agreementMatrix[self.index])*(self.T < 1) # self.T decreased before call
        
        #return np.sum(community.ideaDistanceMatrix[self.index])

    # function : start state
    # --------------------------
    # state = ((all neighbors), (randomly sampled members))
    def startState(self):
        return self.observeCommunity()

    # Function : actions
    # -------------------
    # state: ((neighbors), (randomly sampled))
    def actions(self, state):
        neighbors = state[0]
        sampleMembers = state[1]

        actions = [('move', index) for _,index in sampleMembers]
        if len(neighbors) == 0:
            return actions
        return actions + [tuple(['interact'])]

    # Function: (Succ)essor and (Prob)ability Reward
    # ----------------------------------------------
    # Given a state action (s,a) pair, return a list of (Succ(s,a), T(s,a,Succ(s,a)), R(s,a,Succ(s,a)) tuples.
    def succAndProbReward(self, state, action):
        # Something to note is that i could udpate an internal gregariousness counter -- see how many
        # times it chose to move vs interact. but maybe i can see that in post
        # Reached end state
        results = []
        if self.T  == 0:
            return []

        if action[0] == 'move':
            newPosition = np.round(self.c.allPositions[action[1]],1)
            self.c.updatePosition(self.index, newPosition)
            self.T -= 1
            return [(self.observeCommunity(), 1.0, self.getReward())]

        if action[0] == 'interact':
            ## something I may do is uncomment below, to enable the agent to 100% affect neighbors.
            ## I think it's cheating, so for now it shall remain uncommented. 
            #for i in neighbors:
            #    self.c.ideaDistribution[i] = (1 - self.c.allGamma[i]) * self.c.ideaDistribution[i] + self.c.allGamma[i]*self.c.ideaDistribution[self.index]
            #    self.c.ideaDistribution = util.normalizeDistribution(self.c.ideaDistribution)
            self.T -= 1
            return [(self.observeCommunity(), 1.0, self.getReward())]

        #if action == 'move':
        #    newPositions = [state[i][0] for i in range(len(state))]
        #    for newPos in newPositions:
        #        c_copy = copy.deepcopy(self.c)
        #        Transfer = transfer(c_copy)
        #        c_copy.updatePosition(0, np.array(newPos) + c_copy.allPositions[self.index])
        #        results.append((self.stepForward(c_copy, Transfer), 1.0/len(state), self.getReward(c_copy)*(self.T==1)))
        #    self.T -= 1
        #    return results
        #if action == 'interact':
        #    for i in range(len(state)):
        #        c_copy = copy.deepcopy(self.c)
        #        Transfer = transfer(c_copy)
        #        c_copy.ideaDistribution[i] = (1 - c_copy.allGamma[i]) * c_copy.ideaDistribution[i] + c_copy.allGamma[i]*c_copy.ideaDistribution[self.index]
        #        c_copy.ideaDistribution = util.normalizeDistribution(c_copy.ideaDistribution)
        #        results.append((self.stepForward(c_copy, Transfer), 1.0/len(state), self.getReward(c_copy)))
        #    self.T -= 1
        #    return results

    # Function: discount
    # --------------------------
    # If I want discounts, I need partial rewards. But what is a good reward?
    def discount(self):
        return 1

class QLearningAlgorithm(utilRL.RLAlgorithm):
    def __init__(self, index, tau, community, actions, discount, featureExtractor, explorationProb=0.2):
        self.index = index
        self.tau = tau
        self.c = community
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        score = 0
        for f,v in self.featureExtractor(self.index, self.c, state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights:
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        estimateValue = max(self.getQ(newState, a) for a in self.actions(newState))
        util.incrementSparseVector(self.weights, -1*self.getStepSize()*(self.getQ(state, action) - (reward + self.discount*estimateValue)), self.featureExtractor(self.index, self.c, state, action))

def communityFeatureExtractor(agentIndex, community, state, action):
    if state is not None:
        feature = []
        neighbors = state[0]
        sampledMembers = state[1]

        ######################### Featurize state ############################
        # number of neighbors
        numberNeighbors = len(neighbors)

        # average agreement
        averageAgreementNeighbors = round(sum([np.round(community.agreementMatrix[agentIndex][member[1]],1) for member in neighbors]),1)
        averageAgreementSampled = round(sum([np.round(community.agreementMatrix[agentIndex][member[1]],1) for member in sampledMembers]),1)
        
        # distances from sampled members
        # rounding to the nearest digit -- keep things simple
        sampleDistances = [np.round(community.distanceMatrix[agentIndex][member[1]]) for member in sampledMembers]
        
        # agreements of sampled member
        sampleAgreements = [np.round(community.agreementMatrix[agentIndex][member[1]],1) for member in sampledMembers]
        
        ######################## Featurize Action ############################
        if action[0] == 'move': 
            # radius of sample member
            rad = np.round(community.allRadii[action[1]], 1)

            # gregariousness of sample member
            # gre = np.round(community.allGregariousness[action[1],1)

            # agreement of sample member
            ag = np.round(community.agreementMatrix[agentIndex][action[1]],1)
        
            # distance of sample member to agent
            # again, I am rounding this to nearest digit
            dist = np.round(community.agreementMatrix[agentIndex][action[1]])
           
            # simplest feature in my opinion
            feature.append(((action[0], averageAgreementNeighbors, ag, dist), 1))
            #feature.append(((action[0], averageAgreementNeighbors, averageAgreementSampled, \
                #frozenset(sampleDistances), frozenset(sampleAgreements), rad, ag, dist), 1))
        else:
            feature.append(((action[0], averageAgreementNeighbors), 1))
            #feature.append(((action[0], averageAgreementNeighbors, averageAgreementSampled, \
                #frozenset(sampleDistances), frozenset(sampleAgreements)), 1))
       
        # Regardless of what the sample population looks like. 
        feature.append(((numberNeighbors, action[0]), 1))
 
        return feature
    else:
        return []

def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]
