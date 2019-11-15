""" transfer.py
-----------------------
This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. Thus a call of idea transfer updates the
community object by modeling the transfer of ideas between its members -- once.

Note: this is the numpy version!
"""

from util import *

# deterministically merge everyone's ideas with everybody else, each time step.
# For this, we create the indicator, @param agreementMatrix - allThresholds > 0 -- and let this value
# assume 1 if > 0, and -1 if not.
# From this we also subtract diagonal agreements because we don't want people to reinforce their own ideas
# every time step. Finally we normalize.
def deterministicMerge(community, gamma):
    agreement = 1*(community.agreementMatrix > community.allThresholds) - 1*(community.agreementMatrix < -community.allThresholds) - np.eye(community.numberMembers)
    #agreement = agreement * community.agreementMatrix # This might be better -- but tbh, it doesn't matter much

    community.allIdeas = (1 - gamma)*community.allIdeas + gamma * agreement @ community.allIdeas 
    community.allIdeas = normalize(community.allIdeas)

# Probabilistically merge everyone's ideas with a subset of the community each time step.
# For this, again we create the indicator, @param agreementMatrix - allThresholds > 0 which assumes +1 
# or -1. This time we also do element-wise multiplication with another term, an indicator for whether
# the distance between two members is inside a tight bound, defined by the normal distribution with mean
# 0 and standard deviation equal to 2/3 * positionBound. (Quick calculation tells me that 2/3 * positionBound
# is in fact the average distance between two points inside the square bounded by (+/- positionBound, +/- positionBound).)
# So on average ... a member will interact with somewhere less than half of all the other members. 
# So these two constraints combine multiplicatively for the idea update / transfer. And again, we normalize at end of each step.
def probabilisticMerge(community, gamma):
    N = community.numberMembers
    
    # chanceEncounter -- (A > r) * (g > rand) returns an n x n matrix, whose columns correspond to member_i, interacting with
    # all other members j. 
    # chanceAgreement -- returns an n x n matrix that calculates the agreement between any pair of members.
    chanceEncounter = (community.distanceMatrix < community.allRadii) * (community.allGregariousness >  np.random.random(N))
    agreement = 1*(community.agreementMatrix > community.allThresholds) - 1*(community.agreementMatrix < -community.allThresholds) - np.eye(community.numberMembers)

    ideaTransfer = agreement * chanceEncounter
    #community.allIdeas = (1-gamma)*community.allIdeas + gamma * ideaTransfer @ community.allIdeas
    community.allIdeas += gamma*ideaTransfer @ community.allIdeas
    community.allIdeas = normalize(community.allIdeas)
    

def positionUpdate(community, chanceEncounter):
    """ few ideas could work here:
    1: update position to that of best idea + noise
    2: update position to that of worst idea + noise
    3: update position to that of average member agreed with
    4: update position to avg of k best ideas
    5: update position to that of random encounter
    Personally I like the last one the best. But it will probably be very noisy? Don't think positions will equilibriate.
    """

    #community.allPositions = 
