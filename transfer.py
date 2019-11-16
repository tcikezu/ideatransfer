""" transfer.py
-----------------------
Branch: distExchange
Date: 11/15/2019

This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. Thus a call of idea transfer updates the
community object by modeling the transfer of ideas between its members -- once.

"""

from util import *

# function probAgreement:
# returns an n x n matrix that calculates the agreement between any pair of members.
def probAgreement(community):
    return 1*(community.agreementMatrix > community.allThresholds) - 1*(community.agreementMatrix < -community.allThresholds) - np.eye(community.numberMembers)

# function probInteraction: 
# returns (A > r) * (g > rand), which is an n x n matrix whose columns correspond to member_i, interacting with all other members j. 
def probInteraction(community):
    return (community.distanceMatrix < community.allRadii) * (community.allGregariousness >  np.random.random(community.numberMembers))


def ideaUpdate(ideaTransfer, community):
    product = np.tensordot(ideaTransfer, community.ideaDistribution, axes=1)
    #product = product*(np.random.random(product.shape)<=0.5)
    return normalizeDistribution(product)

# deterministically merge everyone's ideas with everybody else, each time step.
# For this, we create the indicator, @param agreementMatrix - allThresholds > 0 -- and let this value
# assume 1 if > 0, and -1 if not.
# From this we also subtract diagonal agreements because we don't want people to reinforce their own ideas
# every time step.
def deterministicMerge(community, gamma, t):
    community.ideaDistribution = normalizeDistribution((1-gamma)*community.ideaDistribution + gamma*ideaUpdate(probAgreement(community), community))
    community.resampleIdeas()

# Probabilistically merge everyone's ideas with a subset of the community each time step.
# For this, again we create the indicator, @param agreementMatrix - allThresholds > 0 which assumes +1 
# or -1. This time we also do element-wise multiplication with another term, an indicator for whether
# the distance between two members is inside a tight bound, defined by the normal distribution with mean
# 0 and standard deviation equal to 2/3 * positionBound. (Quick calculation tells me that 2/3 * positionBound
# is in fact the average distance between two points inside the square bounded by (+/- positionBound, +/- positionBound).)
# So on average ... a member will interact with somewhere less than half of all the other members. 
def probabilisticMerge(community, gamma, t):
    ideaTransfer = probAgreement(community) * probInteraction(community)
    community.ideaDistribution = (1 - gamma)*community.ideaDistribution + gamma*ideaUpdate(ideaTransfer, community)
    if t%5 == 0: 
        community.resampleIdeas()


    # to do: right now, transferring all ideas over is making things difficult.
    # what i need to do instead is only transfer idea count seen. 
    # not sure what else can explain the convergence to uniform distributions. 

    # Besides this i need to formulate everything as state-based

def positionUpdate(community, chanceEncounter):
    """ few ideas could work here:
    1: update position to that of best idea + noise
    2: update position to that of worst idea + noise
    3: update position to that of average member agreed with
    4: update position to avg of k best ideas
    5: update position to that of random encounter
    Personally I like the last one the best. But it will probably be very noisy? Don't think positions will equilibriate.
    """


