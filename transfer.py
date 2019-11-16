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
    np.random.seed()
    return (community.distanceMatrix < community.allRadii) * (community.allGregariousness >  np.random.random(community.numberMembers))


def ideaUpdate(ideaTransfer, community):
    product = np.tensordot(ideaTransfer, community.ideaDistribution, axes=1)
    #product = product*(np.random.random(product.shape)<=0.5)
    return normalizeDistribution(product)
    #return softMaxDistribution(product)

# deterministically merge everyone's ideas with everybody else, each time step.
# For this, we create the indicator, @param agreementMatrix - allThresholds > 0 -- and let this value
# assume 1 if > 0, and -1 if not.
# From this we also subtract diagonal agreements because we don't want people to reinforce their own ideas
# every time step.
def deterministicMerge(community, gamma, t):
    #community.ideaDistribution = softMaxDistribution((1-gamma)*community.ideaDistribution + gamma*ideaUpdate(probAgreement(community), community))

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
    community.ideaDistribution = normalizeDistribution((1 - gamma)*community.ideaDistribution + gamma*ideaUpdate(ideaTransfer, community))
    if t%5 == 0: 
        community.resampleIdeas()
        positionUpdate(community, ideaTransfer,0.1)

def positionUpdate(community,ideaTransfer,beta):
    """ few ideas could work here:
    1: update position to that of best idea + noise
    2: update position to that of worst idea + noise
    3: update position to that of average member agreed with
    4: update position to avg of k best ideas
    5: update position to that of random encounter
    Personally I like the last one the best. But it will probably be very noisy? Don't think positions will equilibriate.
    """
    np.random.seed()

    # This implementation -- if person A likes person B, person A moves towards B.
    # If person A doesn't like person B, person A moves away from person B.
    # Finally, every person moves randomly in position space. 
    # if all goes well, the net movement is on average, 0, so people should on average
    # stay in roughly the same space as where they began -- ie, in the unit square.
    # we'll see if that is truly the case. 
    deltaAttract = (ideaTransfer>0).reshape(community.numberMembers,community.numberMembers,1)*community.differenceMatrix
    deltaRepel = (ideaTransfer < 0).reshape(community.numberMembers,community.numberMembers,1)*community.differenceMatrix
    community.allPositions += (2*np.random.rand(community.numberMembers,2)-1)*beta + beta*deltaAttract.sum(axis=1) - beta*deltaRepel.sum(axis=1)

    community.distanceMatrix = community.createDistanceMatrix()
    community.differenceMatrix = community.createDifferenceMatrix()

