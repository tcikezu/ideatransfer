""" transfer.py
-----------------------
This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. Thus a call of idea transfer updates the
community object by modeling the transfer of ideas between its members -- once.

Note: this is the numpy version!
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

# deterministically merge everyone's ideas with everybody else, each time step.
# For this, we create the indicator, @param agreementMatrix - allThresholds > 0 -- and let this value
# assume 1 if > 0, and -1 if not.
# From this we also subtract diagonal agreements because we don't want people to reinforce their own ideas
# every time step.
def deterministicMerge(community, gamma):
    community.allideas += gamma*probAgreement(community) @ community.allIdeas
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

    ideaTransfer = probAgreement(community) * probInteraction(community)
    
    community.communityIdeaDistribution = (1 - gamma)*community.communityIdeaDistribution + \
        gamma * ideaTransfer @ community.communityIdeaDistribution * \
        np.random.random((community.numberMembers, community.numberIdeas, community.domainSize))
    # generate allIdeas here, again. 
    # the below code probably doesn't work. but it's gonna be something like this
    community.allIdeas = community.allIdeas[np.random.randint(community.allIdeas.shape[0],1)]


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
