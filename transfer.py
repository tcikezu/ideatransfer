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
    community.allIdeas += gamma*(2*((community.agreementMatrix - community.allThresholds) > 0) - 1\
            - np.eye(community.numberMembers)) @ community.allIdeas
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
    community.allIdeas += gamma*np.multiply(2*((community.agreementMatrix - community.allThresholds) > 0)\
            - 1 - np.eye(community.numberMembers),\
            community.distanceMatrix < np.abs(np.random.normal(0,community.getPositionBounds()*2/3,\
            (community.numberMembers, community.numberMembers)))) @ community.allIdeas
    community.allIdeas = normalize(community.allIdeas)

