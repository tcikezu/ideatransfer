""" transfer.py
-----------------------
This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. Thus a call of idea transfer updates the
community object by modeling the transfer of ideas between its members -- once.
"""

from util import *

def merge(ideas0, gamma, ideas1):
    incrementSparseVector(ideas0, gamma, ideas1)
    sparseVectorNormalize(ideas0)

def deterministicMerge(community, gamma):
    for i in range(community.N):
        for j in range(community.N):
            if (i != j):
                if (community.agreement(i,j) >= community.getThreshold(i)):
                    merge(community.getIdeas(i), gamma, community.getIdeas(j))
                else:
                    merge(community.getIdeas(i), -gamma, community.getIdeas(j))

def probabilisticMerge(community, gamma):
    for i in range(community.N):
        for j in range(community.N):
            if (i != j):
                if (community.getPositionDistance(i,j)\
                        /(math.sqrt(2)*community.getPositionBounds()) < random.random()):
                    if (community.agreement(i,j) >= community.getThreshold(i)):
                        merge(community.getIdeas(i), gamma, community.getIdeas(j))
                    else:
                        merge(community.getIdeas(i,), -gamma, community.getIdeas(j))
