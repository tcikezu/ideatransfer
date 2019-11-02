""" transfer.py
-----------------------
This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. Thus a call of idea transfer updates the
community object by modeling the transfer of ideas between its members -- once.

Note: this is the numpy version!
"""

from util import *

def deterministicMerge(community, gamma):
    community.allIdeas += gamma*(2*((community.agreementMatrix - community.allThresholds) > 0) - 1 - np.eye(community.numberMembers)) @ community.allIdeas
    community.allIdeas = normalize(community.allIdeas)

def probabilisticMerge(community, gamma):
    community.allIdeas += gamma*(2*((community.agreementMatrix - community.allThresholds) > np.random.normal

