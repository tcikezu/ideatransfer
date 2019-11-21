"""
File: mdp.py
-------------------
Here I attempt to code up a MDP for an agent that is trying to spread its beliefs to the rest of the members of the community. The environment is the community, its members, and the state will comprise the members' beliefs and position. The reward is the community agreement with the member. The set of actions is twofold -- an action comprises a movement in some direction, and also an interaction with some subset of the community. 
"""

from utilRL import MDP
from community import *


# Rather than encode the entire community as a markov-decision-process, I am going to make
# the community as purely data, an environment that the agent MDP interacts with.

# the main issue is computing succAndProbReward.
# Here is what should happen: there are 2 actions to take - move, or interact with others. 
# But within move and interact, there are sub-actions. 
### For move, can move pretty much anywhere. But where exactly, ought to be computed in some
# reasonable way. Probably the best way, is to train a neural network that calculates a reward based on some heuristic, and moves in the way that maximizes reward? Unsure. 
# Within interactions -- this is a decision that has to rely not on the member index, but actual features of the member. How to code this ... well, I need a feature vector for the member. 

# agentMDP should incorporate elements of member class, community class, and transfer module.
# Maybe I should code things up from scratch and then determine how classes fit together and such. ...


# Actually, here's an idea. I'm going to create an N+1'st member of community class.
# At community initialization, I am going to update the N+1st member's gregariousness, 
# radii, everything -- for all intents and purposes, this is a valid member of the community.

# However, I am going to modify the transfer function, so that probAgreement, probInteraction, all these things, will be modified by the agentMDP. Also, the agentMDP will listen to everything that is happening to the N+1st member, as well as the rest of the community, to make informed decisions. 

# Done this way, it should be trivial to add even more agents to the community. 


class agentMDP(MDP):
    def __init__(self, numberIdeas=3, domainSize=10):
        self.numberIdeas = numberIdeas
        self.domainSize = domainSize
        self.index = community.numberMembers + 1;
    
    def startState(self):
        
