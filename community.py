""" File: community.py
------------------------------------
Implementations of community, member, and idea classes. 
Class hierarchy is ideas <-- member <-- community -- which is weird. I might have to 
change that in the future. 

This is different from the other branch because every container used in here is an
ndarray from numpy. Allows for fast computations of idea transfer. 

The member interaction is modeled as stochastic dependent on positions in space. 
Hence, no adjacency or graph network is being modeled here. 

"""

from util import *

######################################################################

DEFAULT_IDEA_SIZE = 3
DEFAULT_COMMUNITY_SIZE = 10 


""" class idea(): 
This class implements idea objects, things that represent some opinion about the environment
or world. Ideas agree or disagree, which is why we also export an agreement method."""
class idea():
    def __init__(self, numberIdeas = None):
        if numberIdeas is None:
            self.numberIdeas = DEFAULT_IDEA_SIZE
        else:
            self.numberIdeas = numberIdeas
        self.ideas = np.random.uniform(-1,1,self.numberIdeas)

    def getIdeaDistance(self, idea):
        return euclideanDistance(self.ideas, idea)

    def agreement(self, ideas):
        if self.ideas == ideas:
            return 1.0
        return cosine(self.ideas, ideas) 

""" Class member: 
    An instance of member, is something that has position, ideas, and threshold. By 
    having ideas, the member is something that has opinions, and we can measure its
    agreement with other members with the idea - inherited agreement method."""
class member(idea):
    def __init__(self, numberIdeas=None):
        super().__init__(numberIdeas)
        self.positionBound = 1
        self.position = np.random.uniform(-self.positionBound, self.positionBound, 2)
        self.threshold = np.random.rand() 

    def getPositionDistance(self, member):
        return euclideanDistance(self.position, member.position)

    def agreement(self, ideas):
        return super().agreement(ideas)

""" community class:
An instance of this class creates a set of members who are close in position, and can
interact by exchanging ideas. The idea exchange is actually implemented in transfer.py.
Here we implement the storing of every member's ideas, thresholds, and positions into
ndarrays. We also implement a fast distance-matrix calculation for positions."""
class community():
    def __init__(self, numberMembers = None, numberIdeas = None):
        if numberMembers is None:
            self.numberMembers = DEFAULT_COMMUNITY_SIZE
        else:
            self.numberMembers = numberMembers
        
        if numberIdeas is None:
            self.numberIdeas = DEFAULT_IDEA_SIZE
        else:
            self.numberIdeas = numberIdeas
        
        self.members = [member(self.numberIdeas) for i in range(self.numberMembers)]
        self.allIdeas = np.ndarray((self.numberMembers, self.numberIdeas))
        self.allThresholds = np.ndarray((self.numberMembers))
        self.allPositions = np.ndarray((self.numberMembers, len(self.members[0].position))) 
        self.defineParameters()
        self.distanceMatrix = self.createDistanceMatrix()
        self.agreementMatrix = self.createAgreementMatrix()

    def defineParameters(self):
        for i in range(self.numberMembers):
            self.allIdeas[i] = self.members[i].ideas
            self.allPositions[i] = self.members[i].position
            self.allThresholds[i] = self.members[i].threshold
        self.allIdeas = normalize(self.allIdeas)
    
    def updateMembers(self):
        for i in range(self.numberMembers):
            self.members[i].ideas = self.allIdeas[i]
            self.members[i].threshold = self.allThresholds[i]
            self.members[i].position = self.allPositions[i]

    def getPositionBounds(self):
        return self.members[0].positionBound

    def getMember(self, index):
        return self.members[index]

    def getIdeas(self, index = None):
        if index is None:   
            return self.allIdeas
        return self.allIdeas[index]

    def getPosition(self, index):
        return self.allPositions[index]

    def getThreshold(self, index):
        return self.allThresholds[index]

    def getPositionDistance(self, index1, index2): 
        return self.distanceMatrix[index, index2]

    def agreement(self, index1, index2):
        return cosine(self.allIdeas[index1], self.allIdeas[index2]) 
           
    def createDistanceMatrix(self):
        return distanceMatrix(self.allPositions)

    def createAgreementMatrix(self):
        return cosineMatrix(self.allIdeas)
