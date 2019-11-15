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

DEFAULT_IDEA_SIZE = 100
DEFAULT_DOMAIN_SIZE = 4
DEFAULT_COMMUNITY_SIZE = 300



""" class idea(): 
This class implements idea objects, things that represent some opinion about the environment
or world. Ideas agree or disagree, which is why we also export an agreement method.
"""
class idea():
    def __init__(self, numberIdeas = None, domainSize = None):
        if numberIdeas is None:
            self.numberIdeas = DEFAULT_IDEA_SIZE
        else:
            self.numberIdeas = numberIdeas

        if domainSize is None:
            self.domainSize = DEFAULT_DOMAIN_SIZE
        else:
            self.domainSize = domainSize
        
        self.Lambda = 1
        self.ideaDomain = np.arange(self.domainSize)
        self.ideaDistribution = np.ndarray((self.numberIdeas,self.domainSize))
        self.generateDistribution()
        self.ideas = np.ndarray(self.numberIdeas)
        self.sampleIdeas()

    def generateDistribution(self):
        for i in range(self.numberIdeas):
            self.ideaDistribution[i] = np.ones(self.domainSize)*self.Lambda
            self.ideaDistribution[i] = self.ideaDistribution[i] / sum(self.ideaDistribution[i])

    def sampleIdeas(self):
        for i in range(self.numberIdeas):
            self.ideas[i] = np.random.choice(self.ideaDomain, p=self.ideaDistribution[i])

    #def getIdeaDistance(self, idea):
        #return euclideanDistance(self.ideas, idea)

    def agreement(self, ideas):
        if self.ideas == ideas:
            return 1.0
        return cosine(self.ideas, ideas) 

""" Class member: 
    An instance of member, is something that has position, ideas, and threshold. By 
    having ideas, the member is something that has opinions, and we can measure its
    agreement with other members with the idea - inherited agreement method."""
class member(idea):
    def __init__(self, numberIdeas=None, domainSize = None):
        super().__init__(numberIdeas)
        self.positionBound = 1
        self.position = np.random.uniform(-self.positionBound, self.positionBound, 2)
        self.threshold = np.random.rand()
        self.radius = np.random.rand()
        self.gregariousness = np.random.rand()

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
    def __init__(self, numberMembers = None, numberIdeas = None, domainSize = None):
        if numberMembers is None:
            self.numberMembers = DEFAULT_COMMUNITY_SIZE
        else:
            self.numberMembers = numberMembers
        
        if numberIdeas is None:
            self.numberIdeas = DEFAULT_IDEA_SIZE
        else:
            self.numberIdeas = numberIdeas

        if domainSize is None:
            self.domainSize = DEFAULT_DOMAIN_SIZE
        else:
            self.domainSize = domainSize
        
        self.members = [member(self.numberIdeas,self.domainSize) for i in range(self.numberMembers)]
        self.allIdeas = np.ndarray((self.numberMembers, self.numberIdeas))
        self.allThresholds = np.ndarray(self.numberMembers)
        self.allPositions = np.ndarray((self.numberMembers, len(self.members[0].position))) 
        self.allRadii = np.ndarray(self.numberMembers)
        self.allGregariousness = np.ndarray(self.numberMembers)
        self.defineParameters()
        self.distanceMatrix = self.createDistanceMatrix()
        self.agreementMatrix = self.createAgreementMatrix()

    def defineParameters(self):
        for i in range(self.numberMembers):
            self.allIdeas[i] = self.members[i].ideas
            self.allPositions[i] = self.members[i].position
            self.allThresholds[i] = self.members[i].threshold
            self.allRadii[i] = self.members[i].radius
            self.allGregariousness[i] = self.members[i].gregariousness
    
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
        return cosineMatrix(self.allIdeas) - np.eye(self.numberMembers)
    
