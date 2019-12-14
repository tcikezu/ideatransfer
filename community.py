""" File: community.py
------------------------------------
Implementations of community, member, and idea classes.

A community has members, each of which has ideas.

The member interaction is modeled as stochastic dependent on positions in some real space.
Hence, no adjacency or graph network is being modeled here.

"""
import util, random
import numpy as np
from collections import defaultdict


######################################################################
# Default parameter values
DEFAULT_IDEA_SIZE = 4
DEFAULT_DOMAIN_SIZE = 10
DEFAULT_COMMUNITY_SIZE = 100

# Class: Idea
# -----------------------------------
# This class implements idea objects, things that represent some opinion about the environment
# or world.
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

        self.ideaDomain = np.arange(-self.domainSize, self.domainSize+1)
        self.ideaDistribution = np.ndarray((self.numberIdeas,self.domainSize*2+1))
        self.generateDistribution()
        self.ideas = np.ndarray(self.numberIdeas)
        self.sampleIdeas()

    def generateDistribution(self):
        for i in range(self.numberIdeas):
            self.ideaDistribution[i] = np.random.rand(self.domainSize*2+1)
            self.ideaDistribution[i] = self.ideaDistribution[i]/sum(self.ideaDistribution[i])

    def sampleIdeas(self):
        for i in range(self.numberIdeas):
            self.ideas[i] = np.random.choice(self.ideaDomain, p = self.ideaDistribution[i])

######################################################################################

# Class:  Member
# An instance of member is something that has position, ideas, and threshold. By
# having ideas, the member is something that has opinions, and we can measure its
# agreement with other members with the idea - inherited agreement method.
# In addition, members have parameters: gregariousness, radius, velocity, and gamma.
# Gregariousness: rate of interaction with other members
# Radius: limit to physical distane of other members, a member can interact with
# velocity: average velocity with which member moves in physical space. 0 < velocity < 1
# gamma: controls rate of adding new ideas / rate of forgetting old ideas, 0 < gamma < 1
class member(idea):
    def __init__(self, numberIdeas=None, domainSize = None):
        super().__init__(numberIdeas, domainSize)
        self.positionBound = 1
        self.positionDimensions = 2

        self.position = np.random.uniform(-self.positionBound, self.positionBound, self.positionDimensions)
        self.threshold = np.random.normal(0.5,0.1)
        self.radius = (np.random.normal(0.2,0.1)**2 + np.random.normal(0.2,0.1)**2)**0.5
        self.gregariousness = np.random.normal(0.5,0.1)
        self.velocity = np.random.lognormal(-3,0.5)*self.positionBound
        self.gamma = np.random.lognormal(-2.5, 0.1)

#######################################################################################
# Class: Community
# An instance of this class creates a set of members who are close in position, and can
# interact by exchanging ideas. The idea exchange is actually implemented in transfer.py.
# Here we implement the storing of all member parameters (e.g. thresholds).
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
        self.ideaDistribution = np.ndarray((self.numberMembers, self.numberIdeas, self.domainSize*2+1))
        self.positionDimensions = self.members[0].positionDimensions
        self.domain = np.arange(-self.domainSize, self.domainSize+1)
        self.allThresholds = np.ndarray(self.numberMembers)
        self.allPositions = np.ndarray((self.numberMembers, self.positionDimensions))
        self.allRadii = np.ndarray(self.numberMembers)
        self.allGregariousness = np.ndarray(self.numberMembers)
        self.allVelocities = np.ndarray((self.numberMembers, self.positionDimensions))
        self.allGamma = np.ndarray(self.numberMembers)
        self.updateCommunity()

        self.distanceMatrix = self.createDistanceMatrix()
        self.agreementMatrix = self.createAgreementMatrix()
        self.differenceMatrix = self.createDifferenceMatrix()
        self.ideaDistribution = util.normalizeDistribution(self.ideaDistribution)
        self.ideaDistanceMatrix = self.createIdeaDistanceMatrix()

    def updateCommunity(self):
        for i in range(self.numberMembers):
            self.allIdeas[i] = self.members[i].ideas
            self.ideaDistribution[i] = self.members[i].ideaDistribution
            self.allPositions[i] = self.members[i].position
            self.allThresholds[i] = self.members[i].threshold
            self.allRadii[i] = self.members[i].radius
            self.allGregariousness[i] = self.members[i].gregariousness
            self.allVelocities[i] = self.members[i].velocity
            self.allGamma[i] = self.members[i].gamma
        self.allGamma = self.allGamma[:,np.newaxis].reshape(self.numberMembers, 1, 1)

    def updatePosition(self, index, newPosition):
        self.allPositions[index] = newPosition
        self.distanceMatrix = self.createDistanceMatrix()
        self.differenceMatrix = self.createDifferenceMatrix()

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

    def getNeighbors(self, index):
        return [i for i,d in enumerate(self.distanceMatrix[index]) if i != index and d < self.allRadii[index]]

    def getThreshold(self, index):
        return self.allThresholds[index]

    def getPositionDistance(self, index1, index2):
        return self.distanceMatrix[index, index2]

    def agreement(self, index1, index2):
        return util.cosine(self.allIdeas[index1], self.allIdeas[index2])

    def createDistanceMatrix(self):
        return util.distanceMatrix(self.allPositions)

    def createAgreementMatrix(self):
        return util.cosineMatrix(self.allIdeas)

    def createIdeaDistanceMatrix(self):
        return util.distanceMatrix(self.allIdeas)

    # function: Resample Ideas
    # --------------------------
    # Some random subset of members resample their ideas from their underlying
    # idea distributions.
    # The sampling frequency ... may be some function of gregariousness, threshold.
    # Or it may be random.
    def resampleIdeas(self, proportion):
        sampleSize = int(proportion * self.numberMembers)
        sampleMembers = np.random.choice(np.arange(self.numberMembers), size=sampleSize, replace=False)
#         self.allIdeas[sampleMembers] = np.random.normal(self.domain[util.resampleDistribution(self.ideaDistribution, sampleMembers)],1 - self.allThresholds[sampleMembers, np.newaxis].reshape(sampleSize, 1))
        self.allIdeas[sampleMembers] = np.random.normal(self.domain[util.resampleDistribution(self.ideaDistribution, sampleMembers)], 1)
        self.agreementMatrix = self.createAgreementMatrix()

    def createDifferenceMatrix(self):
        P1 = self.allPositions.reshape(self.numberMembers,1,self.positionDimensions).repeat(self.numberMembers,axis=1)
        P2 = np.transpose(P1, axes=[1,0,2])
        return P2 - P1
