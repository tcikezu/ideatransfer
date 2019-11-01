""" File: test.py
------------------------------------
This file contains the classes idea, member, and community, which are implemented to allow
for some small-scale tests of any of my algorithms. Members inherits idea, and community 
constructs N member objects.
"""

from util import * 
import random

class idea():
    def __init__(self, categories = ['a', 'b', 'c']):
        self.categories = categories
        self.ideas = collections.defaultdict(float)
        self.randomIdeas()

    def randomIdeas(self):
        for idea in self.categories:
            self.ideas[idea] = random.uniform(-1, 1)
        sparseVectorNormalize(self.ideas)

    def getIdeaDistance(self, idea):
        return sparseVectorDistance(self.ideas, idea)

    def agreement(self, ideas):
        if self.ideas == ideas:
            return 1.0
        return sparseVectorDotProduct(self.ideas, ideas) / (sparseVectorDotProduct(self.ideas, self.ideas)**0.5 * sparseVectorDotProduct(ideas, ideas)**0.5)

    def merge(self, gamma, ideas):
        copy = self.ideas
        self.ideas = incrementSparseVector(copy, gamma, ideas)
        ideas = incrementSparseVector(ideas, gamma, copy)

        sparseVectorNormalize(self.ideas)
        sparseVectorNormalize(ideas)


class member(idea):
    def __init__(self):
        super().__init__()
        self.position = [random.uniform(0,1), random.uniform(0,1)] 
        self.threshold = random.uniform(0,1)

    def getPositionDistance(self, member):
        return euclideanDistance(self.position, member.position)

    def agreement(self, member):
        return super().agreement(member.ideas)

    def merge(self,gamma, member):
        super().merge(gamma, member.ideas)

class community():
    def __init__(self, N=3):
        self.N = N
        self.members = [member() for i in range(self.N)]
        self.allThresholds = [self.members[i].threshold for i in range(self.N)]
        self.allPositions = [self.members[i].position for i in range(self.N)]
        self.positionMatrix = collections.defaultdict(float)
        self.createPositionMatrix()

    def getMember(self, index):
        return self.members[index]

    def getIdeas(self, index):
        return self.members[index].ideas
    
    def getPosition(self, index):
        return self.members[index].position

    def getThreshold(self, index):
        return self.members[index].threshold

    def getPositionDistance(self, index1, index2): # Given integer index but ... hm, that may be tedious.
        return self.positionMatrix[(index1, index2)]

    def agreement(self, index1, index2):
        return self.members[index1].agreement(self.members[index2])

    def merge(self, index1, gamma, index2):
        self.members[index1].merge(gamma, self.members[index2])

    def getAllIdeas(self):
        result = collections.defaultdict(list)
        for i in range(self.N):
            for k, v in self.getIdeas(i).items():
                result[k].append(v)
        return result
    
    # at some point, I will need to vectorize all this
    def createPositionMatrix(self):
        for i in range(self.N):
            for j in range(self.N):
                self.positionMatrix[(i,j)] = euclideanDistance(self.getPosition(i), self.getPosition(j))

