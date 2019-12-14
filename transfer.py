""" transfer.py
-----------------------
Branch: RL Agent

This file implements the idea transfer class. The methods exported accept a
community object, and modifies it. 

Something worth considering is modifying all previous versions of transfer.py to be classes and not modules? For backwards-compatibility reasons. 

"""

import util
import numpy as np
##########################################################################

class transfer():
    def __init__(self, community):
        self.c = community
   
    # function Conditonal Agreement:
    # ---------------------------
    # returns an n x n matrix that calculates the agreement between any pair of members, 
    # with threshold that's member-specific. (specifically, |A_{ij}| > threshold_i, for 
    # all members i.)
    def conditionalAgreement(self):
        return self.c.agreementMatrix*(1*(self.c.agreementMatrix > self.c.allThresholds) + 1*(self.c.agreementMatrix < -self.c.allThresholds) - 1*np.eye(self.c.numberMembers))

    # function probInteraction: 
    # ---------------------------
    # returns (D > r) * (g > rand), which is an n x n matrix whose columns 
    # correspond to member_i, interacting with all other members j.
    # More specifically, (D_{ij} > radius_{i}) * (gregariousness_i > np.random.rand, for
    # each member i.) 
    def probInteraction(self):
        np.random.seed()
        return (self.c.distanceMatrix < self.c.allRadii) * (self.c.allGregariousness >  np.random.random(self.c.numberMembers))

    def ideaUpdate(self, ideaTransfer):
        product = np.tensordot(ideaTransfer, self.c.ideaDistribution, axes=1)
        #product = product*(np.random.random(self.numberMembers, self.numberIdeas, self.domainSize)<=0.5)
        return product

    # deterministically merge everyone's ideas with everybody else, each time step.
    def deterministicMerge(self):
        ideaTransfer = self.conditionalAgreement()
        self.c.ideaDistribution = util.normalizeDistribution((1-gamma)*self.c.ideaDistribution + self.c.allGamma*self.ideaUpdate(ideaTransfer))
        self.c.resampleIdeas(0.1)
        #self.positionUpdate(ideaTransfer)

    # Probabilistically merge everyone's ideas with a subset of the community each time step.
    def probabilisticMerge(self):
        ideaTransfer = self.conditionalAgreement() * self.probInteraction()
        self.c.ideaDistribution = util.normalizeDistribution((1 - self.c.allGamma)*self.c.ideaDistribution + self.c.allGamma * self.ideaUpdate(ideaTransfer))

        self.c.resampleIdeas(0.1)
        self.positionUpdate(ideaTransfer)

    def positionUpdate(self, ideaTransfer):
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
        # Finally, every person moves randomly / diffuses in position space. 
        # A diffusion term is needed for a similar reason to why we need heat 
        # to make chemicals react -- people need some thermal energy to find each before they can interact

        # multiply each diffMatrix(i,j) by the corresponding ideaTransfer(i,j)
        deltaAttraction = (ideaTransfer*(ideaTransfer > 0)).reshape(self.c.numberMembers, self.c.numberMembers, 1) * self.c.differenceMatrix
        deltaRepulsion = (ideaTransfer*(ideaTransfer < 0)).reshape(self.c.numberMembers, self.c.numberMembers, 1) * self.c.differenceMatrix

        # I wonder if I could learn parameters here that achieve some desirable effect (e.g. like a power-law distribution of cluster sizes... etc)

        # sum columns together to create position updates
        self.c.allPositions += (np.random.normal(0, 1, (self.c.numberMembers, self.c.positionDimensions)) + deltaAttraction.sum(axis=1) + deltaRepulsion.sum(axis=1))*self.c.allVelocities
        self.c.distanceMatrix = self.c.createDistanceMatrix()
        self.c.differenceMatrix = self.c.createDifferenceMatrix()

