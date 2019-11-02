""" simulate.py
-----------------------
This file creates a community object, defined in test.py, and runs to see how ideas
evolve over some number of iterations."""

import transfer
from community import *

def simulate(gamma, T):
    X = community(40,5)
    print(X.getIdeas())

    for t in range(T):
        transfer.deterministicMerge(X, gamma)
        #transfer.probabilisticMerge(X,gamma)

    print(X.getIdeas())
simulate(0.1, 1000)
