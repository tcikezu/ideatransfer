""" simulate.py
-----------------------
This file creates a community object, defined in test.py, and runs to see how ideas
evolve over some number of iterations."""


from test import * 

def simulate(gamma, T):
    X = community(20)
    print(X.getAllIdeas())

    for t in range(T):
        for i in range(X.N):
            for j in range(X.N):
                if (i != j): 
                    if (X.agreement(i,j) >= X.getThreshold(i)):
                        X.merge(i, gamma, j)
                    else:
                        X.merge(i, -gamma, j)
    print(X.getAllIdeas())
simulate(0.01, 100)
