""" util.py
-------------------
Implementations of some methods that are used in the rest of the files.
Note that this is the numpy implementation."""

import sys
import collections
import numpy as np

def cosine(v1, v2):
    return np.dot(v1,v2) / (v1.T.dot(v1)**0.5 * v2.T.dot(v2)**0.5)

def euclideanDistance(v1, v2):
    # make sure that v1 and v2 are same dimension and ndarrays
    return (np.sum(v1**2) + np.sum(v2**2) - 2*np.sum(v1*v2))**0.5

# Compute the distance matrix for positional data X, which is arranged as n x d.
def distanceMatrix(X):
    return (np.sum(X**2,1) + np.sum(X**2,1)[:,np.newaxis] - 2*X @ X.T)**0.5

# Compute the cosine matrix for idea data X, which is arranged as n x d.
# This is pretty analogous to the distance matrix. Product matrix sounds a bit
# confusing though. I'm unsure. Basically, element (i,j), is Xi* Xj / (||Xi|| ||Xj||)
def cosineMatrix(X):
    return X @ X.T / (np.sum(X**2, 1)**0.5 * np.sum(X**2, 1)[:, np.newaxis]**0.5)

def L2normalize(X):
    return np.sum(X**2,1)[:,np.newaxis]**-0.5 * X

def L1normalize(X):
    return 1/np.sum(np.abs(X),1)[:,np.newaxis] * X

def normalizeDistribution(X):
    X = X*(X > 0) + 0.001
    return X / np.sum(X,axis=2).reshape(X.shape[0], X.shape[1],1)

# Given an n x m x d matrix, n corresponding to number members,
# m corresponding to # ideas, 
# d corresponding to domain size,
# and where the matrix represents the joint probability distribution over all
# members and their ideas, return an n x m matrix, which samples from the joint distribution.
def resampleDistribution(X):
    normalizeDistribution(X)
    c = np.cumsum(X,axis=2)
    u = np.random.rand(X.shape[0], X.shape[1], X.shape[2])
    return (u<c).argmax(axis=2)
