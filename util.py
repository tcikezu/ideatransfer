import sys
import collections


""" Utility class with a bunch of random stuff that may be useful."""

def dot(v1, v2):
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def euclideanDistance(v1, v2):
    # make sure that v1 and v2 are same dimension
    diff = [v1[i] - v2[i] for i in range(len(v1))] 
    return dot(diff, diff)

def sparseVectorDotProduct(v1, v2):
    return sum([v2[k]*v for k, v in v1.items()])

def incrementSparseVector(v1, scale, v2):
    v3 = v1
    for k, v in v2.items(): # Don't want to change v1 or v2
        v3[k] += scale*v
    return v3

def sparseVectorDistance(v1, v2):
    d = incrementSparseVector(v1, -1, v2)
    return sparseVectorDotProduct(d, d)**0.5

def sparseVectorNormalize(v1):
    m = sparseVectorDotProduct(v1, v1)**0.5
    for k,v in v1.items():
        v1[k] = v1[k] / m
        
