import numpy as np
from typing import List
from collections import set
from math import prod

"""
Program inspired by the paper:
==============================
"A Generalization of the Noisy-Or Model"
By Sampath Srinivas published in 1993

F is a discrete function that maps the space
of joint states of model outputs into the set
of states X.

F degenerates to boolean noisy or when the inputs
and outputs are boolean:

F(u') = x([(m_x - 1)*((1/n)*sum((I'(u'))/(m - 1)))]

this is the F described in the paper
"""

def BoolOR(probs):
    return 1 - prod(probs)

def NoisyOr(probs: List[float], X: List[str]):
    # want to map from our U' probs to X categories
    s = set()
    X = zip(range(len(X)), X)

print(BoolOR([0.99, 0.91, 0.98]))
