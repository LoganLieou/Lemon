import numpy as np

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

def scuffedNet(probs):
    X = ["brain_tumor", "mild dem", "moderate dem", "non dem", "very dem"]
    return X[np.argmax(probs)]

# assume this is some sort of network output
print(scuffedNet([0.1, 0.921, 0.991, 0.111, 0.114]))
