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
In essence, this function is a weighted average
"""

def NoisyOr(probs):
    """
    In our case this will always output
    1 * pmax, because probs in range 0:1
    """
    return np.max(probs) * np.ceil(np.average(probs))

# assume this is some sort of network output
print(NoisyOr([0.1, 0.6, 0.2, 0.1]))
print(NoisyOr([0.33]))

