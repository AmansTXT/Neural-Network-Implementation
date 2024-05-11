import math
layer_outputs = [[4.8, 1.21, 2.385],
                 [4.8, 1.21, 2.385],
                 [4.8, 1.21, 2.385], ]


""" Raw Python Version """
print("!!! Raw Python Version !!!")
E = math.e

exp_values_vec = [[(E**(out- max(vec))) for out in vec] for vec in layer_outputs]
print("Exponential values:", exp_values_vec)

norm_bases = [sum(vec) for vec in exp_values_vec] 
norm_values_vec = [ [(x/sum(vec))  for x in vec] for vec in exp_values_vec]
print("Normalized Exp Values:", norm_values_vec)
print("Sum of Norm_values", [sum(norm_vec) for norm_vec in norm_values_vec])

print("|||||||||||||||||||||||||||||||||||||")

""" Numpy Version """
print("!!! Numpy Version !!!")
import numpy as np

exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1))
print(np.sum(layer_outputs, axis=1, keepdims=True))
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)