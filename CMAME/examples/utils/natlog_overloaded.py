from fenics import *
from fenics_adjoint import *

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function
import numpy as np

def natlog(func):
    return ln(func)

backend_natlog = natlog

class NatLogBlock(Block):
    def __init__(self, func, **kwargs):
        super(NatLogBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)

    def __str__(self):
        return 'NatLogBlock'

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        x = inputs[idx]
        derivative = 1.0 / x
        return derivative * adj_input 

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_natlog(inputs[0])
    
    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
#        return derivative_example_function(inputs[0], tlm_inputs[0])
        return tlm_inputs[0]/inputs[0]
    
    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                   relevant_dependencies, prepared=None):
        return -hessian_inputs[0]/inputs[0]**2

natlog = overload_function(natlog, NatLogBlock)
