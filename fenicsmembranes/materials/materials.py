#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:28:14 2019

@author: alexanderniewiarowski
"""

from dolfin import * #Constant, det, tr, inner, inv, as_matrix, sqrt, dot,as_tensor, split,as_vector, outer,cos, sin, exp
import ufl
from . import Material, register_material

try:
    from dolfin_adjoint import * #Constant
    ADJOINT = True
except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False


@register_material('Incompressible NeoHookean')
class IncompressibleNeoHookean(Material):

    def __init__(self, membrane, kwargs):
        mem = membrane

        # Define material - incompressible Neo-Hookean
        self.nu = 0.5  # Poisson ratio, incompressible

        E = kwargs.get("E", None)
        if E is not None:
            self.E = Constant(E, name="Elastic modulus")

        mu = kwargs.get("mu", None)
        if mu is not None:
            self.mu = Constant(mu, name="Shear modulus")
        else:
            self.mu = Constant(self.E/(2*(1 + self.nu)), name="Shear modulus")

        if kwargs['cylindrical']:
            mem.I_C = tr(mem.C) + 1/det(mem.C) - 1

        else:
            C_n = mem.C_n  #mem.F_n.T*mem.F_n
            C_0 = mem.C_0  #mem.F_0.T*mem.F_0
            C_0_sup = mem.C_0_sup
            i,j = ufl.indices(2)
            I1 = C_0_sup[i,j]*C_n[i,j]

            mem.I_C = I1 + 1/det(mem.C)

        self.psi = 0.5*self.mu*(mem.I_C - Constant(mem.nsd))

