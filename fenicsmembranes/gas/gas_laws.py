#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:23:17 2020

@author: alexanderniewiarowski
"""

from dolfin import *

try:
    from dolfin_adjoint import *
    ADJOINT = True
except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False
    import contextlib
    def stop_annotating():
        return contextlib.nullcontext()

from . import register_gas_law

@register_gas_law('Isentropic Gas')
class IsentropicGas():
    """
        p = p_0(V_0/V)^k

        Calculates and saves the gas constant by inflating membrane to p_0.

        Used for initialization or when internal air mass changes.

    """
    def __init__(self, membrane):
        self.ready = False
        self.mem = mem = membrane
        self.kappa = mem.kwargs.get("kappa", 1)
        self.p_0 = mem.p_0
        
    def setup(self):
        '''
        Called after inflation solve!
        '''
        mem = self.mem

        if mem.has_free_surface:
            mem.free_surface.setup()
            if mem.nsd == 2:
                x_H = Constant((0, mem.x_H))
            elif mem.nsd == 3:
                x_H = Constant((0, 0, mem.x_H))
                
            self.V_0 = assemble((1/mem.nsd)*\
                                (dot(mem.x_g, (1 - mem.c)*mem.gsub3) \
                                 - dot(x_H, mem.free_surface.n0))*dx(mem.mesh, degree=5))
        else:
            self.V_0 = mem.calculate_volume(mem.u)
            
        # TODO: why doesn't dolfin adjoint like the Constant? try AdjFloat?
        if ADJOINT:
            self.constant = self.p_0*self.V_0**self.kappa
        else:
            self.constant = Constant(self.p_0*self.V_0**self.kappa, name='gas_constant')

        self.ready = True

    def update_pressure(self):
        mem = self.mem
        
        if mem.has_free_surface:
            if mem.nsd ==2:
                x_H = Constant((0, mem.x_H))
            elif mem.nsd ==3:
                x_H = Constant((0, 0, mem.x_H))

            self.V = assemble((1/mem.nsd)*\
                              dot(mem.x_g - (1 - mem.c)*x_H, mem.gsub3)*dx(mem.mesh, degree=5))
#            assemble((1/mem.nsd)*dot((1 - mem.c)*(mem.get_position() - x_H), mem.gsub3)*dx(mem.mesh, degree=5))
            mem.free_surface.update()

        else:
            self.V = mem.calculate_volume(mem.u)
        
        self.p = self.constant/(self.V**self.kappa)

        # update dpDV 
#        self.dpdV = -self.kappa*self.constant/(self.V**(self.kappa + 1))
        self.dpdV = -self.kappa*self.p/self.V
        if ADJOINT:
            return self.p
        else:
            return Constant(self.p)


@register_gas_law('Boyle')
class Boyle():
    """
    Calculates and saves the Boyle constant by inflating membrane to p_0.

    Used for initialization or when internal air mass changes.
    """
    def __init__(self, membrane, **kwargs):
        self.mem = mem = membrane
        self.p_0 = mem.p_0

    def setup(self):
        mem = self.mem
        self.V_0 = mem.calculate_volume(mem.u)

        if ADJOINT:
            self.boyle = self.V_0*self.p_0
        else:
            self.boyle = Constant(self.V_0*self.p_0, name='Boyle_constant')

    def update_pressure(self):
        mem = self.mem
        self.V = mem.calculate_volume(mem.u)
        self.p = self.boyle/self.V
        self.dpdV = -self.boyle/(self.V**2)
#        self.dpdV = -(self.constant)/(V**2)  # boyle
