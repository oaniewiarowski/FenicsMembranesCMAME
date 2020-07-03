#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:12:50 2019

@author: alexanderniewiarowski
"""
from dolfin import *
import numpy as np

try:
    from dolfin_adjoint import *
    ADJOINT = True


except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False
    import contextlib
    def stop_annotating():
        return contextlib.suppress()


class ConstantHead(object):
    '''
    Hydrostatic (linearly varying) pressure field in the z direction.
    '''

    def __init__(self, membrane, rho=0, g=0, depth=0, surround=False):

        self.membrane = membrane
        self.rho = Constant(rho)
        self.g = Constant(g)
        self.depth = Constant(depth)
        if membrane.nsd==3:
            self.x_vec = Constant(('1', '0', '0'), name="x_vec")
            self.z_vec = Constant(('0', '0', '1'), name="z_vec")
        if membrane.nsd==2:
            self.x_vec = Constant(('1', '0'), name="x_vec")
            self.z_vec = Constant(('0', '1'), name="z_vec")
        self.surround = surround

    def update(self):

        mem = self.membrane

        position = mem.get_position()
        x = dot(position, self.x_vec)
        z = dot(position, self.z_vec)
        
        pressure = conditional(le(z, self.depth), self.rho*self.g*(self.depth - z), 0.0)

        # return if "underwater" -- this is water on both sides
        if self.surround:
            with stop_annotating():
                mem.p_ext.assign(project(pressure, mem.Vs))  # for visualization
            return pressure
        
        # else, water on one side only: p = p if n_x<0, else 0
        pressure = conditional(le(dot((mem.gsub3), self.x_vec), 0.0), pressure, 0)  # zero the pressure after the crown

        with stop_annotating():
            mem.p_ext.assign(project(pressure, mem.Vs))  # for visualization

        return pressure
    

class VariableHead(object):
    '''
    Hydrostatic (linearly varying) pressure field in the z direction.
    '''

    def __init__(self, membrane, rho=10, g=10, depth_U=.3, depth_D=.3):

        self.membrane = membrane
        self.rho = Constant(rho)
        self.g = Constant(g)
        self.depth_U = Constant(depth_U)
        self.depth_D = Constant(depth_D)
        
        if membrane.nsd==3:
            self.x_vec = Constant(('1', '0', '0'), name="x_vec")
            self.z_vec = Constant(('0', '0', '1'), name="z_vec")
        if membrane.nsd==2:
            self.x_vec = Constant(('1', '0'), name="x_vec")
            self.z_vec = Constant(('0', '1'), name="z_vec")

    def update(self):

        mem = self.membrane

        position = mem.get_position()
        x = dot(position, self.x_vec)
        z = dot(position, self.z_vec)


        x_vals = project(x, mem.Vs)
        z_vals = project(z, mem.Vs)

        # p = \rho g (H-z) if z<H, else 0
        
        pressure_U = conditional(le(z, self.depth_U), self.rho*self.g*(self.depth_U - z), 0.0)
        pressure_D = conditional(le(z, self.depth_D), self.rho*self.g*(self.depth_D - z), 0.0)
        pressure = conditional(le(dot((mem.gsub3), self.x_vec), 0.0), pressure_U, pressure_D)

        with stop_annotating():
            mem.p_ext.assign(project(pressure, mem.Vs))  # for visualization

        return pressure