#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:16:49 2019

@author: alexanderniewiarowski


This module contains various classes and functions to generate a parametric 
geometry based on circular arches.

"""
import sympy as sp
import numpy as np
from dolfin import *

try:
    from dolfin_adjoint import Expression, Constant
    ADJOINT = True
except ModuleNotFoundError:
    ADJOINT = False

DEGREE = 3


def ccode(z):
    return sp.printing.ccode(z)


def ligaro(W, eta):
    x1 = sp.symbols('x[0]')
    A = -0.5*W
    B = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))
    C = 0.5*W
    D = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))

    X = A*sp.cos(x1*eta) + B*sp.sin(x1*eta) + C
    Z = A*sp.sin(x1*eta) - B*sp.cos(x1*eta) + D
    return X, Z


class AdjointGeoWEta(object):

    def build_derivative(self, diff_args=None, name=None):
        return Expression(tuple([ccode(i.diff(*diff_args)) for i in self.gamma_sp]),
                          eta=self.eta,
                          w=self.w,
                          t=self.t,
                          degree=DEGREE,
                          name=name)

    def __init__(self, w=None, eta=None, dim=None, length=1):
        self.w = Constant(w, name="width")
        self.eta = Constant(eta, name="eta")
        self.t = Constant(length, name="length")

        W, eta, x1, x2, t = sp.symbols('w, eta, x[0], x[1], t')
        X, Z = ligaro(W, eta)

        if dim == 2:
            self.Gsub2 = None
            self.gamma_sp = gamma_sp = [X, Z]
            self.gamma = Expression(tuple([ccode(i) for i in gamma_sp]),
                                    eta=self.eta,
                                    w=self.w,
                                    degree=DEGREE,
                                    name="gamma")

        if dim == 3:
            self.gamma_sp = gamma_sp = [X, t*x2, Z]
            # fixed-end open surface volume correction
            # ONLY VALID FOR FIXED EDGES
            R = self.w/sqrt(2*(1 - cos(self.eta)))
            Area = -0.5*R**2*(self.eta - sin(self.eta))  # negative because upsidedown #TODO fix
            self.vol_correct = (Area*self.t/3)

            self.gamma = Expression(tuple([ccode(i) for i in gamma_sp]),
                                    eta=self.eta,
                                    w=self.w,
                                    t=self.t,
                                    degree=DEGREE,
                                    name="gamma")

        self.Gsub1 = self.build_derivative([x1], name='Gsub1')
        if dim == 3:
            self.Gsub2 = self.build_derivative([x2], name='Gsub2')

        if ADJOINT:
            # adjoint gamma
            self.gamma.dependencies = [self.w, self.eta, self.t]
            self.gamma.user_defined_derivatives = {}
            self.gamma.user_defined_derivatives[self.w] = self.build_derivative(
                    [W],
                    name='d_gamma_dw')
            self.gamma.user_defined_derivatives[self.eta] = self.build_derivative(
                    [eta],
                    name='d_gamma_deta')
            self.gamma.user_defined_derivatives[self.t] = self.build_derivative(
                    [t],
                    name='d_gamma_dt')

            # adjoint Gsub1
            self.Gsub1.dependencies = [self.w, self.eta, self.t]
            self.Gsub1.user_defined_derivatives = {}
            self.Gsub1.user_defined_derivatives[self.w] = self.build_derivative(
                    [x1, W],
                    name='d_Gsub1_dw')
            self.Gsub1.user_defined_derivatives[self.eta] = self.build_derivative(
                    [x1, eta],
                    name='d_Gsub1_deta')
            self.Gsub1.user_defined_derivatives[self.t] = self.build_derivative(
                    [x1, t],
                    name='d_Gsub1_dt')

            if dim == 3:
                # adjoint Gsub2
                self.Gsub2.dependencies = [self.w, self.eta, self.t]
                self.Gsub2.user_defined_derivatives = {}
                self.Gsub2.user_defined_derivatives[self.w] = self.build_derivative(
                        [x2, W],
                        name='d_Gsub2_dw')
                self.Gsub2.user_defined_derivatives[self.eta] = self.build_derivative(
                        [x2, eta],
                        name='d_Gsub2_deta')
                self.Gsub2.user_defined_derivatives[self.t] = self.build_derivative(
                        [x2, t],
                        name='d_Gsub2_dt')

