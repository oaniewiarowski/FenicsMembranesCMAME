#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:14:02 2020

@author: alexanderniewiarowski
"""

import sympy as sp
from dolfin import *
try:
    from dolfin_adjoint import Expression, Constant
    ADJOINT = True
except ModuleNotFoundError:
    ADJOINT = False

DEGREE = 3


def ccode(z):
    return sp.printing.ccode(z)


class VariableGeometry(object):

    def build_derivative(self, diff_args=None, name=None):
        return Expression(tuple([ccode(i.diff(*diff_args)) for i in self.gamma_sp]),
                          eta=self.eta,
                          w=self.w,
                          t=self.t,
                          aw=self.aw,
                          ae=self.ae,
                          degree=DEGREE,
                          name=name)

    def __init__(self, w=None, eta=None, length=1, aw=0, ae=0):
        self.w = Constant(w, name="width")  # initial w to be varied along length
        self.eta = Constant(eta, name="eta")  # initial eta to be varied along length
        self.t = Constant(length, name="length")
        self.aw = Constant(aw, name="aw")  # amplitude of w variation, pos or neg
        self.ae = Constant(ae, name="ae")  # amplitude of eta variation, pos or neg

        ## fixed-end open surface volume correction - ONLY VALID FOR FIXED EDGES
        # Radius and Area for 3D vol correction
        R = self.w/sqrt(2*(1 - cos(self.eta)))
        Area = 0.5*R**2*(self.eta - sin(self.eta))
        self.vol_correct = Area*self.t/3
        ##

        W, eta, amp_w, amp_eta, t, x1, x2 = sp.symbols('w, eta, aw, ae, t,  x[0], x[1]')
        W_func = W + amp_w*sp.sin(pi*x2)  # vary W from xi2 = [0 --> 1]
        eta_func = eta + amp_eta*sp.sin(pi*x2)  # vary eta from xi2 = [0 --> 1]

        A = -0.5*W_func
        B = 0.5*W_func * (sp.sin(eta_func)/(1-sp.cos(eta_func)))
        C = 0.5*W  # last term modified to keep symmetry
        D = 0.5*W_func * (sp.sin(eta_func)/(1-sp.cos(eta_func)))

        X = A*sp.cos(x1*eta_func) + B*sp.sin(x1*eta_func) + C
        Z = -(A*sp.sin(x1*eta_func) - B*sp.cos(x1*eta_func) + D)

        self.gamma_sp = gamma_sp = [X, t*x2, Z]
        self.gamma = Expression(tuple([ccode(i) for i in gamma_sp]),
                                eta=self.eta,
                                w=self.w,
                                t=self.t,
                                aw=self.aw,
                                ae=self.ae,
                                degree=DEGREE,
                                name="gamma")

        self.Gsub1 = self.build_derivative([x1], name='Gsub1')
        self.Gsub2 = self.build_derivative([x2], name='Gsub2')

        if ADJOINT:
            # adjoint gamma
            self.gamma.dependencies = [self.w, self.eta, self.t, self.aw, self.ae]
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
            self.gamma.user_defined_derivatives[self.aw] = self.build_derivative(
                    [amp_w],
                    name='d_gamma_d_a_w')
            self.gamma.user_defined_derivatives[self.ae] = self.build_derivative(
                    [amp_eta],
                    name='d_gamma_d_a_e')

            # adjoint Gsub1
            self.Gsub1.dependencies = [self.w, self.eta, self.t, self.aw, self.ae]
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
            self.Gsub1.user_defined_derivatives[self.aw] = self.build_derivative(
                    [x1, amp_w],
                    name='d_Gsub1_d_a_w')
            self.Gsub1.user_defined_derivatives[self.ae] = self.build_derivative(
                    [x1, amp_eta],
                    name='d_Gsub1_d_a_eta')

            # adjoint Gsub2
            self.Gsub2.dependencies = [self.w, self.eta, self.t, self.aw, self.ae]
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
            self.Gsub2.user_defined_derivatives[self.aw] = self.build_derivative(
                    [x2, amp_w],
                    name='d_Gsub2_d_a_w')
            self.Gsub2.user_defined_derivatives[self.ae] = self.build_derivative(
                    [x2, amp_eta],
                    name='d_Gsub2_d_a_eta')
