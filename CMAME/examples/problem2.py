#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:40:46 2020
Modified on Thu Jun 18 09:06:49 2020

@author: alexanderniewiarowski
"""
from fenicsmembranes.parametric_membrane import *
from fenicsmembranes.variablegeometry import VariableGeometry as Cylinder
from fenicsmembranes.hydrostatic import ConstantHead
from dolfin_adjoint import *
from matplotlib import pyplot as plt
import argparse
import numpy as np
import sympy as sp
import pandas as pd
import json


class OptProblem(object):
    def __init__(self, J, controls, constraints=None):

        self.input_dict = {
                'controls': [],
                'bounds': [],
                'name': [],
                }
        self.controls = controls
        for ctrl in controls:
            self.add_control(*ctrl)

        self.m = [Control(c) for c in self.input_dict['controls']]
        self.m_iterates = [[] for x in range(len(controls))]
        self.j_iterates = []
        self.J = J
        self.constraint = constraints

    def eval_cb(self, j, m):
        print('*'*50)
        print('Current w: ', m[0].values()[0])
        print('Current eta: ', m[1].values()[0])
        print('Current j: ', j)
        print('*'*50)

    def derivative_cb(self, j, dj, m):
        self.j_iterates.append(j)
        self.j = j
        print(m)
        for i in range(len(m)):
            self.m_iterates[i].append(m[i].values()[0])

    def add_control(self, var, bounds, name=None):
        if name is None:
            name = var.name()
        ipt = self.input_dict
        ipt['controls'].append(var)
        ipt['bounds'].append(bounds)
        ipt['name'].append(name)

    def solve(self, J):
        Jhat = ReducedFunctional(self.J,
                                 self.m,
                                 eval_cb_post=self.eval_cb,
                                 derivative_cb_post=self.derivative_cb)
        problem = MinimizationProblem(Jhat,
                                      bounds=self.input_dict['bounds'],
                                      constraints=self.constraint)
        parameters = {"acceptable_tol": 1.0e-6, "maximum_iterations": 50}
        solver = IPOPTSolver(problem, parameters=parameters)
        self.opt_vals = solver.solve()


class EtaConstraint(InequalityConstraint):

    def __init__(self, H, bound=None):
        self.H = float(H)

        # we have g(a) > 0
        # for lower bound, Cz + R -H > 0 so is ok
        if bound == 'lower':
            self.bound = 1
        # for upper bound we have Cz + R - H < 0 so we mult by -1 to flip ineq
        elif bound == 'upper':
            self.bound = -1

    def function(self, m):
        W = m[0]
        eta = m[1]
        c_z = -0.5 * W * (sin(eta)/(1-cos(eta)))
        R = W/sqrt(2*(1 - cos(eta)))
        return [self.bound*(c_z + R - self.H)]

    def jacobian(self, m):
        W = variable(m[0])
        eta = variable(m[1])
        c_z = -0.5 * W * (sin(eta)/(1-cos(eta)))
        R = W/sqrt(2*(1 - cos(eta)))
        j = [0 for x in range(len(m))]
        j[0] = diff(self.bound*(c_z + R), W)
        j[1] = diff(self.bound*(c_z + R), eta)
        return j

    def output_workspace(self):
        return [0.0]

    def length(self):
        """
        Return the number of components in the constraint vector (here, one).
        """
        return 1


class MidPtConstraint(InequalityConstraint):

    def __init__(self, H, bound=None):
        self.H = float(H)
        # we have g(a) > 0
        # Cz + R - H > 0 at midsection
        if bound == 'lower':
            self.bound = 1
        # for upper bound we have Cz + R - H < 0 so we mult by -1 to flip ineq
        elif bound == 'upper':
            self.bound = -1

    def function(self, m):
        # w at midpoint
        W = m[0] + m[2]
        eta = m[1] + m[3]
        c_z = -0.5 * W * (sin(eta)/(1-cos(eta)))
        R = W/sqrt(2*(1 - cos(eta)))
        return [self.bound*(c_z + R - self.H)]

    def jacobian(self, m):
        W = variable(m[0])
        eta = variable(m[1])
        a_w = variable(m[2])
        a_eta = variable(m[3])

        c_z = -0.5 * (W + a_w) * (sin(eta + a_eta)/(1-cos(eta + a_eta)))
        R = (W + a_w)/sqrt(2*(1 - cos(eta + a_eta)))
        j = [0 for x in range(len(m))]
        j[0] = diff(self.bound*(c_z + R), W)
        j[1] = diff(self.bound*(c_z + R), eta)
        j[2] = diff(self.bound*(c_z + R), a_w)
        j[3] = diff(self.bound*(c_z + R), a_eta)
        return j

    def output_workspace(self):
        return [0.0]

    def length(self):
        """
        Return the number of components in the constraint vector (here, one).
        """
        return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Shape opt problem A and B')
    parser.add_argument('--resx', type=int, required=True)
    parser.add_argument('--resy', type=int, required=True)
    parser.add_argument('--bcs', type=str, required=True)
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    
    # Options: Run the optimization? Calculate and write dJdm? Taylor test?
    parser.add_argument('--no-optimize', dest='opt', action='store_false', default=True)
    parser.add_argument('--deriv', dest='deriv', action='store_true', default=False)
    parser.add_argument('--tt', dest='tt', action='store_true', default=False)
    args = parser.parse_args()

    OPTIMIZE = args.opt
    DERIVATIVE_STUDY = args.deriv
    TAYLOR_TEST = args.tt

    H = 1  # depth
    RESX = args.resx
    RESY = args.resy
    BCS = args.bcs

    # Initial guess for both load cases
    w = 3.5
    eta = 3
    aw = 0
    ae = 0

    p = args.p
    w_bounds = (0.25, 4)
    eta_bounds = (1, 6.2)

    # Only for case B (general plane strain)
    aw_bounds = (-0.2, 0.2)
    ae_bounds = (-0.2, 0.2)

    # Inequality contraints
    min_eta = EtaConstraint(H, bound='lower')
    max_eta = EtaConstraint(2*H, bound='upper')
    midpt_min = MidPtConstraint(H, bound='lower')  # only for case B
    midpt_max = MidPtConstraint(2*H, bound='upper')  # only for case B

    input_dict = {
            "resolution": [RESX, RESY],
            "thickness": 0.12,
            "pressure": 18.5,
            "material": 'Incompressible NeoHookean',
            "mu": 10*116,
            "Boundary Conditions": BCS,
            "cylindrical": True,
            "solver":
                    {"solver": "Naive",
                     "steps": 5,
                     "scale load": False}}

    fname = args.path

    # For hydrostatic loads on one and both sides
    for SURROUND in [False, True]:

        if BCS == 'Pinned':  # plane strain case with two controls
            length = 1
            geo_initial = Cylinder(w=w, eta=eta, aw=aw, ae=ae, length=length)
            constraints = [min_eta, max_eta]
            controls = [[geo_initial.w, w_bounds],
                        [geo_initial.eta, eta_bounds]]
        elif BCS == 'Capped':  # general plane stress with four controls
            length = 5
            geo_initial = Cylinder(w=w, eta=eta, aw=aw, ae=ae, length=length)
            constraints = [min_eta, max_eta, midpt_min, midpt_max]
            controls = [[geo_initial.w, w_bounds],
                        [geo_initial.eta, eta_bounds],
                        [geo_initial.aw, aw_bounds],
                        [geo_initial.ae, ae_bounds]]

        name_str = 'SURROUND' if SURROUND else 'NO_SURROUND'

        # update the input dictionary
        input_dict["geometry"] = geo_initial
        input_dict["output_file_path"] = fname+name_str+'_initial'

        membrane = ParametricMembrane(input_dict)
        forcing = ConstantHead(membrane,
                               rho=10,
                               g=10,
                               depth=1,
                               surround=SURROUND)
        membrane.solve(forcing, output=True)
        from utils.natlog_overloaded import natlog
        base = assemble(exp(membrane.lambda1*p)*dx(membrane.mesh))
        J = (1/p)*natlog(base)
        Jhat = ReducedFunctional(J, [Control(item[0]) for item in controls])
        
        if TAYLOR_TEST:
#            if p==300:
#                tt_results = taylor_to_dict(Jhat, [item[0] for item in controls],
#                                        [Constant(0.01) for x in range(len(controls))])
#
#            else:
            tt_results = taylor_test(Jhat, [item[0] for item in controls],
                                        [Constant(0.01) for x in range(len(controls))])
            tt_results = {'min_res': tt_results}
            with open(fname+name_str+'_taylor_test.json', 'w') as fp:
                json.dump(tt_results, fp, indent=4)
#            conv_rate = taylor_test(Jhat,
#                                    [item[0] for item in controls],
#                                    [Constant(0.01) for x in range(len(controls))])
#            print(conv_rate)
#            assert(conv_rate > 1.9)

        if DERIVATIVE_STUDY:
            # calc and store J and dJdm along min height constraint
            ws = []  # list of w points
            etah1 = []  # list of etas along H=1 (min height constraint)

            def get_eta(W, H):
                '''
                W = width
                H = minimum height ie depth
                '''
                assert W > 0.001, "does not work for small W"
                eta = sp.symbols('eta', negative=True)
                z_c = 0.5 * W * (sp.sin(eta)/(1-sp.cos(eta)))
                R = W/sp.sqrt(2*(1 - sp.cos(eta)))
                # tried many different settings, this seems to work for wide range of W
                e = -sp.nsolve(z_c + R - H, -6.2, bisect=True)  # - b.c want +ve eta
                return e

            for w in np.linspace(w_bounds[0], w_bounds[1], 100):
                try:
                    e1 = get_eta(w, 1)
                    ws.append(float(w))
                    etah1.append(float(e1))
                except ValueError:
                    continue

            df = pd.DataFrame({'w': [],
                               'eta': [],
                               'J': [],
                               'dJ': []})
            for i, w_ in enumerate(ws):
                df.at[i, 'w'] = ws[i]
                df.at[i, 'eta'] = etah1[i]
                x = [Constant(w_), Constant(etah1[i])]
                df.at[i, 'J'] = Jhat(x)
                djdm = Jhat.derivative()
                df.at[i, 'dJdw'] = djdm[0].values()[0]
                df.at[i, 'dJdeta'] = djdm[1].values()[0]
            df.to_json(fname+name_str+'_J_dJdm.json')

        if OPTIMIZE:
            # Set up the problem and solve
            problem = OptProblem(J, controls, constraints)
            problem.solve(J)
            opt_vals = problem.opt_vals

            # Save solution to json  # Fix these 
            df = pd.DataFrame()
            df['w'] = pd.Series(problem.m_iterates[0])
            df['eta'] = pd.Series(problem.m_iterates[1])
            df['J'] = pd.Series(problem.j_iterates)
            df['w_opt'] = opt_vals[0].values()[0]
            df['eta_opt'] = opt_vals[1].values()[0]
            if BCS == 'Capped':
                df['a_w'] = pd.Series(problem.m_iterates[2])
                df['a_e'] = pd.Series(problem.m_iterates[3])
                df['a_w_opt'] = opt_vals[2].values()[0]
                df['a_e_opt'] = opt_vals[3].values()[0]
            df.to_json(fname+name_str+'_iterates.json')

            # Final problem
            if BCS == 'Pinned':
                geo_final = Cylinder(w=opt_vals[0],
                                           eta=opt_vals[1],
                                           aw=0,
                                           ae=0,
                                           length=length)
            else:
                geo_final = Cylinder(w=opt_vals[0],
                                           eta=opt_vals[1],
                                           aw=opt_vals[2],
                                           ae=opt_vals[3],
                                           length=length)

            # update the input dictionary with final solution
            input_dict["geometry"] = geo_final
            input_dict["output_file_path"] = fname+name_str+'_final'

            # Time adjoint
            adj_timer = Timer("Adjoint run")
            J_ = ReducedFunctional(J, [Control(item[0]) for item in controls])
            J_.derivative(options={"riesz_representation": "L2"})
            adj_time = adj_timer.stop()
            del J_

            # Run & time compiled forward solve
            fwd_timer = Timer("Forward run")
            membrane = ParametricMembrane(input_dict)
            forcing = ConstantHead(membrane,
                                   rho=10,
                                   g=10,
                                   depth=1,
                                   surround=SURROUND)
            membrane.solve(forcing, output=True)
            base = assemble(exp(membrane.lambda1*p)*dx(membrane.mesh))
            J = (1/p)*natlog(base)
            fwd_time = fwd_timer.stop()

            # Store true max lambda1 corresponding to found minimizer
            lambda1_max_check = max(project(membrane.lambda1, membrane.Vs).vector()[:])

            # Write log file
            log_dict = {"fname": fname+name_str,
                        "optimization": {
                             "p KS": p,
                             "num_controls": len(controls),
                             "bounds": {
                                     'w': w_bounds,
                                     'eta': eta_bounds,
                                     'aw': aw_bounds,
                                     'ae': ae_bounds},
                             "fwd_time": fwd_time,
                             "adj_time": adj_time,
                             "efficiency": (fwd_time+adj_time)/fwd_time,
                             "lambda1_max_check": lambda1_max_check},
                        "membrane": {
                                "resolution": input_dict["resolution"],
                                "BCs": input_dict["Boundary Conditions"],
                                "pressure": input_dict["pressure"],
                                "thickness": input_dict["thickness"],
                                "material": input_dict["material"],
                                "mu": input_dict["mu"],
                                "solver": input_dict["solver"]},
                        "initial geometry": {
                                "w": w,
                                "eta": eta,
                                "aw": aw,
                                "ae": ae},
                        'standard_log': {
                             'num_dofs': membrane.V.dim(),
                             'num_cells': membrane.mesh.num_cells(),
                             'ufl_cell': str(membrane.V.ufl_cell()),
                             'V_degree': membrane.V.ufl_element().degree(),
                             'Vs_degree': membrane.Vs.ufl_element().degree()}
                        }
            with open(fname+name_str+'_log.json', 'w') as fp:
                json.dump(log_dict, fp, indent=4)
