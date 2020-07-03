#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:15:53 2020

@author: alexanderniewiarowski

Pressure optimization problem
"""

from fenicsmembranes.parametric_membrane import *
from fenicsmembranes.variablegeometry import VariableGeometry as Cylinder
from fenicsmembranes.geometry import AdjointGeoWEta as WETA
from fenicsmembranes.hydrostatic import ConstantHead, VariableHead
from dolfin_adjoint import *
from matplotlib import pyplot as plt
from utils.natlog_overloaded import natlog
#parameters["form_compiler"]["representation"] = "uflacs"  # supposedly needed fore nested conditionals in variable head 
import numpy as np
import pandas as pd
import json
import argparse

class OptProblem(object):
    def __init__(self, J, controls, constraints=None, name=None, params=None):

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
        self.name = name
        self.J = J
        self.constraint = constraints
        
        self.params = params

    def eval_cb(self, j, m):
#        print('*'*50)
        pass

    def derivative_cb(self, j, dj, m):
        self.j_iterates.append(j)
        self.j = j
        for i in range(len(m)):
            self.m_iterates[i].append(m[i].values()[0])

    def add_control(self, var, bounds, name=None):
        if name == None:
            name = var.name()
        ipt = self.input_dict
        ipt['controls'].append(var)
        ipt['bounds'].append(bounds)
        ipt['name'].append(name)

    def plot_iterates(self, fname):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_title(self.name)

        ln1 = ax1.plot(self.j_iterates, label="obj. func.", c='g')
        ax1.set_xlabel("Iterations");  ax1.set_ylabel('J')

        ln2 = ax2.plot(self.m_iterates[0], '*', label="control")
        ax2.set_ylabel(self.input_dict['name'])
        # put labels on one legend
        lns = ln1+ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)

        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def solve(self, J):
        Jhat = ReducedFunctional(self.J,
                                 self.m,
                                 eval_cb_post=self.eval_cb,
                                 derivative_cb_post=self.derivative_cb)
        problem = MinimizationProblem(Jhat,
                                      bounds=self.input_dict['bounds'],
                                      constraints=self.constraint)
        solver = IPOPTSolver(problem, parameters=self.params)
        opt_vals = solver.solve()
        self.plot_iterates(f'{self.name}_iterates.pdf')
        self.opt_vals = opt_vals
        return opt_vals


class FunctionalConstraint(InequalityConstraint):
    def __init__(self, Jhat, constant):
        self.Jhat = Jhat
        self.c = Constant(constant)
        # we have g(a) > c

    def function(self, m):
        return [self.Jhat(Constant(m[0])) - self.c]

    def jacobian(self, m):
        return [self.Jhat.derivative()]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """
        Return the number of components in the constraint vector (here, one).
        """
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, required=True)

    parser.add_argument('--resx', type=int, required=True)
    parser.add_argument('--resy', type=int, required=False)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--diag', type=str, default="crossed")
    parser.add_argument('--bcs', type=str, default='Pinned')
    parser.add_argument('--pressure', type=float, default=18.5)

    parser.add_argument('--height', type=float, default=1)
    parser.add_argument('--surround', dest='surround', action='store_true')
    parser.add_argument('--no-surround', dest='surround', action='store_false')
    parser.add_argument('--forcing', type=str, required=True)  # constant or variable

    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--itr', type=int, required=True)
    
    # Options: Run the optimization? Calculate and write dJdm? Taylor test?
    parser.add_argument('--no-optimize', dest='opt', action='store_false', default=True)
    parser.add_argument('--deriv', dest='deriv', action='store_true', default=False)
    parser.add_argument('--tt', dest='tt', action='store_true', default=False)
    args = parser.parse_args()
    
    OPTIMIZE = args.opt
    DERIVATIVE_STUDY = args.deriv
    TAYLOR_TEST = args.tt

    DIM = args.dim
    MAX_ITER = args.itr
    H = args.height
    diag = args.diag
    RESX = args.resx
    RESY = args.resy
    BCS = args.bcs
    p = args.p
    p_0 = args.pressure
    p_bounds = (2, 30) if BCS == 'Pinned' else (20,100)
    w = 2
    eta = pi
    SURROUND = args.surround
    forcing_type = args.forcing

    solver_params = {"acceptable_tol": 1.0e-8,
                     "maximum_iterations": MAX_ITER}

    if DIM == 2:
        geo = WETA(w=w, eta=-eta, dim=DIM)
        z_vec = Constant(('0', '1'), name="z_vec")
        res = [RESX]
    else:
        length = 1 if BCS=='Pinned' else 10
        geo = Cylinder(w=w, eta=eta, aw=0, ae=0, length=length)
        z_vec = Constant(('0', '0', '1'), name="z_vec")
        res = [RESX, RESY]


    fname = args.path

    # initial prob
    input_dict = {
        "resolution": res,
        "mesh diagonal": diag,
        "geometry": geo,
        "thickness":
                {"value": '0.12',
                 "type": 'Constant'},
        "pressure": p_0,
        "material": 'Incompressible NeoHookean',
        "mu": 10*116,
        "output_file_path": fname + '_initial',
        "Boundary Conditions": BCS,
        "cylindrical": True,
        "solver":
                {"solver": "Naive",
                 "steps": 5,
                 "scale load": False}}

    membrane = ParametricMembrane(input_dict)
#        vol0 = project(membrane.vol_0,FunctionSpace(membrane.mesh, "R", 0)).vector()[0] 
#        vol_correct =  project(membrane.vol_correct,FunctionSpace(membrane.mesh, "R", 0)).vector()[0]
    if forcing_type == 'constant':
        forcing = ConstantHead(membrane, rho=10, g=10, depth=H, surround=SURROUND)
    elif forcing_type == 'var_head':
        forcing = VariableHead(membrane, rho=10, g=10, depth_U=H, depth_D=.25*H)
    membrane.solve(forcing, output=True)      

    # calc max z using KS func
    position = membrane.get_position()
    base = assemble(exp(p*dot(position, z_vec))*dx(membrane.mesh))
    max_z = (1/p)*natlog(base)
    print(max_z)

    # calc min lambda
    l_target = membrane.lambda1 if DIM == 2 else membrane.lambda2
    neg_min_lambda = assemble(exp(-p*l_target)*dx(membrane.mesh))  # return the largest neg value = the smallest pos
    min_lambda = (-1./p)*natlog(neg_min_lambda)
    print(min_lambda)

    # the objective
    J = assemble(membrane.lambda1*dx(membrane.mesh))
    print(f'J: {J}')

    controls = [[membrane.p_0, p_bounds]]

    #%%
    tags = ['J', 'max_z', 'min_lambda']
    Jhat = ReducedFunctional(J, Control(membrane.p_0))
    Jhat_height = ReducedFunctional(max_z, Control(membrane.p_0))
    Jhat_min_lambda = ReducedFunctional(min_lambda, Control(membrane.p_0))
    
    if DERIVATIVE_STUDY:
        df = pd.DataFrame({'p': [],
                           'J': [],
                           'J_h': [],
                           'J_l': [],
                           'dJ': [],
                           'dJ_h': [],
                           'dJ_l': []})
        for i, p_ in enumerate(np.linspace(p_bounds[0], p_bounds[1], 200)):
            print('*'*100)
            print(i/200)
            df.at[i, 'p'] = p_
            p_ = Constant(p_)
            df.at[i, 'J'] = Jhat(p_)
            df.at[i, 'J_h'] = Jhat_height(p_)
            df.at[i, 'J_l'] = Jhat_min_lambda(p_)
            df.at[i, 'dJ'] = Jhat.derivative()
            df.at[i, 'dJ_h'] = Jhat_height.derivative()
            df.at[i, 'dJ_l'] = Jhat_min_lambda.derivative()
        df.to_json(fname+'J_dJdm.json')


    if TAYLOR_TEST:
        results = {}
        h = Constant(0.5)  # the direction of the perturbation
        for i, j in enumerate([J, max_z, min_lambda]):
            print('******'*50)
            print(f'starting results for {tags[i]}')
            Jhat = ReducedFunctional(j, Control(membrane.p_0))
            conv_rate = taylor_test(Jhat, membrane.p_0, h)
#            print(conv_rate)
            results[tags[i]] = conv_rate #taylor_to_dict(Jhat, membrane.p_0, h) #hessian not working overflow error
        with open(fname+'taylor_test.json', 'w') as fp:
            json.dump(results, fp, indent=4)
        print('******'*50)
#            assert(conv_rate > 1.9)
    #%%
    if OPTIMIZE:
        problem = OptProblem(J, controls, constraints=None, name=fname, params=solver_params)
        Jhat_max_z = ReducedFunctional(max_z, problem.m[0])
        Jhat_min_lambda = ReducedFunctional(min_lambda, problem.m[0])
    
        # g - c>0
        height_constraint = FunctionalConstraint(Jhat_max_z, H)
        stretch_constraint = FunctionalConstraint(Jhat_min_lambda, 1)
        problem.constraint = [height_constraint, stretch_constraint]
        opt_vals = problem.solve(J)
    
        # final prob
        input_dict["pressure"] = opt_vals[0]
        input_dict["output_file_path"] = fname + '_final'
    
        # Run & time compiled forward solve
        fwd_timer = Timer("Forward run")
        membrane = ParametricMembrane(input_dict)
        if forcing_type == 'constant':
            forcing = ConstantHead(membrane, rho=10, g=10, depth=H, surround=SURROUND)
        elif forcing_type == 'var_head':
            forcing = VariableHead(membrane, rho=10, g=10, depth_U=H, depth_D=.25*H)
        membrane.solve(forcing, output=True)
        fwd_time = fwd_timer.stop()
        print(f'FINAL PRESSURE: {opt_vals[0].values()}')
    
        # Time adjoint (all three reduced functionals)
        adj_timer = Timer("Adjoint run")
        Jhat(opt_vals[0])
        Jhat.derivative(options={"riesz_representation": "L2"})
    
        Jhat_max_z(opt_vals[0])
        Jhat_max_z.derivative(options={"riesz_representation": "L2"})
    
        Jhat_min_lambda(opt_vals[0])
        Jhat_min_lambda.derivative(options={"riesz_representation": "L2"})
        adj_time = adj_timer.stop()
    
        max_z = Jhat_max_z(opt_vals[0])
        position = membrane.get_position()
        true_zmax = max(project(dot(position, z_vec),
                                membrane.Vs).vector()[:])
        print(f'FINAL HEIGHT KS: {max_z}, FINAL HEIGHT{true_zmax}')
    
        min_lambda = Jhat_min_lambda(opt_vals[0])
        l_target = membrane.lambda1 if DIM == 2 else membrane.lambda2
        true_lambda_min = min(project(l_target, membrane.Vs).vector()[:])
        print(f'MIN STRETCH: {min_lambda}')
    
        print(problem.m_iterates)
        iterates = problem.m_iterates[0]
        results_dict = {'HEIGHT_KS' : max_z,
                        'HEIGHT': true_zmax,
                        'pressure': opt_vals[0].values()[0],
                        'bounds' :p_bounds,
                        'lambda_min_KS' : min_lambda,
                        'true_lambda_min': true_lambda_min,
                        'p' : p,
                        'res': res,
                        'max_iter': MAX_ITER,
                        'j': problem.j_iterates,
                        'm': iterates
                        }
    
        with open(fname+'_conv_history.json', 'w') as fp:
            json.dump(results_dict, fp, indent=4)
    
        # Write log file
        log_dict = {"fname": fname,
                    "optimization": {
                         "p KS": p,
                         "dim": DIM,
                         "num_controls": len(controls),
                         "bounds": {
                                 'pressure': controls[0][1]},
                         "fwd_time": fwd_time,
                         "adj_time": adj_time,
                         "cost ratio": (fwd_time+adj_time)/fwd_time},
                    "membrane": {
                            "resolution": input_dict["resolution"],
                            "mesh diagonal": diag,
                            "BCs": input_dict["Boundary Conditions"],
                            "pressure": p_0,
                            "initial_thickness": input_dict["thickness"]["value"],
                            "material": input_dict["material"],
                            "mu": input_dict["mu"],
                            "solver": input_dict["solver"]},
                    "initial geometry": {
                             "w": w,
                             "eta": eta},
                    "forcing": {
                                'type': str(type(forcing)),
                                'H': H,
                                'SURROUND': SURROUND
                                },
                    'standard_log': {
                         'num_dofs': membrane.V.dim(),
                         'num_cells': membrane.mesh.num_cells(),
                         'ufl_cell': str(membrane.V.ufl_cell()),
                         'V_degree': membrane.V.ufl_element().degree(),
                         'Vs_degree': membrane.Vs.ufl_element().degree()}
                    }
    
        with open(fname+'log.json', 'w') as fp:
            json.dump(log_dict, fp, indent=4)
