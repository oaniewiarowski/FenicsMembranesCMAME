#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:08:58 2020

@author: alexanderniewiarowski
"""
from fenicsmembranes.parametric_membrane import *
from fenicsmembranes.variablegeometry import VariableGeometry as CappedCylinder
from fenicsmembranes.hydrostatic import ConstantHead
from dolfin_adjoint import *
from matplotlib import pyplot as plt
import json
import argparse

class OptProblem(object):
    def __init__(self, J, controls, constraints=None, name=None, params=None):
        self.input_dict = {
                'controls': [],
                'bounds': [],
                'name': []}

        self.controls = controls
        for ctrl in controls:
            self.add_control(*ctrl)
        self.m = [Control(c) for c in self.input_dict['controls']]
        self.m_iterates = []
        self.j_iterates = []

        self.name = name
        self.J = J
        self.constraint = constraints

        self.params = params
        self.iteration = 0
        self.iter_viz = XDMFFile(f'{name}_viz_iterations.xdmf')
        self.iter_viz.parameters['flush_output'] = True
        self.iter_viz.parameters['rewrite_function_mesh'] = False
        self.iter_viz.parameters['functions_share_mesh'] = True
        self.write_freq = 1
        self.checkpt_freq = 3

    def eval_cb(self, j, m):
        # print("TEST CALLBACK")
        pass

    def derivative_cb(self, j, dj, m):
        self.j_iterates.append(j)
        self.j = j
        if self.iteration % self.write_freq == 0 or self.iteration == self.params['maximum_iterations']:
            with self.iter_viz as xdmf:
                m[0].rename("thickness", "thickness")
                xdmf.write(m[0], self.iteration)
        if self.iteration % self.checkpt_freq == 0:
                checkpt = XDMFFile(f'{self.name}/checkpoint/_{self.iteration}.xdmf')
                checkpt.write_checkpoint(m[0], "thickness", 0, XDMFFile.Encoding.HDF5, False)  # Not appending to file           
                checkpt.close() 
        self.iteration += 1

    def add_control(self, var, bounds, name=None):
        if name == None:
            name = var.name()
        ipt = self.input_dict
        ipt['controls'].append(var)
        ipt['bounds'].append(bounds)
        ipt['name'].append(name)

    def plot_iterates(self, fname):
        fig, ax1 = plt.subplots()
        ax1.set_title(self.name)
        ln1 = ax1.plot(self.j_iterates, label="obj. func.", c='g')
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel('J')

        # put labels on one legend
        lns = ln1
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
        self.opt_vals = solver.solve()
        self.plot_iterates(f'{self.name}_iterates.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thickness opt 3D')
    parser.add_argument('--resx', type=int, required=True)
    parser.add_argument('--resy', type=int, required=True)
    parser.add_argument('--bcs', type=str, required=True)
    parser.add_argument('--diag', type=str, required=True)
    parser.add_argument('--itr', type=int, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--tdeg', type=int, default=2)
    parser.add_argument('--element', type=str)

#    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--lmax', type=float, required=True)
    parser.add_argument('--path', type=str, required=True)
    
    # Options: Run the optimization? Taylor test?
    parser.add_argument('--no-optimize', dest='opt', action='store_false', default=True)
    parser.add_argument('--tt', dest='tt', action='store_true', default=False)
    args = parser.parse_args()

    OPTIMIZE = args.opt
    TAYLOR_TEST = args.tt

    w = 2
    eta = pi
    length = 4
    MAX_ITER = args.itr
    l_max = args.lmax
    BCS = args.bcs
    RESX = args.resx
    RESY = args.resy
    diag = args.diag
    method = args.method
#    p = args.p
    solver_params = {"acceptable_tol": 1.0e-8,
                     "maximum_iterations": MAX_ITER}

    geo = CappedCylinder(w=w, eta=eta, length=length)
    fname = args.path

    # initial prob
    input_dict = {
        "resolution": [RESX, RESY],
        "geometry": geo,
        "mesh diagonal": diag,
        "pressure": 18.5,
        "material": 'Incompressible NeoHookean',
        "mu": 10*116,
        "Boundary Conditions": BCS,
        "cylindrical": True,
        "solver": "Naive",
        "load steps": 5}

    restart_from_checkpoint = False
    if restart_from_checkpoint:
        mesh = UnitSquareMesh(RESX, RESY, diag)
        Vs = FunctionSpace(mesh, "CG", 2)
        t = Function(Vs)

        thickness_in = XDMFFile(fname+"_thickness.xdmf")
        with thickness_in as xdmf:
            xdmf.read_checkpoint(t, "thickness", 0)
        fname = f'P2_thick_opt_3D/{BCS}/lmax{l_max}/res{RESX}x{RESY}/{diag}/checkpt100/'
        input_dict["thickness"] = {"value": t,
                                   "type": 'Function'}
    else:
        input_dict["thickness"] = {"value": '0.12',
                                   "type": 'Function_constant',
                                   "degree": args.tdeg,
                                   "element": args.element}

    input_dict["output_file_path"] = fname + '_initial'

    fwd_timer = Timer("Forward run")
    membrane = ParametricMembrane(input_dict)

    forcing = ConstantHead(membrane, rho=10, g=10, depth=1, surround=False)
    membrane.solve(forcing, output=True)
    membrane.io.xdmf.extra_functions.append(membrane.thickness)
    membrane.io.write_fields()

    j = (membrane.lambda1 - AdjFloat(l_max))**2
    p = 600
    if method == 'abs':
        J = assemble(sqrt(j)*dx(membrane.mesh))
    elif method == 'lsq':
        J = assemble(j*dx(membrane.mesh))
    elif method == 'KS':
        from utils.natlog_overloaded import natlog
        base = assemble(exp(j*p)*dx(membrane.mesh))
        J = (1/p)*natlog(base)

    fwd_time = fwd_timer.stop()

    mat_vol0 = assemble(membrane.J_A*membrane.thickness*dx(membrane.mesh))
#        tape = get_working_tape()
#        tape.optimize_for_functionals([J])
#        tape.visualise()
    controls = [[membrane.thickness, (0.05, 0.2)]]

    if TAYLOR_TEST:
        Jhat = ReducedFunctional(J, Control(membrane.thickness))
        h = interpolate(Expression(("0.01"), degree=1), FunctionSpace(membrane.mesh, args.element, degree=args.tdeg))  # the direction of the perturbation
#        tt_results = taylor_to_dict(Jhat, membrane.thickness, h)
#        with open(fname+'taylor_test.json', 'w') as fp:
#            json.dump(tt_results, fp, indent=4)
        conv_rate = taylor_test(Jhat, membrane.thickness, h)
        print(conv_rate)
        assert(conv_rate > 1.9)

    adj_timer = Timer("Adjoint run")
    dJdm = compute_gradient(J, Control(membrane.thickness), 
                            options={"riesz_representation": "L2"})
    adj_time = adj_timer.stop()

    print("Forward time: ", fwd_time)
    print("Adjoint time: ", adj_time)
    print("Adjoint to forward runtime ratio: ",  fwd_time / adj_time)

    # Write log file
    log_dict = {"fname": fname,
                "optimization": {
                     "p KS": p,
                     "lambda_max": l_max,
                     "num_controls": membrane.Vs.dim(),
                     "bounds": {
                             'thickness': controls[0][1]},
                     "fwd_time": fwd_time,
                     "adj_time": adj_time,
                     "cost ratio": (fwd_time+adj_time)/fwd_time},
                "membrane": {
                        "resolution": input_dict["resolution"],
                        "mesh diagonal": diag,
                        "BCs": input_dict["Boundary Conditions"],
                        "pressure": input_dict["pressure"],
                        "initial_thickness": "Function" if input_dict["thickness"]["type"] == "Function" else input_dict["thickness"]["value"],
                        "thickness degree": args.tdeg,
                        "material": input_dict["material"],
                        "mu": input_dict["mu"],
                        "solver": input_dict["solver"]},
                "initial geometry": {
                         "w": w,
                         "eta": eta},
                'standard_log': {
                     'num_dofs': membrane.V.dim(),
                     'num_cells': membrane.mesh.num_cells(),
                     'ufl_cell': str(membrane.V.ufl_cell()),
                     'V_degree': membrane.V.ufl_element().degree(),
                     'Vs_degree': membrane.Vs.ufl_element().degree()}
                }

    with open(fname+'log.json', 'w') as fp:
        json.dump(log_dict, fp, indent=4)

    problem = OptProblem(J, controls, constraints=None, name=fname, params=solver_params)
    problem.solve(J)
    opt_vals = problem.opt_vals

    # final prob
    input_dict["thickness"] = {"value": opt_vals[0],
                               "type": 'Function'}
    input_dict["output_file_path"] = fname + '_final'

    membrane = ParametricMembrane(input_dict)
    forcing = ConstantHead(membrane, rho=10, g=10, depth=1, surround=False)
    membrane.solve(forcing, output=True)
    membrane.io.xdmf.extra_functions.append(membrane.thickness)
    membrane.io.write_fields()
    
    mat_vol = assemble(membrane.J_A*membrane.thickness*dx(membrane.mesh))
    savings = mat_vol0 - mat_vol
    print(f"savings={mat_vol0} - {mat_vol} = {savings}")

    results_dict = {'j': problem.j_iterates,
                    'mat_vol_0': mat_vol0,
                    'mat_vol': mat_vol,
                    'savings': savings}

    with open(fname+'_conv_history.json', 'w') as fp:
        json.dump(results_dict, fp, indent=4)

    thickness_out = XDMFFile(fname+"_thickness.xdmf") 
    thickness_out.write_checkpoint(membrane.thickness, "thickness", 0, XDMFFile.Encoding.HDF5, False)  # Not appending to file           
    thickness_out.close() 
