from fenicsmembranes.parametric_membrane import *
from fenicsmembranes.variablegeometry import VariableGeometry as Cylinder
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
                'name': [],
                }
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
        self.write_freq = 10

    def eval_cb(self, j, m):
        pass

    def derivative_cb(self, j, dj, m):
        self.j_iterates.append(j)
        self.j = j
        if self.iteration % self.write_freq == 0 or self.iteration == self.params['maximum_iterations']:
            with self.iter_viz as xdmf:
                m[0].rename("thickness", "thickness")
                xdmf.write(m[0], self.iteration)
        self.iteration += 1

    def add_control(self, var, bounds, name=None):
        if name is None:
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


def pre_solve_thickness(args, initial_thickness):

        mesh = UnitSquareMesh(args.resx, args.resy, args.diag)
        Vs = FunctionSpace(mesh, 'CG', 2)
        bd_thickness = Function(Vs, name="boundary_thickness")
        try:
            bd_thickness = interpolate(Expression('t', t=initial_thickness, degree=1), Vs) # float 
        except:
            bd_thickness = initial_thickness  # function

        bd = CompiledSubDomain("near(x[1],0) && on_boundary")
        bc = DirichletBC(Vs, bd_thickness, bd)

        u_ = TrialFunction(Vs)
        v = TestFunction(Vs)
        j_hat = Constant(('0', '1'), name="j_hat")
        a = inner(inner(grad(u_), j_hat), v)*dx(mesh)
        L = v*Constant(0, name="zero")*dx(mesh)
        
        thickness = Function(Vs, name="thickness")
        solve(a == L, thickness, bc)
        return thickness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thickness opt plane strain')
    parser.add_argument('--path', type=str, required=True)

    parser.add_argument('--resx', type=int, required=True)
    parser.add_argument('--resy', type=int, required=True)
    parser.add_argument('--diag', type=str, required=True)

    parser.add_argument('--method', type=str, required=True)
#    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--lmax', type=float, required=True)
    parser.add_argument('--itr', type=int, required=True)

    # Options: Run the optimization? Taylor test?
    parser.add_argument('--no-optimize', dest='opt', action='store_false', default=True)
    parser.add_argument('--tt', dest='tt', action='store_true', default=False)

    args = parser.parse_args()

    OPTIMIZE = args.opt
    TAYLOR_TEST = args.tt

    diag = args.diag
    l_max = AdjFloat(args.lmax)  # AdjFloat(1.02)
    method = args.method
    fname = args.path
    solver_params = {"maximum_iterations": args.itr,
                     "acceptable_tol": 1.0e-8}

    w = 2
    eta = pi
    geo = Cylinder(w=w, eta=eta, length=1)

    RESX = args.resx
    RESY = args.resy

#    thickness = pre_solve_thickness(args, 0.12)
    # initial prob
    input_dict = {
        "resolution": [RESX, RESY],
        "geometry": geo,
        "mesh diagonal": diag,
        "thickness":
                    {"value": '0.12',  # thickness, #
                     "type": 'boundary_function'  #  "Function" #
                     },
        "pressure": 18.5,
        "material": 'Incompressible NeoHookean',
        "mu": 10*116,
        "output_file_path": fname + '_initial',
        "Boundary Conditions": "Pinned",
        "cylindrical": True,
        "solver": "Naive"}

    fwd_timer = Timer("Forward run")
    membrane = ParametricMembrane(input_dict)
    forcing = ConstantHead(membrane, rho=10, g=10, depth=1, surround=False)
    membrane.solve(forcing, output=True)
    membrane.io.xdmf.extra_functions.append(membrane.bd_thickness)
    membrane.io.xdmf.extra_functions.append(membrane.thickness)
    membrane.io.write_fields()

    l = membrane.lambda1
    j = (l - l_max)**2

    if method == 'abs':
        J = assemble(sqrt(j)*dx(membrane.mesh))
    elif method == 'lsq':
        J = assemble(j*dx(membrane.mesh))
    elif method == 'KS':
        from utils.natlog_overloaded import natlog
        p = 300              
        base = assemble(exp(j*p)*dx(membrane.mesh))
        J = (1/p)*natlog(base)

    fwd_time = fwd_timer.stop()
#    tape = get_working_tape()
#    tape.visualise()
    mat_vol0 = assemble(membrane.J_A*membrane.thickness*dx(membrane.mesh))
    controls = [[membrane.bd_thickness, (0.0001, 0.3)]]

    if TAYLOR_TEST:
        Jhat = ReducedFunctional(J, Control(membrane.bd_thickness))
        h = interpolate(Expression(("0.01"), degree=1), membrane.Vs)  # the direction of the perturbation
        conv_rate = taylor_test(Jhat, membrane.thickness, h)
        print(conv_rate)
#        assert(conv_rate > 1.9)
#        tt_results = taylor_to_dict(Jhat, membrane.thickness, h)
        tt_results = {'min_res': conv_rate}
        with open(fname+'taylor_test.json', 'w') as fp:
            json.dump(tt_results, fp, indent=4)


    adj_timer = Timer("Adjoint run")
    dJdm = compute_gradient(J, Control(controls[0][0]), 
                            options={"riesz_representation": "L2"})
    adj_time = adj_timer.stop()

    print("Forward time: ", fwd_time)
    print("Adjoint time: ", adj_time)
    print("Adjoint to forward runtime ratio: ",  fwd_time / adj_time)

    # Write log file
    log_dict = {"fname": fname,
                "optimization": {
                     "p KS": p if method=='KS' else 'none',
                     "lambda_max": l_max,
                     "num_controls": controls[0][0].function_space().dim(),
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
                        "thickness degree": controls[0][0].ufl_element().degree(),
                        "material": input_dict["material"],
                        "mu": input_dict["mu"],
                        "solver": input_dict["solver"]},
                "initial geometry": {
                         "w": w,
                         "eta": eta},
                "notes": '',
                'standard_log': {
                     'num_dofs': membrane.V.dim(),
                     'num_cells': membrane.mesh.num_cells(),
                     'ufl_cell': str(membrane.V.ufl_cell()),
                     'V_degree': membrane.V.ufl_element().degree(),
                     'Vs_degree': membrane.Vs.ufl_element().degree()}
                }

    with open(fname+'log.json', 'w') as fp:
        json.dump(log_dict, fp, indent=4)

    if OPTIMIZE:
        problem = OptProblem(J, controls, name=fname, params=solver_params)
        problem.solve(J)
        opt_vals = problem.opt_vals

    #    thickness = pre_solve_thickness(args, opt_vals[0])
        # final prob
        input_dict["thickness"] = {"value": opt_vals[0],  # thickness,
                                     "type": 'boundary_function'}  # "Function"  #

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
