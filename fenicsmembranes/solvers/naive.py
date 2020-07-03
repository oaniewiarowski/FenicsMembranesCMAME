from dolfin import *

try:
    from dolfin_adjoint import *
    from pyadjoint import annotate_tape, get_working_tape, stop_annotating
    ADJOINT = True
except ModuleNotFoundError:
    print('dolfin_adjoint not available. Running forward model only!')
    ADJOINT = False
    import contextlib
    def stop_annotating():
        return contextlib.nullcontext()

import numpy as np
from . import Solver, register_solver


@register_solver('Naive')
class NaiveSolver(Solver):
    def __init__(self, membrane, external_load, kwargs):
        self.mem = mem = membrane
        self.external_load = external_load
        self.output = kwargs.get("output", True)

        self.discr = kwargs.get("steps", 5)
        self.kwargs = kwargs
        self.scale_load = kwargs.get("scale load", True)
        # Forms
        self.a_form = dot(mem.gsub3, mem.v)*dx(mem.mesh)
        self.f_int = mem.DPi_int()

    def update_F(self):

        mem = self.mem
        self.current_int_p = mem.gas.update_pressure()
        print('pressure=', mem.gas.p); print(f"Volume: {mem.gas.V}")
        self.f_ext_gas_form = -self.inc_load*dot(mem.v, mem.gsub3)*dx(mem.mesh)
        self.F = self.f_int - self.f_ext_gas_form - self.current_int_p*self.a_form  


    def update_K(self):
        self.K = derivative(self.F, self.mem.u, self.mem.du)

    def solve(self):
        load_frac = Constant(0)


        for step, i in enumerate(np.linspace(0.0, 1.0, self.discr)):
            if step == 0:
                continue
            elif step==self.discr - 1:

                if hasattr(self.external_load, 'update'):
                    # update the load
                    self.inc_load = self.external_load.update()
                else:
                    # step up constant pressure
                    self.inc_load.assign(-self.external_load)
                self.update_F()
                self.update_K()
                annotate = True

            else:
                if hasattr(self.external_load, 'update'):
                    # update the load
                    i = i if self.scale_load else 1.
                    load_frac.assign(i)
                    self.inc_load = self.external_load.update()*load_frac
                else:
                    # step up constant pressure
                    self.inc_load = -load_frac*self.external_load

                self.update_F()
                self.update_K()
                annotate = False

            problem = NonlinearVariationalProblem(self.F, self.mem.u, bcs=self.mem.bc, J=self.K, form_compiler_parameters={"optimize": True})
#
#            # Create solver and call solve
            solver = NonlinearVariationalSolver(problem)
#            solver.parameters.update({"nonlinear_solver":"snes"})

            prm = solver.parameters
            prm["nonlinear_solver"] = "snes"
            prm["snes_solver"]["maximum_iterations"] =200
            prm["snes_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
            print(f"SOLVING FORWARD PROBLEM STEP {step}")
            if ADJOINT:
                solver.solve(annotate=annotate)
            else:
                solver.solve()


            if self.output:
                with stop_annotating():
                    self.mem.io.write_fields()
