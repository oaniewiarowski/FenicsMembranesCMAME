#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:38:13 2019

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

from fenicsmembranes.materials import *
from fenicsmembranes.gas import *
from fenicsmembranes.solvers import *
from .io import InputOutputHandling
from fenicsmembranes.boundary_conditions import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize":True, "quadrature_degree":5}
parameters["form_compiler"]["quadrature_degree"] = 5
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

class ParametricMembrane(object):
    """
    Parametric membrane class
    """
    def __init__(self, kwargs):

        self.material = None
        self.thickness = None
        self.solver = None
        self.data = {}
        self.phases = []
        self.kwargs = kwargs

        geo = kwargs.get("geometry")
        self.gamma = geo.gamma
        self.Gsub1 = geo.Gsub1
        self.Gsub2 = geo.Gsub2

        self.nsd = self.gamma.ufl_function_space().ufl_element().value_size()
        self._get_mesh()

        # Define function spaces
        self.Ve = VectorElement("CG", self.mesh.ufl_cell(), degree=2, dim=self.nsd)
        self.V = FunctionSpace(self.mesh, self.Ve)
        self.Vs = FunctionSpace(self.mesh, 'CG', 2)  # should this be 2?

        # Construct spaces for plotting discontinuous fields
        self.W = VectorFunctionSpace(self.mesh, 'DG', 0, dim=self.nsd)
        self.Z = FunctionSpace(self.mesh, 'DG', 1)

        # Define trial and test function
        self.du = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u = Function(self.V, name="u")

        self.initial_position = interpolate(self.gamma, self.V)
        self.p_ext = Function(self.Vs, name="External pressure")  # external pressure field
        self.l1 = Function(self.Vs, name="lambda 1")
        self.l2 = Function(self.Vs, name="lambda 2")
        self.l3 = Function(self.Vs, name="lambda 3")
        self.s11 = Function(self.Vs, name="stress1")
        self.s22 = Function(self.Vs, name="stress2")
        self.normals = Function(self.W, name="Surface unit normals")

        self.data = {'u': self.u,
                     'gamma': self.gamma,
                     'p_ext': self.p_ext,
                     'n': self.normals,
                     'l1': self.l1,
                     'l2': self.l2,
                     'l3': self.l3,
                     's11': self.s11,
                     's22': self.s22}

        # setup thickness
        if type(kwargs['thickness']) is float:
            self.thickness = Constant(kwargs['thickness'], name='thickness')
        else:
            if kwargs['thickness']['type'] == 'Constant':
                self.thickness = Constant(kwargs['thickness']['value'], name='thickness')
            elif kwargs['thickness']['type'] == 'Expression':
                self.thickness = Expression(kwargs['thickness']['value'], degree=1)
            elif kwargs['thickness']['type'] == 'Function_constant':
                deg = kwargs['thickness'].get('degree', 2)
                el = kwargs['thickness'].get('element', 'CG')
                self.thickness = interpolate(Expression(kwargs['thickness']['value'], degree=1), 
                                             FunctionSpace(self.mesh, el, deg))
                self.thickness.rename("thickness", "thickness")
            elif kwargs['thickness']['type'] == 'boundary_function':
                try:
                    self.bd_thickness = interpolate(Expression(kwargs['thickness']['value'], degree=1), self.Vs)
                except:
                    self.bd_thickness = kwargs['thickness']['value']
                
                bd = CompiledSubDomain("near(x[1],0) && on_boundary")
                bc = DirichletBC(self.Vs, self.bd_thickness, bd)
                
                u_ = TrialFunction(self.Vs)
                v = TestFunction(self.Vs)
                j_hat = Constant(('0', '1'), name="j_hat")
                a = inner(inner(grad(u_), j_hat), v)*dx(self.mesh)
                L = v*Constant(0, name="zero")*dx(self.mesh)
                
                self.thickness = Function(self.Vs, name="thickness")
                solve(a == L, self.thickness, bc)
                
            elif kwargs['thickness']['type'] == 'Function':
                self.thickness = Function(self.Vs, name="thickness")
                self.thickness.assign(project(kwargs['thickness']['value'], self.Vs))

        self.setup_kinematics()

        self.bc_type = kwargs['Boundary Conditions']
        if self.bc_type == 'Pinned':
            self.bc = pinBC(self)
        elif self.bc_type == 'Roller':
             self.bc = rollerBC(self)
        elif self.bc_type == 'Capped':
             self.bc = capBC(self)
        else:
            raise(f'Boundary condition {self.bc_type} not implemented!')

        # Initialize material, constitutive law, and internal potential energy
        material_name = kwargs["material"]
        material_class = get_material(material_name)
        self.material = material_class(self, kwargs)
        self.Pi = self.thickness*self.material.psi*self.J_A*dx(self.mesh)  #sqrt(dot(self.Gsub3, self.Gsub3))*dx(self.mesh)

        if self.nsd==3:
            self.PK2 = self.get_PK2()
            self.cauchy = self.F_n*self.PK2*self.F_n.T  #  Bonet eq(39)  = J^{-1} F S F^T Here J == 1

        # volume correction for open surfaces
        if self.nsd==2:
            self.vol_correct = 0

        elif self.bc_type=='Pinned' or self.bc_type=='Roller':
            self.open_bd = CompiledSubDomain("(near(x[1], 0) && on_boundary)")
            mesh_func = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
            self.open_bd.mark(mesh_func, 1)
            self.ds = Measure('ds', domain=self.mesh, subdomain_data=mesh_func)
            pos = self.get_position()
            x = pos[0]
            y = pos[2]
            area = assemble(-0.5*(x*self.gsub1[2] - y*self.gsub1[0])*self.ds(1))
            if ADJOINT:
                self.vol_correct = area*self.kwargs['geometry'].t/3
            else:
                self.vol_correct = Constant(area*self.kwargs['geometry'].t/3)

        elif self.bc_type == 'Capped':
            if ADJOINT:
                self.vol_correct = AdjFloat(self.kwargs['geometry'].vol_correct)
            else:
                self.vol_correct = Constant(self.kwargs['geometry'].vol_correct)
        else:
            raise("BC type not recognized, cannot calculate volume correction")

        # Create input/ouput instance and save initial states
        self.io = InputOutputHandling(self)
        self.output_file_path = kwargs["output_file_path"]
        self.io.setup()

        self.has_free_surface = self.kwargs.get("free_surface", False)

        # Initialize internal gas (if any)
        self.p_0 = Constant(kwargs['pressure'], name="initial pressure")
        gas_name = kwargs.get("gas", "Isentropic Gas")
        if gas_name is not None:
            gas_class = get_gas_law(gas_name)
            self.gas = gas_class(self)
            self.phases.append(self.gas)

        # Write initial state (uninflated) 
        with stop_annotating():
            self.io.write_fields()
            self.vol_0 = assemble((1/self.nsd)*dot(self.gamma, self.Gsub3)*dx(self.mesh)) + self.vol_correct
            self.area_0 = assemble(sqrt(dot(self.Gsub3, self.Gsub3))*dx(self.mesh))
            print(f"Initial volume: {self.vol_0}")
            print(f"Initial area: {self.area_0}")

        self.inflate(self.p_0)

    def _get_mesh(self):
        res = self.kwargs.get("resolution")
        assert len(res)==self.nsd-1, "Mesh resolution does not match dimension" 
        
        mesh_type = self.kwargs.get("mesh_type", "Default")
       
        if self.nsd==2:
            self.mesh = UnitIntervalMesh(res[0])

        elif self.nsd==3:
            diag = self.kwargs.get("mesh diagonal", 'crossed')
            self.mesh = UnitSquareMesh(res[0], res[1], diag)

            # TODO: Fix this up
            try:
                meshx= kwargs['meshx']
                meshxy= kwargs['meshy']
            except:
                meshx = 1; meshy = 1

            if meshx != 1 or meshy !=1:
                x = self.mesh.coordinates()
                scaling_factorx = meshx
                scaling_factory = meshy
                x[:, 0] *= scaling_factorx
                x[:, 1] *= scaling_factory
                self.mesh.bounding_box_tree().build(self.mesh)

    def setup_kinematics(self):
        from dolfin import cross, sqrt, dot
        if self.nsd==2:
            # Get the dual basis
            self.Gsup1 = self.Gsub1/dot(self.Gsub1, self.Gsub1)
            R = as_tensor([[0, -1],
                           [1, 0]])

            self.gsub1 = self.Gsub1 + self.u.dx(0)
            gradu = outer(self.u.dx(0), self.Gsup1)

            I = Identity(self.nsd)
            self.F = I + gradu
            self.C = dot(self.F.T, self.F)

            self.Gsub3 = dot(R, self.Gsub1)
            self.gsub3 = dot(R, self.gsub1)
            lmbda = sqrt(dot(self.gsub1, self.gsub1)/dot(self.Gsub1, self.Gsub1))
            self.lambda1, self.lambda2 = lmbda, lmbda
            self.lambda3 = 1/self.lambda1

        elif self.nsd==3:
            from dolfin import cross, sqrt, dot
            from fenicsmembranes.calculus_utils import contravariant_base_vector
            
            # Get the contravariant tangent basis
            self.Gsup1 = contravariant_base_vector(self.Gsub1, self.Gsub2)
            self.Gsup2 = contravariant_base_vector(self.Gsub2, self.Gsub1)

            # Reference normal 
            self.Gsub3 = cross(self.Gsub1, self.Gsub2)
            self.Gsup3 = self.Gsub3/dot(self.Gsub3, self.Gsub3)


            # Construct the covariant convective basis
            self.gsub1 = self.Gsub1 + self.u.dx(0)
            self.gsub2 = self.Gsub2 + self.u.dx(1)

            # Construct the contravariant convective basis
            self.gsup1 = contravariant_base_vector(self.gsub1, self.gsub2)
            self.gsup2 = contravariant_base_vector(self.gsub2, self.gsub1)

            # Deformed normal
            self.gsub3 = cross(self.gsub1, self.gsub2)
            self.gsup3 = self.gsub3/dot(self.gsub3, self.gsub3)

            # Deformation gradient
            gradu = outer(self.u.dx(0), self.Gsup1) + outer(self.u.dx(1), self.Gsup2)
            I = Identity(self.nsd)

            self.F = I + gradu
            self.C = self.F.T*self.F # from initial to current

            # 3x2 deformation tensors
            # TODO: check/test
            self.F_0 = as_tensor([self.Gsub1, self.Gsub2]).T
            self.F_n = as_tensor([self.gsub1, self.gsub2]).T

            # 2x2 surface metrics
            self.C_0 = self.get_metric(self.Gsub1, self.Gsub2)
            self.C_0_sup = self.get_metric(self.Gsup1, self.Gsup2)
            self.C_n = self.get_metric(self.gsub1, self.gsub2)

            # TODO: not tested. do we need these?
            self.det_C_0 = dot(self.Gsub1, self.Gsub1)*dot(self.Gsub2, self.Gsub2) - dot(self.Gsub1,self.Gsub2)**2
            self.det_C_n = dot(self.gsub1, self.gsub1)*dot(self.gsub2, self.gsub2) - dot(self.gsub1,self.gsub2)**2

            self.lambda1, self.lambda2, self.lambda3 = self.get_lambdas()

            self.I1 = inner(inv(self.C_0), self.C_n)
            self.I2 = det(self.C_n)/det(self.C_0)

        else:
            raise Exception("Could not infer spatial dimension")
        
        # Unit normals
        self.J_A = sqrt(dot(self.Gsub3, self.Gsub3))
        self.N = self.Gsub3/self.J_A
        self.j_a = sqrt(dot(self.gsub3, self.gsub3))
        self.n = self.gsub3/self.j_a

    def get_metric(self, i,j):
        return as_tensor([[dot(i, i), dot(i, j)], [dot(j, i), dot(j, j)]])

    def get_PK2(self):
        '''
        2nd Piola-Kirchhoff Stress
        '''
        # S = mem.material.mu*(dolfin.inv(C_0) - (dolfin.det(C_0)/dolfin.det(C_n))*dolfin.inv(C_n))
        A = 1/self.det_C_0
        B = self.det_C_0/(det(self.C_n)**2)
        G1 = self.Gsub1
        G2 = self.Gsub2
        g1 = self.gsub1
        g2 = self.gsub2

        G1G1 = dot(G1, G1)
        G2G2 = dot(G2, G2)
        G1G2 = dot(G1, G2)

        g1g1 = dot(g1, g1)
        g2g2 = dot(g2, g2)
        g1g2 = dot(g1, g2)

        mu = self.material.mu
        S = mu*as_matrix([[A*G2G2 - B*g2g2, -A*G1G2 + B*g1g2], [-A*G1G2 + B*g1g2, A*G1G1 - B*g1g1]])
        return S

    def get_lambdas(self):
        C_n = self.F_n.T*self.F_n
        C_0 = self.F_0.T*self.F_0
        I1 = inner(inv(C_0), C_n)
        I2 = det(C_n)/det(C_0)
        delta = sqrt(I1**2 -4*I2)

        lambda1 = sqrt(0.5*(I1 + delta))
        lambda2 = sqrt(0.5*(I1 - delta))
        lambda3 = sqrt(det(self.C_0)/det(self.C_n))

        return lambda1, lambda2, lambda3

    def get_I1(self):
        return inner(inv(self.C_0), self.C_n)

    def get_I2(self):
        return det(self.C_n)/det(self.C_0)

    def get_position(self):
        return self.gamma + self.u

    def calculate_volume(self, u):
        # FIXME: Why is this a function of u???
        '''
        Calculates the current volume of the membrane with a deformation given by dolfin function u.
        We do this by integrating a function with unit divergence:
        V = \int dV = \frac{1}{dim} x \cdot n dA

        Args:
            u (Type): Dolfin function

        '''
        volume = assemble((1/self.nsd)*dot(self.gamma + u, self.gsub3)*dx(self.mesh))
        if self.nsd == 3:
            if self.bc_type=='Capped':
                volume += self.vol_correct
            elif self.bc_type == 'Roller' or self.bc_type == 'Pinned':
                pos = self.get_position()
                x = pos[0]
                y = pos[2]
                area = assemble(-0.5*(x*self.gsub1[2] - y*self.gsub1[0])*self.ds(1))
                if ADJOINT:
                    self.vol_correct = area*self.kwargs['geometry'].t/3
                    volume += AdjFloat(self.vol_correct)
                else:
                    self.vol_correct = area*self.kwargs['geometry'].t/3
                    volume += self.vol_correct
        return volume

    def inflate(self, pressure):
        """
        Inflate a membrane to a given pressure, no external forces or tractions

        Args:
            pressure (float): The pressure of the membrane
        """

        if not hasattr(pressure, "values"):
            p = Constant(pressure)
        else:
            p = pressure

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = self.DPi_int() - self.DPi_air(p)
        
        if self.kwargs.get("free_surface"):
            F -= self.liquid.DPi_0

        # Compute Jacobian of F
        K = derivative(F, self.u, self.du)
        problem = NonlinearVariationalProblem(F, self.u, bcs=self.bc, J=K, form_compiler_parameters={"optimize": True})

        # Create solver and call solve
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters

        prm["nonlinear_solver"] = "snes"
        prm["snes_solver"]["linear_solver"] = "cg"

        solver.parameters.update(prm)

        print("*** SOLVING INITIAL INFLATION ***") 
        solver.solve()
        self.inflation_solver = solver
            # Solve variational problem          
        
        for phase in self.phases:
            phase.setup()

        with stop_annotating():
            self.io.write_fields()

    def DPi_int(self):
        '''
        directional derivative of \Pi_int in an arbitrary direction \delta v
        DPi_int(\phi)[\delta v]
        '''
        return derivative(self.Pi, self.u, self.v)

    def DPi_ext(self):
        return dot(self.tractions, self.v)*dx(self.mesh)  #FIXME not implemented yet

    def DPi_air(self, p):
        '''
        p*dot(self.v, dot(self.R, self.gsub1))*dx(self.mesh)

        Args:
            p (): p should be a Constant
        '''
        return p * dot(self.v, self.gsub3)*dx(self.mesh)

    def k_load(self):
        '''
        Need to multiply by p! eq. 27 in 
        Rumpel, T., & Schweizerhof, K. (2003). 
        Volume-dependent pressure loading and its influence 
        on the stability of structures. 
        doi.org/10.1002/nme.561
        '''
        # Construct skew symmetric tensors
        from fenicsmembranes.calculus_utils import wedge
        self.W1 = wedge(self.gsub3, self.gsup1)  # outer(self.gsub3, self.gsup1) - outer(self.gsup1, self.gsub3)
        self.W2 = wedge(self.gsub3, self.gsup2)  # outer(self.gsub3, self.gsup2) - outer(self.gsup2, self.gsub3)

        k_load = (0.5)*(dot(self.W1.T*self.du, self.v.dx(0)) + \
                  dot(self.W2.T*self.du, self.v.dx(1)))*dx(self.mesh) + \
                (0.5)*dot((self.W1*self.du.dx(0) + self.W2*self.du.dx(1)),self.v)*dx(self.mesh)
        return k_load


    def solve(self, external_load, output=True):  # TODO get rid of output
        kwargs = self.kwargs if type(self.kwargs["solver"]) is str else self.kwargs["solver"]  # backwards compatible until I change all input dicts in tests!
        # FIXME
        solver_name = kwargs["solver"]
        solver_class = get_solver(solver_name)
        self.solver = solver_class(self, external_load, kwargs)
        return self.solver.solve()
