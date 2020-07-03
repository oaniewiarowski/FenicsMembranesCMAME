# -*- coding: utf-8 -*-
import dolfin

_SOLVERS = {}

def add_solver(name, solver_class):
    """
    Register a solver
    """
    _SOLVERS[name] = solver_class


def register_solver(name):
    """
    A class decorator to register solver
    """

    def register(solver_class):
        add_solver(name, solver_class)
        return solver_class

    return register


def get_solver(name):
    """
    Return a solver model by name
    """
    try:
        return _SOLVERS[name]
    except KeyError:
        raise


class Solver(object):
    pass

from . import naive
