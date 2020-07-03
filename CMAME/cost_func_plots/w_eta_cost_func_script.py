#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:10:34 2020

@author: alexanderniewiarowski
"""
import argparse
import os
import numpy as np
import pandas as pd
import sympy as sp
#%%
from dolfin import *
from fenicsmembranes.parametric_membrane import *
from fenicsmembranes.ligaro import AdjointGeoWEta as WETA
from fenicsmembranes.hydrostatic import ConstantHead

# Setup empty DFs with W and Eta vals
H = 1
THICKNESS = 0.12

input_dict = {
        "material": 'Incompressible NeoHookean',
        "cylindrical": True,
        "solver":
                {"solver": "Naive",
                 "steps": 5,
                 "scale load": False,
                 "output": False}}


def get_eta(W, H):
    '''
    Calculate eta based on on height contraint from H: C_z+R = H
    W = width
    H = minimum height ie depth
    '''
    assert W > 0.001, "does not work for small W"
    eta = sp.symbols('eta', negative=True)
    z_c = 0.5*W * (sp.sin(eta)/(1-sp.cos(eta)))

    R = W/sp.sqrt(2*(1 - sp.cos(eta)))

    e = sp.nsolve(z_c + R - H, -6.2, bisect=True) # tried many different settings, this seems to work for wide range of W
    return e


def KS(membrane, p):
    base = assemble(exp(membrane.lambda1*p)*dx(membrane.mesh))
    return (1/p)*np.log(base)

# %%
def setup(args, df_all_dir, df_hydro_dir):
    '''
    Setup the parametric study: Populate pd df's with W, eta, and hardcoded cols for results
    If more columns are needed/desired, will have to automate this as well

    NOTE: setup doesn't return anything - instead, it writes the df files to disk!
    '''

    print("RESTARTING PARAMETRIC STUDIES")

    # range for W and eta
    W = np.linspace(0.15, 4, args.num_w)
    eta_all = np.linspace(-1, -6.2, args.num_eta)
    eta_all = np.linspace(-1, -6, args.num_eta)

    # all w, eta pairs; for inflation only
    df_all = pd.DataFrame({'W': [],
                           'eta': [],
                           'p0_lambda_max': []
                           })

    # only W,eta pairs that will be hydrostatically loaded
    df_hydro = pd.DataFrame({'W': [],
                             'eta': [],
                             'p0_lambda_max': [],
                             'US_lambda_max': [],
                             'USDS_lambda_max': [],

                             'US_KS_lambda_300': [],
                             'USDS_KS_lambda_300': [],
                             
                             'US_KS_lambda_500': [],
                             'USDS_KS_lambda_500': [],

                             'US_KS_lambda_600': [],
                             'USDS_KS_lambda_600': [],
                             })

    for w in W:
        for e in eta_all:
            temp = pd.DataFrame({'W': [w], 'eta': [e]})
            df_all = df_all.append(temp, ignore_index=True)

        # get hydro_etas
        hydro_etas = [get_eta(w, h) for h in list(np.linspace(1, 2, args.num_eta)*H)]
        for e in hydro_etas:
            temp = pd.DataFrame({'W': [w], 'eta': [float(e)]})
            df_hydro = df_hydro.append(temp, ignore_index=True, sort=False)
            df_all = df_all.append(temp, ignore_index=True, sort=False)  # we also run the inflation for these etas for easy comparison

    df_all['p_0_mu'] = input_dict['pressure']/input_dict['mu']
    df_hydro['p_0_mu'] = input_dict['pressure']/input_dict['mu']

    try:
        os.makedirs(df_all_dir[:-10])
    except FileExistsError:
        pass

    df_all.to_json(df_all_dir)
    df_hydro.to_json(df_hydro_dir)

    print("RECREATED DATAFRAMES SUCCESSFULLY!")


# %% Parametric study
def _continue(args, df, path, hydrostatic=False):

    for i in df.index:

        if not df.iloc[i].isnull().any():  # if all cols populated, continue
            continue
        else:
            w = df.at[i, 'W']
            eta = df.at[i, 'eta']

            print("w, eta: ",  w, eta)

            if not args.constant_t:
                R = w/np.sqrt(2*(1 - np.cos(float(eta))))
                thickness = float(R)/21  # convert from numpy.float64
                input_dict['thickness'] = thickness
                df.at[i, 'thickness'] = thickness
            else:
                if input_dict['thickness'] is None:
                    input_dict['thickness'] = THICKNESS

            print(f'PROGRESS: {i/df.index.size}')
            print('******'*20)
            with open("./" + args.directory + "/progress.txt", "w") as text_file:
                print(f"PROGRESS: {i/df.index.size}", file=text_file)

            input_dict["geometry"] = WETA(w=w, eta=eta, dim=args.dim)
            input_dict['output_file_path'] = path[2:-5]+'/xdmf/' + f'eta_w'  # drop the ./ and the .json
            membrane = ParametricMembrane(input_dict)  # inflation only

            # Write inflation stretches:
            # Create subdomains and mark:
            df.at[i, 'p0_lambda_max'] =  max(project(membrane.lambda1, membrane.Vs).vector()[:])

            if hydrostatic:
                # Upstream
                upstream = ConstantHead(membrane, rho=10, g=10, depth=H, surround=False)
                membrane.solve(upstream, output=False)
                df.at[i, 'US_lambda_max'] = max(project(membrane.lambda1, membrane.Vs).vector()[:])
                df.at[i, 'US_KS_lambda_300'] = KS(membrane, 300)
                df.at[i, 'US_KS_lambda_500'] = KS(membrane, 500)
                df.at[i, 'US_KS_lambda_600'] = KS(membrane, 600)

                # Both sides hydro loaded UPSTREAM & Downstream
                downstream = ConstantHead(membrane, rho=10, g=10, depth=H, surround=True)
                membrane.solve(downstream, output=False)
                df.at[i, 'USDS_lambda_max'] =  max(project(membrane.lambda1, membrane.Vs).vector()[:])
                df.at[i, 'USDS_KS_lambda_300'] = KS(membrane, 300)
                df.at[i, 'USDS_KS_lambda_500'] = KS(membrane, 500)
                df.at[i, 'USDS_KS_lambda_600'] = KS(membrane, 600)

            del membrane
            df.to_json(path)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parametric study')

    parser.add_argument('--directory', type=str, help='name to append to destination directory', required=True)
    parser.add_argument('--constant_t', dest='constant_t', action='store_true', help="scale thickness by radius?")
    parser.set_defaults(constant_t=False)

    parser.add_argument('--restart', dest='restart', action='store_true')
    parser.add_argument('--no-restart', dest='restart', action='store_false')
    parser.set_defaults(restart=False)

    parser.add_argument('--num_w', type=int, required=True)
    parser.add_argument('--num_eta', type=int, required=True)

    parser.add_argument('--res', action="extend", nargs="+", type=int)
    parser.add_argument('--dim', type=int, required=True)

    parser.add_argument('--thickness', type=float)
    parser.set_defaults(thickness=None)

    parser.add_argument('--pressure', type=float)
    parser.set_defaults(pressure=18.5)

    parser.add_argument('--mu', type=float)
    parser.set_defaults(mu=10*116)

    parser.add_argument('--bc', type=str, required=True)

    args = parser.parse_args()

    input_dict["resolution"] = args.res
    input_dict["thickness"] = args.thickness
    input_dict["pressure"] = args.pressure
    input_dict["mu"] = args.mu
    input_dict["Boundary Conditions"] = args.bc

    print(args.constant_t)
    print(args.directory)

    df_all_dir = "./" + args.directory + "/df_all.json"
    df_hydro_dir = "./" + args.directory + "/df_hydro.json"
    print("The results (2 json files) will be saved here: ", df_all_dir, df_hydro_dir)

    if args.constant_t:
        print("The thickness will be constant!")
    else:
        print("The thickness will be scaled by R!")
    print('*'*20)
    # Force restart?
    if args.restart:
        setup(args, df_all_dir, df_hydro_dir)

    # Try reading the files. If files don't exist, run setup to create them
    try:
        df_all = pd.read_json(df_all_dir)
        df_hydro = pd.read_json(df_hydro_dir)
    except:
        setup(args, df_all_dir, df_hydro_dir)
        df_all = pd.read_json(df_all_dir)
        df_hydro = pd.read_json(df_hydro_dir)

    log = {
        "directory": args.directory,
        "constant t": args.constant_t,
        "H (water)": H,
        "W res": args.num_w,
        "eta res": args.num_eta,
        "dim": args.dim,
        "mesh resolution": input_dict["resolution"],
        "pressure": input_dict["pressure"],
        "mu": input_dict["mu"],
        "BCs": input_dict["Boundary Conditions"],
        "solver": input_dict["solver"]}

    if not args.constant_t:
        log["thickness"] = "variable"
    else:
        if input_dict['thickness'] is None:
            log['thickness'] = THICKNESS
        else:
            log["thickness"] = input_dict['thickness']

    import json
    with open("./" + args.directory + "/log.json", "w") as fp:
        json.dump(log, fp, indent=4)

    # Run the parametric studies.
#    _continue(args, df_all, df_all_dir)  # Entire W, eta set, no hydro loads
    _continue(args, df_hydro, df_hydro_dir, hydrostatic=True)  # Only W, eta pairs that work as dams
