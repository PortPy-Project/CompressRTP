"""

    This example shows creating and modification of wavelet bases for fluence map compression using portpy

"""
import portpy.photon as pp
from low_dim_rt import LowDimRT
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def ex_wavelet():
    # specify the patient data location
    # you first need to download the patient database from the link provided in the PortPy GitHub page
    data_dir = r'F:\Research\Data_newformat\Python-PORT\data'
    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, ...)
    patient_id = 'Lung_Patient_2'
    # create my_plan object for the planner beams_dict and select among the beams which are 30 degrees apart
    # for the customized beams_dict, you can pass the argument beam_ids
    my_plan = pp.Plan(patient_id, data_dir)

    # Let us create rinds for creating reasonable dose fall off for the plan
    rind_max_dose = np.array([1.1, 1.05, 0.9, 0.85, 0.75]) * my_plan.get_prescription()
    rind_params = [{'rind_name': 'RIND_0', 'ref_structure': 'PTV', 'margin_start_mm': 0, 'margin_end_mm': 5,
                    'max_dose_gy': rind_max_dose[0]},
                   {'rind_name': 'RIND_1', 'ref_structure': 'PTV', 'margin_start_mm': 5, 'margin_end_mm': 10,
                    'max_dose_gy': rind_max_dose[1]},
                   {'rind_name': 'RIND_2', 'ref_structure': 'PTV', 'margin_start_mm': 10, 'margin_end_mm': 30,
                    'max_dose_gy': rind_max_dose[2]},
                   {'rind_name': 'RIND_3', 'ref_structure': 'PTV', 'margin_start_mm': 30, 'margin_end_mm': 60,
                    'max_dose_gy': rind_max_dose[3]},
                   {'rind_name': 'RIND_4', 'ref_structure': 'PTV', 'margin_start_mm': 60, 'margin_end_mm': 'inf',
                    'max_dose_gy': rind_max_dose[4]}]
    my_plan.add_rinds(rind_params=rind_params)

    # create cvxpy problem using the clinical criteria
    prob = pp.CvxPyProb(my_plan, opt_params={'smoothness_weight': 10})

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    prob.solve(solver='MOSEK', verbose=True)
    sol = prob.get_sol()

    # run IMRT fluence map optimization using a low dimensional subspace for fluence map compression
    prob = pp.CvxPyProb(my_plan, opt_params={'smoothness_weight': 10})
    # creating the wavelet incomplete basis representing a low dimensional subspace for dimension reduction
    wavelet_basis = LowDimRT.get_low_dim_basis(my_plan.inf_matrix, 'wavelet')
    # Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    prob.constraints += [wavelet_basis @ y == prob.x]
    prob.solve(solver='MOSEK', verbose=False)
    sol_low_dim = prob.get_sol()

    # With no quadratic smoothness
    prob = pp.CvxPyProb(my_plan, opt_params={'smoothness_weight': 0})
    # creating the wavelet incomplete basis representing a low dimensional subspace for dimension reduction
    wavelet_basis = LowDimRT.get_low_dim_basis(my_plan.inf_matrix, 'wavelet')
    # Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    prob.constraints += [wavelet_basis @ y == prob.x]
    prob.solve(solver='MOSEK', verbose=False)
    sol_low_dim_only = prob.get_sol()

    # plot fluence 3D and 2D
    fig, ax = plt.subplots(1, 3, figsize=(12, 12), subplot_kw={'projection': '3d'})
    pp.Visualize.plot_fluence_3d(sol=sol, beam_id=0, ax=ax[0])
    pp.Visualize.plot_fluence_3d(sol=sol_low_dim, beam_id=0, ax=ax[1])
    pp.Visualize.plot_fluence_3d(sol=sol_low_dim_only, beam_id=0, ax=ax[2])
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(12, 12))
    pp.Visualize.plot_fluence_2d(sol=sol, beam_id=0, ax=ax[0])
    pp.Visualize.plot_fluence_2d(sol=sol_low_dim, beam_id=0, ax=ax[1])
    pp.Visualize.plot_fluence_2d(sol=sol_low_dim_only, beam_id=0, ax=ax[2])
    plt.show()
    # plot DVH for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = pp.Visualize.plot_dvh(my_plan, sol=sol, structs=structs, ax=ax)
    ax = pp.Visualize.plot_dvh(my_plan, sol=sol_low_dim, structs=structs, ax=ax)
    pp.Visualize.plot_dvh(my_plan, sol=sol_low_dim_only, structs=structs, ax=ax)
    plt.show()
    # plot 2d axial slice for the given solution and display the structures contours on the slice
    fig, ax = plt.subplots(1, 3, figsize=(12, 12))
    pp.Visualize.plot_2d_dose(my_plan, sol=sol, slice_num=40, structs=['PTV'], ax=ax[0])
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_low_dim, slice_num=40, structs=['PTV'], ax=ax[1])
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_low_dim_only, slice_num=40, structs=['PTV'], ax=ax[2])
    plt.show()
    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan, sol=sol)
    pp.Visualize.plan_metrics(my_plan, sol=sol_low_dim)
    pp.Visualize.plan_metrics(my_plan, sol=sol_low_dim_only)


if __name__ == "__main__":
    ex_wavelet()
