"""

    This example shows creating and modification of wavelet bases for fluence map compression using portpy

"""
# import sys
# sys.path.append('..')
import portpy.photon as pp
from low_dim_rt import LowDimRT
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def ex_wavelet():
    # specify the patient data location
    # you first need to download the patient database from the link provided in the PortPy GitHub page
    data_dir = r'.\data'
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

    # With no quadratic smoothness
    prob = pp.CvxPyProb(my_plan, smoothness_weight=0)
    prob.solve(solver='MOSEK', verbose=False)
    sol_no_quad_no_wav = prob.get_sol()

    # creating the wavelet incomplete basis representing a low dimensional subspace for dimension reduction
    wavelet_basis = LowDimRT.get_low_dim_basis(my_plan.inf_matrix, 'wavelet')
    # Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    prob.constraints += [wavelet_basis @ y == prob.x]
    prob.solve(solver='MOSEK', verbose=False)
    sol_no_quad_with_wav = prob.get_sol()

    # create cvxpy problem using the clinical criteria
    prob = pp.CvxPyProb(my_plan, smoothness_weight=10)
    # run IMRT fluence map optimization using a low dimensional subspace for fluence map compression
    prob.solve(solver='MOSEK', verbose=False)
    sol_quad_no_wav = prob.get_sol()

    # Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    prob.constraints += [wavelet_basis @ y == prob.x]
    prob.solve(solver='MOSEK', verbose=False)
    sol_quad_with_wav = prob.get_sol()

    pp.save_plan(my_plan, plan_name='my_plan', path=r'C:\temp')
    pp.save_optimal_sol(sol_no_quad_no_wav, sol_name='sol_no_quad_no_wav', path=r'C:\temp')
    pp.save_optimal_sol(sol_no_quad_with_wav, sol_name='sol_no_quad_with_wav', path=r'C:\temp')
    pp.save_optimal_sol(sol_quad_no_wav, sol_name='sol_quad_no_wav', path=r'C:\temp')
    pp.save_optimal_sol(sol_quad_with_wav, sol_name='sol_quad_with_wav', path=r'C:\temp')

    # my_plan = pp.load_plan(plan_name='my_plan', path=r'C:\temp')
    # sol_no_quad_no_wav = pp.load_optimal_sol(sol_name='sol_no_quad_no_wav', path=r'C:\temp')
    # sol_no_quad_with_wav = pp.load_optimal_sol(sol_name='sol_no_quad_with_wav', path=r'C:\temp')
    # sol_quad_no_wav = pp.load_optimal_sol(sol_name='sol_quad_no_wav', path=r'C:\temp')
    # sol_quad_with_wav = pp.load_optimal_sol(sol_name='sol_quad_with_wav', path=r'C:\temp')

    # plot fluence 3D and 2D
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})
    fig.suptitle('Without Quadratic smoothness')
    pp.Visualize.plot_fluence_3d(sol=sol_no_quad_no_wav, beam_id=37, ax=ax[0], title='Without Wavelet')
    pp.Visualize.plot_fluence_3d(sol=sol_no_quad_with_wav, beam_id=37, ax=ax[1], title='With Wavelet')

    fig, ax = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})
    fig.suptitle('With Quadratic smoothness')
    pp.Visualize.plot_fluence_3d(sol=sol_quad_no_wav, beam_id=37, ax=ax[0], title='Without Wavelet')
    pp.Visualize.plot_fluence_3d(sol=sol_quad_with_wav, beam_id=37, ax=ax[1], title='With Wavelet')
    plt.show()

    # plot DVH for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNG_L', 'LUNG_R']
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax0 = pp.Visualize.plot_dvh(my_plan, sol=sol_no_quad_no_wav, structs=structs, style='solid', ax=ax[0])
    ax0 = pp.Visualize.plot_dvh(my_plan, sol=sol_no_quad_with_wav, structs=structs, style='dashed', ax=ax0)
    fig.suptitle('DVH comparison')
    ax0.set_title('Without Quadratic smoothness \n solid: Without Wavelet, Dash: With Wavelet')
    # plt.show()
    # print('\n\n')

    # fig, ax = plt.subplots(figsize=(12, 8))
    ax1 = pp.Visualize.plot_dvh(my_plan, sol=sol_quad_no_wav, structs=structs, style='solid', ax=ax[1])
    ax1 = pp.Visualize.plot_dvh(my_plan, sol=sol_quad_with_wav, structs=structs, style='dashed', ax=ax1)
    ax1.set_title('With Quadratic smoothness \n solid: Without Wavelet, Dash: With Wavelet')
    plt.show()

    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan,
                              sol=[sol_no_quad_no_wav, sol_no_quad_with_wav, sol_quad_no_wav, sol_quad_with_wav],
                              sol_names=['no_quad_no_wav', 'no_quad_with_wav', 'quad_no_wav', 'quad_with_wav'])


if __name__ == "__main__":
    ex_wavelet()
