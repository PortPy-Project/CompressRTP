"""

This example demonstrates compressed planning based on a sparse-plus-low-rank matrix compression technique:

"""
import os
import portpy.photon as pp
from compress_rtp.compress_rtp_optimization import CompressRTPOptimization
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import scipy


def sparse_plus_low_rank():
    """
     1) Accessing the portpy data (DataExplorer class)
     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    # specify the patient data location.
    data_dir = r'../../PortPy/data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Patient_3'

    ct = pp.CT(data)
    structs = pp.Structures(data)

    # If the list of beams are not provided, it uses the beams selected manually
    # by a human expert planner for the patient (manually selected beams are stored in portpy data).
    # Create beams for the planner beams by default
    # for the customized beams, you can pass the argument beam_ids
    # e.g. beams = pp.Beams(data, beam_ids=[0,10,20,30,40,50,60])
    beams = pp.Beams(data, load_inf_matrix_full=True)

    # In order to create an IMRT plan, we first need to specify a protocol which includes the disease site,
    # the prescribed dose for the PTV, the number of fractions, and the radiation dose thresholds for OARs.
    # These information are stored in .json files which can be found in a directory named "config_files".
    # An example of such a file is 'Lung_2Gy_30Fx.json'. Here's how you can load these files:
    protocol_name = 'Lung_2Gy_30Fx'
    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    # Optimization problem formulation
    protocol_name = 'Lung_2Gy_30Fx'
    # Loading hyper-parameter values for optimization problem
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # Creating optimization structures (i.e., Rinds)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
    # Loading influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams, is_full=True)

    """

    2) creating a simple IMRT plan using CVXPy (Plan class, Optimization class)
    Note: you can call different opensource / commercial optimization engines from CVXPy.
      For commercial engines (e.g., Mosek, Gorubi, CPLEX), you first need to obtain an appropriate license.
      Most commercial optimization engines give free academic license.

    Create my_plan object which would store all the data needed for optimization
      (e.g., influence matrix, structures and their voxels, beams and their beamlets).

    """
    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)

    # run optimization with naive thresold of 1% of max(A) and no low rank
    # create cvxpy problem using the clinical criteria and optimization parameters
    A = deepcopy(inf_matrix.A)
    tol = np.max(A) * 1 * 0.01
    S = np.where(A > tol, A, 0)
    S = scipy.sparse.csr_matrix(S)
    inf_matrix.A = S
    opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_sparse = opt.solve(solver='MOSEK', verbose=True)

    # run optimization with thresold of 1% and rank 5
    # create cvxpy problem using the clinical criteria and optimization parameters
    opt = CompressRTPOptimization(my_plan, opt_params=opt_params)
    S, H, W = opt.get_sparse_plus_low_rank(A=A, thresold_perc=1, rank=5)
    opt.create_cvxpy_problem_compressed(S=S, H=H, W=W)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    mosek_params = {'MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES': 0,
                    'MSK_IPAR_INTPNT_SCALING': 'MSK_SCALING_NONE'}
    sol_slr = opt.solve(solver='MOSEK', verbose=True, mosek_params=mosek_params)

    """ 
    3) visualizing the dvh with and without compression (Visualization class)

    """

    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
    dose_1d_sparse = (S @ sol_sparse['optimal_intensity']) * my_plan.get_num_of_fractions()
    dose_1d_full = (A @ sol_sparse['optimal_intensity']) * my_plan.get_num_of_fractions()
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_sparse, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_full, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax.set_title("sparse_vs_full")
    plt.show(block=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
    dose_1d_slr = (S @ sol_slr['optimal_intensity'] + H @ (W @ sol_slr['optimal_intensity'])) * my_plan.get_num_of_fractions()
    dose_1d_full = (A @ sol_slr['optimal_intensity']) * my_plan.get_num_of_fractions()
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_slr, struct_names=struct_names, style='dashed', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_full, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax.set_title("slr_vs_full")
    plt.show(block=False)

    """ 
    4) evaluating the plan (Evaluation class) 
    The Evaluation class offers a set of methods for quantifying the optimized plan. 
    If you need to compute individual dose volume metrics, you can use methods such as *get_dose* or *get_volume*. 
    Furthermore, the class also facilitates the assessment of the plan based on a collection of metrics, 
    such as mean, max, and dose-volume histogram (DVH), as specified in the clinical protocol. This capability is demonstrated below
    """

    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan, dose_1d=[dose_1d_sparse, dose_1d_slr], in_browser=True)

    """ 
    5) saving and loading the plan for future use (utils) 

    """
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol_slr, sol_name='sol_sparse.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol_slr, sol_name='sol_slr.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_sparse.pkl', path=os.path.join(r'C:\temp', data.patient_id))


if __name__ == "__main__":
    sparse_plus_low_rank()
