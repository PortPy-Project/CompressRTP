"""

This example demonstrates compressed planning based on a sparse-plus-low-rank matrix compression technique:

"""
import os
import portpy.photon as pp
from compress_rtp.utils.get_sparse_only import get_sparse_only
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def matrix_sparse_only_rmr():
    """
     1) Accessing the portpy data (DataExplorer class)
     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    # specify the patient data location.
    data_dir = r'../../PortPy/data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Patient_6'
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
    S_sparse = get_sparse_only(matrix=A, threshold_perc=1)
    inf_matrix.A = S_sparse
    opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_sparse_naive = opt.solve(solver='MOSEK', verbose=True)

    # run optimization with thresold of 1% and sparsifying matrix using RMR method
    # create cvxpy problem using the clinical criteria and optimization parameters
    S_rmr = get_sparse_only(matrix=A, threshold_perc=10, compression='rmr')
    inf_matrix.A = S_rmr
    opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_sparse_rmr = opt.solve(solver='MOSEK', verbose=True)


    """ 
    3) visualizing the dvh with and without compression (Visualization class)

    """

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
    dose_1d_naive = (S_sparse @ sol_sparse_naive['optimal_intensity']) * my_plan.get_num_of_fractions()
    dose_1d_full_naive = (A @ sol_sparse_naive['optimal_intensity']) * my_plan.get_num_of_fractions()
    ax0 = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_naive, struct_names=struct_names, style='dotted', ax=ax[0], norm_flag=True)
    ax0 = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_full_naive, struct_names=struct_names, style='solid', ax=ax0, norm_flag=True)
    ax0.set_title("sparse_naive_vs_full")

    dose_1d_rmr = (S_rmr @ sol_sparse_rmr['optimal_intensity']) * my_plan.get_num_of_fractions()
    dose_1d_full_rmr = (A @ sol_sparse_rmr['optimal_intensity']) * my_plan.get_num_of_fractions()
    ax1 = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_rmr, struct_names=struct_names, style='dashed', ax=ax[1], norm_flag=True)
    ax1 = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d_full_rmr, struct_names=struct_names, style='solid', ax=ax1, norm_flag=True)
    ax1.set_title("sparse_rmr_vs_full")
    plt.show(block=False)

    """ 
    4) evaluating the plan (Evaluation class) 
    The Evaluation class offers a set of methods for quantifying the optimized plan. 
    If you need to compute individual dose volume metrics, you can use methods such as *get_dose* or *get_volume*. 
    Furthermore, the class also facilitates the assessment of the plan based on a collection of metrics, 
    such as mean, max, and dose-volume histogram (DVH), as specified in the clinical protocol. This capability is demonstrated below
    """

    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan, dose_1d=[dose_1d_full_naive, dose_1d_full_rmr], sol_names=['Naive Sparsification', 'RMR Sparsification'])

    """ 
    5) saving and loading the plan for future use (utils) 

    """
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol_sparse_naive, sol_name='sol_sparse_naive.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol_sparse_rmr, sol_name='sol_sparse_rmr.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_sparse.pkl', path=os.path.join(r'C:\temp', data.patient_id))


if __name__ == "__main__":
    matrix_sparse_only_rmr()