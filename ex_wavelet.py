"""

    This example shows creating and modification of wavelet bases for fluence map compression using portpy

"""

import portpy.photon as pp
from low_dim_rt import LowDimRT
import cvxpy as cp
import matplotlib.pyplot as plt


def ex_wavelet():

    """
    1) Accessing the portpy data (DataExplorer class)

    """

    #  Note: you first need to download the patient database from the link provided in the GitHub page.

    # specify the patient data location.
    data_dir = r'..\data'

    # Use PortPy DataExplorer class to explore PortPy data and pick one of the patient
    data = pp.DataExplorer(data_dir=data_dir)
    patient_id = 'Lung_Patient_3'
    data.patient_id = patient_id

    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    structs = pp.Structures(data)

    # If the list of beams are not provided, it uses the beams selected manually
    # by a human expert planner for the patient (manually selected beams are stored in portpy data).
    # Create beams for the planner beams by default
    # for the customized beams, you can pass the argument beam_ids
    # e.g. beams = pp.Beams(data, beam_ids=[0,10,20,30,40,50,60])
    beams = pp.Beams(data)

    # create rinds based upon rind definition in optimization params
    protocol_name = 'Lung_2Gy_30Fx'
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    structs.create_opt_structures(opt_params=opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    '''
    2) Optimizing the plan without quadratic smoothness (with and without wavelet constraint)
    
    '''
    # - Without wavelet constraint

    # remove smoothness objective
    for i in range(len(opt_params['objective_functions'])):
        if opt_params['objective_functions'][i]['type'] == 'smoothness-quadratic':
            opt_params['objective_functions'][i]['weight'] = 0

    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)

    # create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_no_quad_no_wav = opt.solve(solver='MOSEK', verbose=False)

    # - With wavelet constraint

    # creating the wavelet incomplete basis representing a low dimensional subspace for dimension reduction
    wavelet_basis = LowDimRT.get_low_dim_basis(my_plan.inf_matrix, 'wavelet')
    # Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    opt.constraints += [wavelet_basis @ y == opt.vars['x']]
    sol_no_quad_with_wav = opt.solve(solver='MOSEK', verbose=False)

    ''' 
    3) Optimizing the plan with quadratic smoothness
    
    '''
    # - Without wavelet constraint

    for i in range(len(opt_params['objective_functions'])):
        if opt_params['objective_functions'][i]['type'] == 'smoothness-quadratic':
            opt_params['objective_functions'][i]['weight'] = 10

    # create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()

    sol_quad_no_wav = opt.solve(solver='MOSEK', verbose=False)

    # - With Wavelet constraint

    # Wavelet Smoothness Constraint
    y = cp.Variable(wavelet_basis.shape[1])
    opt.constraints += [wavelet_basis @ y == opt.vars['x']]
    sol_quad_with_wav = opt.solve(solver='MOSEK', verbose=False)

    '''
    4) Saving and loading plans

    '''

    pp.save_plan(my_plan, plan_name='my_plan.pkl', path=r'C:\temp')
    pp.save_optimal_sol(sol_no_quad_no_wav, sol_name='sol_no_quad_no_wav.pkl', path=r'C:\temp')
    pp.save_optimal_sol(sol_no_quad_with_wav, sol_name='sol_no_quad_with_wav.pkl', path=r'C:\temp')
    pp.save_optimal_sol(sol_quad_no_wav, sol_name='sol_quad_no_wav.pkl', path=r'C:\temp')
    pp.save_optimal_sol(sol_quad_with_wav, sol_name='sol_quad_with_wav.pkl', path=r'C:\temp')

    # my_plan = pp.load_plan(plan_name='my_plan', path=r'C:\temp')
    # sol_no_quad_no_wav = pp.load_optimal_sol(sol_name='sol_no_quad_no_wav', path=r'C:\temp')
    # sol_no_quad_with_wav = pp.load_optimal_sol(sol_name='sol_no_quad_with_wav', path=r'C:\temp')
    # sol_quad_no_wav = pp.load_optimal_sol(sol_name='sol_quad_no_wav', path=r'C:\temp')
    # sol_quad_with_wav = pp.load_optimal_sol(sol_name='sol_quad_with_wav', path=r'C:\temp')
    
    '''
    5) Visualization and compare the plans
    
    '''
    # plot fluence 3D and 2D
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})
    fig.suptitle('Without Quadratic smoothness')
    pp.Visualization.plot_fluence_3d(sol=sol_no_quad_no_wav, beam_id=37, ax=ax[0], title='Without Wavelet')
    pp.Visualization.plot_fluence_3d(sol=sol_no_quad_with_wav, beam_id=37, ax=ax[1], title='With Wavelet')

    fig, ax = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': '3d'})
    fig.suptitle('With Quadratic smoothness')
    pp.Visualization.plot_fluence_3d(sol=sol_quad_no_wav, beam_id=37, ax=ax[0], title='Without Wavelet')
    pp.Visualization.plot_fluence_3d(sol=sol_quad_with_wav, beam_id=37, ax=ax[1], title='With Wavelet')
    plt.show()

    # plot DVH for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNG_L', 'LUNG_R']
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax0 = pp.Visualization.plot_dvh(my_plan, sol=sol_no_quad_no_wav, structs=structs, style='solid', ax=ax[0])
    ax0 = pp.Visualization.plot_dvh(my_plan, sol=sol_no_quad_with_wav, structs=structs, style='dashed', ax=ax0)
    fig.suptitle('DVH comparison')
    ax0.set_title('Without Quadratic smoothness \n solid: Without Wavelet, Dash: With Wavelet')
    # plt.show()
    # print('\n\n')

    # fig, ax = plt.subplots(figsize=(12, 8))
    ax1 = pp.Visualization.plot_dvh(my_plan, sol=sol_quad_no_wav, structs=structs, style='solid', ax=ax[1])
    ax1 = pp.Visualization.plot_dvh(my_plan, sol=sol_quad_with_wav, structs=structs, style='dashed', ax=ax1)
    ax1.set_title('With Quadratic smoothness \n solid: Without Wavelet, Dash: With Wavelet')
    plt.show()

    '''
    6) Evaluate the plan based upon clinical criteria
    
    '''
    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.plan_metrics(my_plan,
                              sol=[sol_no_quad_no_wav, sol_no_quad_with_wav, sol_quad_no_wav, sol_quad_with_wav],
                              sol_names=['no_quad_no_wav', 'no_quad_with_wav', 'quad_no_wav', 'quad_with_wav'])


if __name__ == "__main__":
    ex_wavelet()
