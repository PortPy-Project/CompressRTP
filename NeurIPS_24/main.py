import portpy.photon as pp
import algorithms
import numpy as np
import math
import matplotlib.pyplot as plt

def objective_function_value(x):
    obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
    obj = 0
    for i in range(len(obj_funcs)):
        if obj_funcs[i]['type'] == 'quadratic-overdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                    continue
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dO = np.maximum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dO ** 2))
        elif obj_funcs[i]['type'] == 'quadratic-underdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dU = np.minimum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dU ** 2))
        elif obj_funcs[i]['type'] == 'quadratic':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum((A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x) ** 2))
        elif obj_funcs[i]['type'] == 'smoothness-quadratic':
            [Qx, Qy, num_rows, num_cols] = opt.get_smoothness_matrix(inf_matrix.beamlets_dict)
            smoothness_X_weight = 0.6
            smoothness_Y_weight = 0.4
            obj += obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * np.sum((Qx @ x) ** 2) +
                                                    smoothness_Y_weight * (1 / num_rows) * np.sum((Qy @ x) ** 2))
    print("objective function value:", obj)

def l2_norm(matrix):
    values, vectors = np.linalg.eig(np.transpose(matrix) @ matrix)
    return math.sqrt(np.max(np.abs(values)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method', type=str, choices=['Naive', 'AHK06', 'AKL13', 'DZ11', 'RMR'], help='The name of method.'
    )
    parser.add_argument(
        '--patient', type=str, help='Patient\'s name'
    )
    parser.add_argument(
        '--threshold', type=float, help='The threshold using for the input of algorithm.'
    )
    parser.add_argument(
        '--solver', type=str, default='SCS', help='The name of solver for solving the optimization problem'
    )

    args = parser.parse_args()
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir='')
    # Pick a patient
    data.patient_id = args.patient
    # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    # Pick a protocol
    protocol_name = 'Lung_2Gy_30Fx'
    # Load clinical criteria for a specified protocol
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
    # Load hyper-parameter values for optimization problem for a specified protocol
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # Create optimization structures (i.e., Rinds)
    structs.create_opt_structures(opt_params=opt_params)
    # create plan_full object by specifying load_inf_matrix_full=True
    beams_full = pp.Beams(data, load_inf_matrix_full=True)
    # load influence matrix based upon beams and structure set
    inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
    plan_full = pp.Plan(ct, structs, beams, inf_matrix_full, clinical_criteria)
    # Load influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    opt_full = pp.Optimization(plan_full, opt_params=opt_params)
    opt_full.create_cvxpy_problem()

    A = inf_matrix_full.A
    print("number of non-zeros of the original matrix: ", len(A.nonzero()[0]))
    
    method = getattr(algorithms, args.method)
    S = method(A, args.threshold)
    print("number of non-zeros of the sparsed matrix: ", len(S.nonzero()[0]))
    print("relative L2 norm (%): ", l2_norm(A - S) / l2_norm(A) * 100)

    inf_matrix.A = S
    plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)
    opt = pp.Optimization(plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    x = opt.solve(solver=args.solver, verbose=False)

    opt_full.vars['x'].value = x['optimal_intensity']
    violation = 0
    for constraint in opt_full.constraints[2:]:
        violation += np.sum(constraint.violation())
    print("feasibility violation:", violation)
    objective_function_value(x['optimal_intensity'])

    dose_1d = S @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    dose_full = A @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    print("relative dose discrepancy (%): ", (np.linalg.norm(dose_full - dose_1d) / np.linalg.norm(dose_full)) * 100)

    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # Turn on norm flag for same normalization for sparse and full dose.
    ax = pp.Visualization.plot_dvh(plan, dose_1d=dose_1d , struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(plan_full, dose_1d=dose_full, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    plt.savefig(str(args.method) + "_" + str(args.threshold) + "_" + str(args.patient) + ".pdf")
