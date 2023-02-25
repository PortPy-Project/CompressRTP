import pywt
import time
import scipy
import numpy as np
import cvxpy as cp
from portpy_photon.plan import Plan
from portpy_photon.optimization import Optimization
from portpy_photon.influence_matrix import InfluenceMatrix

class LowDimRT:
    """
    An incomplete basis representing a low dimensional subspace for fluence map compression

    - **Methods** ::

        :get_low_dim_basis(inf_matrix, compression):
            Get the incomplete basis for dimension reduction
        :run_IMRT_fluence_map_low_dim
            Optimization method for creating and optimizing the IMRT treatment plan

    """
    
    @staticmethod
    def get_low_dim_basis(inf_matrix: InfluenceMatrix, compression: str = 'wavelet'):
        """
        :param inf_matrix: an object of class InfluenceMatrix for the specified plan
        :param compression: the compression method
        :type compression: str
        :return: a list that contains the dimension reduction basis in the format of array(float)
        """
        num_of_beams = len(inf_matrix.beamlets_dict)
        low_dim_basis = list()
        beam_id = [inf_matrix.beamlets_dict[i]['beam_id'] for i in range(num_of_beams)]
        beamlets = inf_matrix.get_bev_2d_grid_in_orig_res(beam_id=beam_id)
        index_position = list()
        num_of_beamlets = inf_matrix.beamlets_dict[num_of_beams-1]['end_beamlet'] + 1
        for ind in range(num_of_beams):
            for i in range(inf_matrix.beamlets_dict[ind]['start_beamlet'], inf_matrix.beamlets_dict[ind]['end_beamlet'] + 1):
                index_position.append((np.where(beamlets[ind]==i)[0][0],np.where(beamlets[ind]==i)[1][0]))          
        if compression == 'wavelet':
            max_dim_0 = np.max([beamlets[ind].shape[0] for ind in range(num_of_beams)])
            max_dim_1 = np.max([beamlets[ind].shape[1] for ind in range(num_of_beams)])
            beamlet_2d_grid = np.zeros((int(np.ceil(max_dim_0/2)), int(np.ceil(max_dim_1/2))))
            for row in range(beamlet_2d_grid.shape[0]):
                for col in range(beamlet_2d_grid.shape[1]):
                    beamlet_2d_grid[row][col] = 1
                    approximation_coeffs = pywt.idwt2((beamlet_2d_grid, (None, None, None)), 'sym4', mode='periodization')
                    horizontal_coeffs = pywt.idwt2((None, (beamlet_2d_grid, None, None)), 'sym4', mode='periodization')
                    for ind in range(num_of_beams):
                        if 2*row-1 < beamlets[ind].shape[0] and 2*col-1 < beamlets[ind].shape[1] and beamlets[ind][2*row-1][2*col-1] != -1:
                            approximation = np.zeros(num_of_beamlets)
                            horizontal = np.zeros(num_of_beamlets)
                            for i in range(inf_matrix.beamlets_dict[ind]['start_beamlet'], inf_matrix.beamlets_dict[ind]['end_beamlet'] + 1):
                                approximation[i] = approximation_coeffs[index_position[i]]
                                horizontal[i] = horizontal_coeffs[index_position[i]]
                            low_dim_basis.append(np.stack((approximation, horizontal)))
                    beamlet_2d_grid[row][col] = 0
        return np.transpose(np.concatenate(low_dim_basis, axis=0))
    
    @staticmethod
    def run_IMRT_fluence_map_low_dim(my_plan: Plan, inf_matrix: InfluenceMatrix = None, solver: str = 'MOSEK',
                                   verbose: bool = True, cvxpy_options: dict = None, **opt_params) -> dict:
        """
        It runs optimization to create optimal plan based upon clinical criteria
        :param my_plan: object of class Plan
        :param inf_matrix: object of class InfluenceMatrix
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers
        :param verbose: Default to True. If set to False, it will not print the solver iterations.
        :param cvxpy_options: cvxpy and the solver settings
        :param opt_params: optimization parameters for modifying parameters of problem statement
        :return: returns the solution dictionary in format of:
            dict: {
                   'optimal_intensity': list(float),
                   'dose_1d': list(float),
                   'inf_matrix': Pointer to object of InfluenceMatrix }
                  }
        :Example:
        >>> Optimization.run_IMRT_fluence_map_CVXPy(my_plan=my_plan,inf_matrix=inf_matrix,solver='MOSEK')
        """

        if cvxpy_options is None:
            cvxpy_options = dict()

        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy, num_rows, num_cols] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict[
            'name']:  # check if rind is already created. If yes, skip rind creation
            Optimization.create_rinds(my_plan, size_mm=[5, 5, 20, 30, 500])
            Optimization.set_rinds_opt_voxel_idx(my_plan,
                                                 inf_matrix=inf_matrix)  # rind_0 is 5mm after PTV, rind_2 is 5 mm after rind_1, and so on..
        else:
            Optimization.set_rinds_opt_voxel_idx(my_plan, inf_matrix=inf_matrix)

        rind_max_dose_perc = [1.1, 1.05, 0.9, 0.85, 0.75]
        for i, rind in enumerate(rinds):  # Add rind constraints
            parameters = {'structure_name': rind}
            total_pres = cc_dict['pres_per_fraction_gy'] * cc_dict['num_of_fractions']
            constraints = {'limit_dose_gy': total_pres * rind_max_dose_perc[i]}
            my_plan.clinical_criteria.add_criterion(criterion='max_dose', parameters=parameters,
                                                    constraints=constraints)

        # # setting weights for oar objectives
        all_vox = np.arange(A.shape[0])
        oar_voxels = all_vox[~np.isin(np.arange(A.shape[0]), st.get_opt_voxels_idx('PTV'))]
        oar_weights = np.ones(A[oar_voxels, :].shape[0])
        if cc_dict['disease_site'] == 'Prostate':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RECT_WALL')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('BLAD_WALL')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3
        elif cc_dict['disease_site'] == 'Lung':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 10
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # Form objective.
        ptv_overdose_weight = opt_params['ptv_overdose_weight'] if 'ptv_overdose_weight' in opt_params else 10000
        ptv_underdose_weight = opt_params['ptv_underdose_weight'] if 'ptv_underdose_weight' in opt_params else 100000
        smoothness_weight = opt_params['smoothness_weight'] if 'smoothness_weight' in opt_params else 10
        total_oar_weight = opt_params['total_oar_weight'] if 'total_oar_weight' in opt_params else 10
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [(1 / len(st.get_opt_voxels_idx('PTV'))) * (
                ptv_overdose_weight * cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (
                       smoothness_X_weight * (1/num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (1/num_rows) * cp.sum_squares(Qy @ x)),
               total_oar_weight * (1 / A[oar_voxels, :].shape[0]) * cp.sum_squares(
                   cp.multiply(cp.sqrt(oar_weights), A[oar_voxels, :] @ x))]

        print('Objective done')
        print('Constraints Start')
        constraints = []
        # constraints += [wMean == cp.sum(w)/w.shape[0]]
        for i in range(len(criteria)):
            if 'max_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    if org != 'GTV' or org != 'CTV':
                        constraints += [A[st.get_opt_voxels_idx(org), :] @ x <= limit / num_fractions]
            elif 'mean_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    constraints += [(1 / sum(st.get_opt_voxels_size(org))) *
                                    (cp.sum((cp.multiply(st.get_opt_voxels_size(org), A[st.get_opt_voxels_idx(org),
                                                                                      :] @ x)))) <= limit / num_fractions]

        # Step 1 and 2 constraint
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        # creating the wavelet incomplete basis representing a low dimensional subspace for dimension reduction
        wavelet_basis = LowDimRT.get_low_dim_basis(inf_matrix, 'wavelet')
        u, s, vh = scipy.sparse.linalg.svds(wavelet_basis, k=int(np.ceil(A.shape[1]/2.5)))
        # Smoothness Constraint
        y = cp.Variable(u.shape[1])
        constraints += [u @ y == x]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)

        print('Problem loaded')
        prob.solve(solver=solver, verbose=verbose, **cvxpy_options)
        print("optimal value with {}:{}".format(solver, prob.value))
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to the solution dictionary
        sol = {'optimal_intensity': x.value.astype('float32'), 'inf_matrix': inf_matrix}

        return sol