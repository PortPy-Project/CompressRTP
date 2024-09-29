from __future__ import annotations

from portpy.photon import Optimization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon.clinical_criteria import ClinicalCriteria
import cvxpy as cp
import numpy as np
from copy import deepcopy
try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    pass
import scipy
try:
    import pywt
except ImportError:
    pass


class CompressRTPOptimization(Optimization):
    """
    Class for Compressed RTP optimization. It is child class of PortPy.Photon Optimization class

    - **Attributes** ::
        :param my_plan: object of class Plan
        :param inf_matrix: object of class InfluenceMatrix
        :param clinical_criteria: object of class ClinicalCriteria
        :param opt_params: dictionary of vmat optimization parameters
        :param vars: dictionary of variables

    :Example:
    >>> opt = CompressRTPOptimization(my_plan=my_plan, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria, opt_params=vmat_opt_params)
    >>> opt.create_cvxpy_problem_compressed(solver='MOSEK', verbose=True)

    - **Methods** ::
        :create_cvxpy_problem_compressed()
            Creates cvxpy problem for solving using compressed data
    """
    def __init__(self, my_plan: Plan, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None):
        # Call the constructor of the base class (Optimization) using super()
        super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                         clinical_criteria=clinical_criteria,
                         opt_params=opt_params, vars=vars)

    def create_cvxpy_problem_compressed(self, S=None, H=None, W=None):

        """
        It runs optimization to create optimal plan based upon clinical criteria

        :param S: sparse influence matrix. Uses influence matrix in my_plan by default
        :param H: tall skinny matrix. It is obtained using SVD of L = A-S. UKV = svd(L, rank=k). H=U
        :param W: thin wide matrix. It is obtained using SVD of L = A-S. W=KV

        """
        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        x = self.vars['x']
        obj = self.obj
        constraints = self.constraints

        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []

        if S is None:
            S = inf_matrix.A
        num_fractions = clinical_criteria.get_num_of_fractions()
        st = inf_matrix
        if W is None and H is None:
            H = np.zeros((S.shape[0], 1))
            W = np.zeros((1, S.shape[1]))

        # Construct optimization problem
        Wx = cp.Variable(H.shape[1])  # creating dummy variable for dose
        # Generating objective functions
        print('Objective Start')
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                    constraints += [S[st.get_opt_voxels_idx(struct), :] @ x + H[st.get_opt_voxels_idx(struct), :] @ Wx <= dose_gy + dO]
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                    constraints += [S[st.get_opt_voxels_idx(struct), :] @ x + H[st.get_opt_voxels_idx(struct), :] @ Wx >= dose_gy - dU]
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (
                            obj_funcs[i]['weight'] * cp.sum_squares(S[st.get_opt_voxels_idx(struct), :] @ x + H[st.get_opt_voxels_idx(struct), :] @ Wx))]
            elif obj_funcs[i]['type'] == 'smoothness-quadratic':
                [Qx, Qy, num_rows, num_cols] = self.get_smoothness_matrix(inf_matrix.beamlets_dict)
                smoothness_X_weight = 0.6
                smoothness_Y_weight = 0.4
                obj += [obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) +
                                                              smoothness_Y_weight * (1 / num_rows) * cp.sum_squares(Qy @ x))]

        print('Objective done')

        print('Constraints Start')

        constraint_def = deepcopy(
            clinical_criteria.get_criteria())  # get all constraints definition using clinical criteria

        # add/modify constraints definition if present in opt params
        for opt_constraint in opt_params_constraints:
            # add constraint
            param = opt_constraint['parameters']
            if param['structure_name'] in my_plan.structures.get_structures():
                criterion_exist, criterion_ind = clinical_criteria.check_criterion_exists(opt_constraint,
                                                                                          return_ind=True)
                if criterion_exist:
                    constraint_def[criterion_ind] = opt_constraint
                else:
                    constraint_def += [opt_constraint]

        # Adding max/mean constraints
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    if org != 'GTV' and org != 'CTV':
                        if org in my_plan.structures.get_structures():
                            if len(st.get_opt_voxels_idx(org)) == 0:
                                continue
                            constraints += [S[st.get_opt_voxels_idx(org), :] @ x + H[st.get_opt_voxels_idx(org), :] @ Wx <= limit / num_fractions]
            elif constraint_def[i]['type'] == 'mean_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
                        limit = limit / fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints += [(1 / sum(st.get_opt_voxels_volume_cc(org))) *
                                        (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(org),
                                                             S[st.get_opt_voxels_idx(org), :] @ x + H[st.get_opt_voxels_idx(org), :] @ Wx))))
                                        <= limit / num_fractions]

        constraints += [Wx == (W @ x)]
        print('Constraints done')

    def get_sparse_plus_low_rank(self, A=None, thresold_perc=1, rank=5):
        """
            :param A: dose influence matrix
            :param thresold_perc: thresold percentage. Default to 1% of max(A)
            :type rank: rank of L = A-S.
            :returns: S, H, W using randomized svd
        """
        if A is None:
            A = deepcopy(self.inf_matrix.A)
        tol = np.max(A) * thresold_perc * 0.01
        # S = S*0
        S = np.where(A > tol, A, 0)
        if rank == 0:
            H = np.zeros((A.shape[0], 1))
            W = np.zeros((1, A.shape[1]))
        else:
            print('Running svd..')
            [U, svd_S, V] = randomized_svd(A - S, n_components=rank + 1, random_state=0)
            print('svd done!')
            H = U[:, :rank]
            W = np.diag(svd_S[:rank]) @ V[:rank, :]
        S = scipy.sparse.csr_matrix(S)
        return S, H, W

    def get_low_dim_basis(self, inf_matrix: InfluenceMatrix = None, compression: str = 'wavelet'):
        """
        :param inf_matrix: an object of class InfluenceMatrix for the specified plan
        :param compression: the compression method
        :type compression: str
        :return: a list that contains the dimension reduction basis in the format of array(float)
        """
        if inf_matrix is None:
            inf_matrix = self.inf_matrix
        low_dim_basis = {}
        num_of_beams = len(inf_matrix.beamlets_dict)
        num_of_beamlets = inf_matrix.beamlets_dict[num_of_beams - 1]['end_beamlet_idx'] + 1
        beam_id = [inf_matrix.beamlets_dict[i]['beam_id'] for i in range(num_of_beams)]
        beamlets = inf_matrix.get_bev_2d_grid(beam_id=beam_id)
        index_position = list()
        for ind in range(num_of_beams):
            low_dim_basis[beam_id[ind]] = []
            for i in range(inf_matrix.beamlets_dict[ind]['start_beamlet_idx'],
                           inf_matrix.beamlets_dict[ind]['end_beamlet_idx'] + 1):
                index_position.append((np.where(beamlets[ind] == i)[0][0], np.where(beamlets[ind] == i)[1][0]))
        if compression == 'wavelet':
            max_dim_0 = np.max([beamlets[ind].shape[0] for ind in range(num_of_beams)])
            max_dim_1 = np.max([beamlets[ind].shape[1] for ind in range(num_of_beams)])
            beamlet_2d_grid = np.zeros((int(np.ceil(max_dim_0 / 2)), int(np.ceil(max_dim_1 / 2))))
            for row in range(beamlet_2d_grid.shape[0]):
                for col in range(beamlet_2d_grid.shape[1]):
                    beamlet_2d_grid[row][col] = 1
                    approximation_coeffs = pywt.idwt2((beamlet_2d_grid, (None, None, None)), 'sym4',
                                                      mode='periodization')
                    horizontal_coeffs = pywt.idwt2((None, (beamlet_2d_grid, None, None)), 'sym4', mode='periodization')
                    for b in range(num_of_beams):
                        if ((2 * row + 1 < beamlets[b].shape[0] and 2 * col + 1 < beamlets[b].shape[1] and
                             beamlets[b][2 * row + 1][2 * col + 1] != -1) or
                                (2 * row + 1 < beamlets[b].shape[0] and 2 * col < beamlets[b].shape[1] and
                                 beamlets[b][2 * row + 1][2 * col] != -1) or
                                (2 * row < beamlets[b].shape[0] and 2 * col + 1 < beamlets[b].shape[1] and
                                 beamlets[b][2 * row][2 * col + 1] != -1) or
                                (2 * row < beamlets[b].shape[0] and 2 * col < beamlets[b].shape[1] and
                                 beamlets[b][2 * row][2 * col] != -1)):
                            approximation = np.zeros(num_of_beamlets)
                            horizontal = np.zeros(num_of_beamlets)
                            for ind in range(inf_matrix.beamlets_dict[b]['start_beamlet_idx'],
                                             inf_matrix.beamlets_dict[b]['end_beamlet_idx'] + 1):
                                approximation[ind] = approximation_coeffs[index_position[ind]]
                                horizontal[ind] = horizontal_coeffs[index_position[ind]]
                            low_dim_basis[beam_id[b]].append(np.transpose(np.stack([approximation, horizontal])))
                    beamlet_2d_grid[row][col] = 0
        for b in beam_id:
            low_dim_basis[b] = np.concatenate(low_dim_basis[b], axis=1)
            u, s, vh = scipy.sparse.linalg.svds(low_dim_basis[b], k=min(low_dim_basis[b].shape[0], low_dim_basis[b].shape[1]) - 1)
            ind = np.where(s > 0.0001)
            low_dim_basis[b] = u[:, ind[0]]
        return np.concatenate([low_dim_basis[b] for b in beam_id], axis=1)