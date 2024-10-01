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