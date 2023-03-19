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
        beamlets = inf_matrix.get_bev_2d_grid(beam_id=beam_id)
        index_position = list()
        num_of_beamlets = inf_matrix.beamlets_dict[num_of_beams - 1]['end_beamlet'] + 1
        for ind in range(num_of_beams):
            for i in range(inf_matrix.beamlets_dict[ind]['start_beamlet'],
                           inf_matrix.beamlets_dict[ind]['end_beamlet'] + 1):
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
                            for ind in range(inf_matrix.beamlets_dict[b]['start_beamlet'],
                                             inf_matrix.beamlets_dict[b]['end_beamlet'] + 1):
                                approximation[ind] = approximation_coeffs[index_position[ind]]
                                horizontal[ind] = horizontal_coeffs[index_position[ind]]
                            low_dim_basis.append(np.stack((approximation, horizontal)))
                    beamlet_2d_grid[row][col] = 0
        low_dim_basis = np.transpose(np.concatenate(low_dim_basis, axis=0))
        u, s, vh = scipy.sparse.linalg.svds(low_dim_basis, k=min(low_dim_basis.shape[0], low_dim_basis.shape[1]) - 1)
        ind = np.where(s > 0.0001)
        return u[:, ind[0]]