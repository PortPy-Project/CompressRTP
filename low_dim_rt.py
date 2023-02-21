import pywt
import numpy as np
from portpy_photon.influence_matrix import InfluenceMatrix

class LowDimRT:
    """
    An incomplete basis representing a low dimensional subspace for fluence map compression

    - **Methods** ::

        :get_low_dim_basis(inf_matrix, compression):
            Get the incomplete basis for dimension reduction

    """
    
    @staticmethod
    def get_low_dim_basis(inf_matrix: InfluenceMatrix, compression: str = 'wavelet'):
        """
        :param inf_matrix: an object of class InfluenceMatrix for the specified plan
        :param compression: the compression method
        :type compression: str
        :return: a dictionary that contains the dimension reduction basis in the format of
        dict: {
                   'beam_ID': list(int),
                   'low_dim_basis': list(array)
                  }
        """
        num_of_beams = len(inf_matrix.beamlets_dict)
        low_dim_basis = dict.fromkeys(range(num_of_beams), [])
        beamlets = inf_matrix.get_bev_2d_grid_in_orig_res(range(num_of_beams))
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
                            low_dim_basis[ind].append(np.concatenate((approximation, horizontal)))
                    beamlet_2d_grid[row][col] = 0
        return low_dim_basis