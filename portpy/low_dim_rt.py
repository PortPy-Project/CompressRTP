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
                   'wavelet_basis': list(float)
                  }
        """
        low_dim_basis = dict()
        if compression == 'wavelet':
            for ind in range(len(inf_matrix.beamlets_dict)):
                    index = 0
                    beamlets = inf_matrix.create_beamlet_idx_2d_grid(ind)
                    x = int(np.ceil(beamlets.shape[0]/2));
                    y = int(np.ceil(beamlets.shape[1]/2));
                    wavelet_basis_approximation = np.zeros((x*y, 4*x*y))
                    wavelet_basis_horizontal = np.zeros_like(wavelet_basis_approximation)
                    for row in range(x):
                        for col in range(y):
                            if beamlets[2*row-1][2*col-1] != -1:
                                beamlet_2d_grid = np.zeros((x, y))
                                beamlet_2d_grid[row][col] = 1;
                                approximation_coeffs = pywt.idwt2((beamlet_2d_grid, (None, None, None)), 'haar', mode='symmetric')
                                horizontal_coeffs = pywt.idwt2((None, (beamlet_2d_grid, None, None)), 'haar', mode='symmetric')
                                wavelet_basis_approximation[index] = np.concatenate(approximation_coeffs)
                                wavelet_basis_horizontal[index] = np.concatenate(horizontal_coeffs)
                                index += 1
                    low_dim_basis[ind] = np.concatenate([wavelet_basis_approximation, wavelet_basis_horizontal])
        return low_dim_basis