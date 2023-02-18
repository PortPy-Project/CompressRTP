"""

    This example shows creating and modification of wavelet bases for fluence map compression using portpy

"""
import portpy_photon as pp
from portpy.low_dim_rt import LowDimRT

def ex_wavelet():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'.\Data'
    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_id = 'Lung_Patient_1'
    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # creating plan and select among the beams which are 30 degrees apart
    my_plan = pp.Plan(patient_id, data_dir, beam_ids=[0,1,2,3,4,5,6])
    wavelet_basis = LowDimRT.get_low_dim_basis(my_plan.inf_matrix, 'wavelet')
    
if __name__ == "__main__":
    ex_wavelet()