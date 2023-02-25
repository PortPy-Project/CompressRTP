"""

    This example shows creating and modification of wavelet bases for fluence map compression using portpy

"""
import portpy_photon as pp
from low_dim_rt import LowDimRT


def ex_wavelet():
    # specify the patient data location
    # you first need to download the patient database from the link provided in the PortPy GitHub page
    data_dir = r'..\data'
    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, ...)
    patient_id = 'Lung_Patient_1'
    # create my_plan object for the planner beams_dict and select among the beams which are 30 degrees apart
    # for the customized beams_dict, you can pass the argument beam_ids
    my_plan = pp.Plan(patient_id, data_dir)
    # run IMRT fluence map optimization using a low dimensional subspace for fluence map compression
    # sol = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK', opt_params={'smoothness_weight': 10})
    sol_low_dim = LowDimRT.run_IMRT_fluence_map_low_dim(my_plan, solver='MOSEK', opt_params={'smoothness_weight': 10})
    sol_low_dim_only = LowDimRT.run_IMRT_fluence_map_low_dim(my_plan, solver='MOSEK',
                                                             opt_params={'smoothness_weight': 0})
    # plot fluence 3D and 2D
    pp.Visualize.plot_fluence_3d(sol=sol, beam_id=0)
    pp.Visualize.plot_fluence_2d(sol=sol, beam_id=0)
    pp.Visualize.plot_fluence_3d(sol=sol_low_dim, beam_id=0)
    pp.Visualize.plot_fluence_2d(sol=sol_low_dim, beam_id=0)
    pp.Visualize.plot_fluence_3d(sol=sol_low_dim_only, beam_id=0)
    pp.Visualize.plot_fluence_2d(sol=sol_low_dim_only, beam_id=0)
    # plot DVH for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    my_plan.plot_dvh(sol=sol, structs=structs)
    my_plan.plot_dvh(sol=sol_low_dim, structs=structs)
    my_plan.plot_dvh(sol=sol_low_dim_only, structs=structs)
    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualize.plot_2d_dose(my_plan, sol=sol, slice_num=40, structs=['PTV'])
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_low_dim, slice_num=40, structs=['PTV'])
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_low_dim_only, slice_num=40, structs=['PTV'])
    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan, sol=sol)
    pp.Visualize.plan_metrics(my_plan, sol=sol_low_dim)
    pp.Visualize.plan_metrics(my_plan, sol=sol_low_dim_only)


if __name__ == "__main__":
    ex_wavelet()
