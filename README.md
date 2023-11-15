<h1 align="center">
  <img src="./images/LowDimRT.png" width=70% height="40%">
</h1>

# LowDimRT

#### Note: The package is at its early stages of development. At this point, we're only including our recent work [1] where we used wavelets to induce fluence smoothness. This work was originally written in Matlab.

Low-dimensional radiation therapy (LowDimRT) is a python package that aims to leverage and re-purpose the dimensionality reduction tools to improve the speed and quality of the radiation treatment planning. The dimensionality reduction has a rich history in statistics and has gained attention recently and re-emerged as a powerful tool to deal with the increasingly dimensional problems arising in the fields of big-data and machine learning. They also have made profound impacts in imaging science where the high-dimensional images are often lying in a low-dimensional subspace. 

The optimization problems arising in radiation therapy also suffer from the curse of dimensionality with so many radiotherapy machine parameters to be optimized (e.g., beamlets, MLC leaf positions, beam angles) and the discretization of the patient body to many small 3-dimensional cubes (known as voxels). This is an on-going project and at this point we only included the work based on our recent publication [1] which uses low-frequency wavelets to represent and characterize the beamlet intensities in order to induce fluence smoothness in intensity modulated radiation therapy (IMRT).  

The key dimensionality reduction idea in the wavelet-induced smoothness project is that the beamlet intensity maps are lying in a low-dimensional subspace of all possible intensity maps since they  need to be smooth (to reduce the plan complexity, an unnecessary modulation, and improve the plan delivery efficiency). We propose representing the beamlet intensities using an incomplete wavelet basis of low-frequency wavelets that explicitly excludes fluctuating intensity maps from the decision space (explicit hard constraint). This technique provides a built-in wavelet-induced smoothness and excludes complex and clinically irrelevant radiation plans from the search space that improves both dosimetric plan quality and delivery efficiency. 

From the implementation perspective, the current tool is an add-on for the PortPy project. However, the proposed technique can be easily integrated with any optimization approach by simply adding a set of linear constraints in the form of (_x=Wy_), where _x_ represents the beamlet intensity, _W_ is the matrix including low-frequency wavelets (can be generated using method `get_low_dim_basis` in the code), and _y_ is a free variable.

## License
LowDimRT code is distributed under **Apache License 2.0 with Commons Clause**, and is available for non-commercial academic purposes.

## Team
1. [Mojtaba Tefagh](https://mtefagh.github.io/) ([Sharif University of Technology](https://en.sharif.edu/))
2. [Masoud Zarepisheh](https://masoudzp.github.io/) ([Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/))
3. [Gourav Jhanwar](https://github.com/gourav3017) ([Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/))

## Reference 
If you find our work useful in your research or if you use parts of this code, please cite the following paper:
```
@article{tefagh2023built,
  title={Built-in wavelet-induced smoothness to reduce plan complexity in intensity modulated radiation therapy (IMRT)},
  author={Tefagh, Mojtaba and Zarepisheh, Masoud},
  journal={Physics in Medicine \& Biology},
  volume={68},
  number={6},
  pages={065013},
  year={2023},
  publisher={IOP Publishing}
}
```

