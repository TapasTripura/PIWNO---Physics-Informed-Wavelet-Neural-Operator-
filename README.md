# PIWNO---Physics-Informed-Wavelet-Neural-Operator-
This repository contains the python codes of the paper 
  > + Navaneeth, N., Tripura, T., & Chakraborty, S. (2024). Physics informed WNO. Computer Methods in Applied Mechanics and Engineering, 418, 116546. [Article](https://doi.org/10.1016/j.cma.2023.116546)
  > + Tripura, T., & Chakraborty, S. (2023). Wavelet Neural Operator for solving parametric partial differential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering, 404, 115783. [Article](https://doi.org/10.1016/j.cma.2022.115783)

## Allen-Cahn PDE.
![](/Allen-Cahn/Allen_Cahn.gif)

## Super resolution using Wavelet Neural Operator.
  > Train in low resolution:
  ![Train in Low resolution](/Nagumo/Nagumo.png)
  > Test on high resolution:
  ![Train at high resolution](/Nagumo/Nagumo_super.png)

## Essential Python Libraries
Following packages are required to be installed to run the above codes:
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + [Wavelet Transform Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)
  + [Xarray-Grib reader (To read ERA5 data in section 5)](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html?highlight=install)

Copy all the data in the folder 'data' and place the folder 'data' inside the same mother folder where the codes are present.	Incase, the location of the data are changed, the correct path should be given.


## BibTex
If you use any part our codes, please cite us at,
```
@article{tripura2023wavelet,
  title={Wavelet Neural Operator for solving parametric partial differential equations in computational mechanics problems},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={404},
  pages={115783},
  year={2023},
  publisher={Elsevier}
}

@article{navaneeth2024physics,
  title={Physics informed WNO},
  author={Navaneeth, N and Tripura, Tapas and Chakraborty, Souvik},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={418},
  pages={116546},
  year={2024},
  publisher={Elsevier}
}
```
