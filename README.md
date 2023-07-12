# HighDimPDE_examples
This repository contains the code used in the examples of the paper

> Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. [arXiv](https://arxiv.org/abs/2205.03672) (2022), 59 pages.

- `figure/` contains all scripts to generate the manuscript figures.
- `results/` contains raw simulations results, exported in `.jld2`, `.csv` and `.tex` formats
- `tables/` contains all final tables 
- `MLP_*.jl` specify the example simulations for the MLP method, and can be run independently
- `DeepSplitting_*.jl` specify the example simulations for the machine learning-based method, and can be run independently
- `run_*.jl` are wrapper files to run example simulations with specific meta parameter values.
- `run_*.sh` are files used to run the wrapper files