# HighDimPDE_examples
This repository contains the code used in the examples of the paper

> Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. [arXiv](https://arxiv.org/abs/2205.03672) (2022), 59 pages.

## How to run simulation
All example files `MLP_*.jl` and `DeepSplitting_*.jl` can be run independently.

Examples can be also run together with `run_example_files_arg_parse.jl`, where the first argument
given corresponds to the CUDA device required, and the second argument corresponds to the 
example and can be "rep_mut", "nonlocal_comp", "nonlocal_sinegordon", "fisherkpp_neumann", "allencahn_neumann"/

## Example
```
julia --project=. --threads 20 run_example_files_arg_parse.jl 1 rep_mut &> rep_mut.stdout
```
The script exports the simulation results in `.tex` files.