# HighDimPDE_examples
Examples for paper 
Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions

# How to run simulation
All example files `MLP_*.jl` and `DeepSplitting_*.jl` can be run independently.

Similarly, examples can be run together with the custom wrapper `run_example_files.jl`:
```
julia16 --project=. --threads 20 run_example_files.jl &> run_examples_files.stdout
```
The wrapper exports the simulation results in `.tex` files.