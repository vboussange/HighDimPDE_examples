#=
    Here we recreate the tex table from the JLD2, 
    adding the relative L1 approx error (not calculated previously) 
=#
cd(@__DIR__)
using JLD2 
using DataFrames

dfu_ds = JLD2.load("rep_mut_ds_x0_sample_L1.jld2", "dfu_ds")

# removing burnin
dfu_ds = dfu_ds[61:end,:]

# removing burnin
dfu_ds_g = groupby(dfu_ds, ["d", "T"])

# calculating relative L1 approx error

# calculating 