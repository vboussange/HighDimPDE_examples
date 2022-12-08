#=
Plotting complexity vs error, for different combination of parameters
- Complexity is measured in terms of time or in terms of random variable evalutations
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

###########
### DS ####
###########
path_results = "../results/2022-10-04/explo_param_DS_x0_sample_T=0.2/"
dict_results_DS =load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2")
df_all_params_DS = vcat([dict_results_DS[k]["df_ds"] for k in keys(dict_results_DS)]...)

###########
### MLP ###
###########
path_results = "../results/2022-10-04/explo_param_MLP_T=0.5_explo_K_M_4/"
dict_results_MLP =load(path_results*"dict_results_MLP_param_explo.jld2")
df_all_params_MLP = vcat([dict_results_MLP[k]["df_ds"] for k in keys(dict_results_MLP)]...)

fig, axs = subplots(1,2)

## time vs error
ax = axs[1]
# DS
data = dict_results_DS["explo_N"]["df_ds"]
ax.errorbar(data[:,6], 
            (data[:,4]), 
            yerr=(data."Std. dev. error"), 
            fmt="o", 
            color="tab:blue",
            ecolor="tab:blue", elinewidth=3, capsize=0)
display(fig)
ax.set_xlabel("Time (s)")
ax.set_ylabel(L"$L^1-$approx. error")
# ax.set_xscale("log")
display(fig)
# MLP
ax = axs[2]
ax.errorbar(df_all_params_MLP[:,6], 
            log.(df_all_params_MLP[:,4]), 
            yerr=log.(df_all_params_MLP."Std. dev. error"), 
            fmt="o", 
            color="tab:red",
            ecolor="tab:red", elinewidth=3, capsize=0)
display(fig)
ax.set_xlabel("Time (s)")
ax.set_ylabel(L"$L^1-$approx. error")
ax.set_xscale("log")
display(fig)
