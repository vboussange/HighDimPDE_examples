#=
Plotting complexity vs error, for different combination of parameters 
    - in terms of random variable evalutations
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

function nb_rand_var_eval_DS(d, N, maxiters, K, batch_size)
    nb_rand_var = 0
    for n in 1:N, m in 1:maxiters, j in 1:batch_size, t in 1:n
        nb_rand_var += d
    end
    for n in 1:N, m in 1:maxiters, j in 1:batch_size, k in 1:K
        nb_rand_var += d
    end
    return nb_rand_var
end

indicator(l) = l < 1 ? 0 : 1
function nb_rand_var_eval_MLP(d, L, M, K)
    nb_rand_var = 0
    if !(L < 0)
        for l in 0:L-1, m in 1:M^(L-l)
            nb_rand_var += d + nb_rand_var_eval_MLP(d, l, m, K) + indicator(l) * nb_rand_var_eval_MLP(d, l-1, m, K)
        end
        for l in 0:L-1, m in 1:M^(L-l), k in 1:K
            nb_rand_var += d + nb_rand_var_eval_MLP(d, l, m, K) + indicator(l) * nb_rand_var_eval_MLP(d, l-1, m, K)
        end
    end
    if !(L < 1)
        for m in 1:M^L
            nb_rand_var += d
        end
    end
    return nb_rand_var
end

# testing nb_rand_var_eval_MLP
nb_rand_var_eval_MLP(10, 2, 2, 5)
# testing nb_rand_var_eval_DS
nb_rand_var_eval_DS(1, 1, 1, 1, 1000)

###########
### DS ####
###########
path_results = "../results/2022-10-04/explo_param_DS_x0_sample_T=0.2/"
dict_results_DS =load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2")
df_all_params_DS = vcat([dict_results_DS[k]["df_ds"] for k in keys(dict_results_DS)]...)
batch_size = 8000
maxiters = 1000
df_all_params_DS["nb_rand_var"] = nb_rand_var_eval_DS.(df_all_params_DS[:, "d"], 
                                                        df_all_params_DS[:,"N"],
                                                        maxiters,
                                                        df_all_params_DS[:, "K"],
                                                         )

###########
### MLP ###
###########
path_results = "../results/2022-10-04/explo_param_MLP_T=0.5_explo_K_M_4/"
dict_results_MLP =load(path_results*"dict_results_MLP_param_explo.jld2")
df_all_params_MLP = vcat([dict_results_MLP[k]["df_ds"] for k in keys(dict_results_MLP)]...)
df_all_params_DS["nb_rand_var"] = nb_rand_var_eval_DS.(df_all_params_MLP[:, "d"], 
                                                        df_all_params_DS[:,"N"],
                                                        maxiters,
                                                        df_all_params_DS[:, "K"],
                                                        batch_size)

#######################################
# Plotting only for a single parameter
#######################################
fig, axs = subplots(1,2, figsize=(6,3))
## time vs error
ax = axs[1]
# DS
data = dict_results_DS["explo_N"]["df_ds"]
ax.errorbar(data[:,6], 
            (data[:,4]), 
            yerr=(data."Std. dev. error"), 
            fmt="o", 
            color="tab:blue",
            ecolor="tab:blue", elinewidth=3, capsize=0, label="ML-based approx. method")
display(fig)
ax.set_xlabel("Time (s)")
ax.set_ylabel(L"$L^1-$approx. error")
# ax.set_xscale("log")
display(fig)
# MLP
data = dict_results_MLP["explo_L"]["df_ds"]
ax = axs[2]
ax.errorbar(data[:,6], 
            data[:,4], 
            yerr=data."Std. dev. error", 
            fmt="o", 
            color="tab:red",
            ecolor="tab:red", elinewidth=3, capsize=0, label="MLP approx. method")
display(fig)
ax.set_xlabel("Time (s)")
# ax.set_ylabel(L"$L^1-$approx. error")
ax.set_xscale("log")
fig.legend(loc="lower center", bbox_to_anchor=(0.55, 1.,))
fig.tight_layout()
display(fig)
fig.savefig("time_vs_l1_err.pdf", dpi=100)

#######################################
# Plotting for all parameters where d and T are similar
#######################################
fig, axs = subplots(1,2, figsize=(6,3))
## time vs error
ax = axs[1]
# DS
data = vcat(dict_results_DS["explo_batch_size"]["df_ds"],dict_results_DS["explo_K"]["df_ds"])
ax.errorbar(data[:,6], 
            (data[:,4]), 
            yerr=(data."Std. dev. error"), 
            fmt="o", 
            color="tab:blue",
            ecolor="tab:blue", elinewidth=3, capsize=0, label="ML-based approx. method")
display(fig)
ax.set_xlabel("Time (s)")
ax.set_ylabel(L"$L^1-$approx. error")
# ax.set_xscale("log")
display(fig)
# MLP
data = df_all_params_MLP
ax = axs[2]
ax.errorbar(data[:,6], 
            data[:,4], 
            yerr=data."Std. dev. error", 
            fmt="o", 
            color="tab:red",
            ecolor="tab:red", elinewidth=3, capsize=0, label="MLP approx. method")
display(fig)
ax.set_xlabel("Time (s)")
# ax.set_ylabel(L"$L^1-$approx. error")
ax.set_xscale("log")
fig.legend(loc="lower center", bbox_to_anchor=(0.55, 1.,))
fig.tight_layout()
display(fig)
fig.savefig("time_vs_l1_err_bis.pdf", dpi=100)


#######################################
# Plotting for all parameters where d and T are similar
# only for MLP method
#######################################
fig, ax = subplots(1, figsize=(3,3))
data = df_all_params_MLP
ax.errorbar(data[:,6], 
            data[:,4], 
            yerr=data."Std. dev. error", 
            fmt="o", 
            color="tab:red",
            ecolor="tab:red", elinewidth=3, capsize=0, label="MLP approx. method")
display(fig)
ax.set_xlabel("Time (s)")
# ax.set_ylabel(L"$L^1-$approx. error")
ax.set_xscale("log")
fig.legend(loc="lower center", bbox_to_anchor=(0.55, 1.,))
fig.tight_layout()
display(fig)
fig.savefig("time_vs_l1_err_MLP.pdf", dpi=100)
