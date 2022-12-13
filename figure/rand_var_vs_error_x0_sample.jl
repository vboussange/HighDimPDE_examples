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
compute_nb_rand_var = false

function nb_rand_var_eval_DS(d, N, maxiters, K, batch_size)
    nb_rand_var = 0
    for n in 1:N, t in 1:n
        nb_rand_var += d * maxiters * batch_size
    end
    nb_rand_var += d * N * maxiters * batch_size * K
    return nb_rand_var
end

indicator(l) = l < 1 ? 0 : 1
function nb_rand_var_eval_MLP(d, L, M, K)
    nb_rand_var = 0
    for l in 0:L-1, m in 1:M^(L-l)
        nb_rand_var += d + nb_rand_var_eval_MLP(d, l, M, K) + indicator(l) * nb_rand_var_eval_MLP(d, l-1, M, K)
    end
    for l in 0:L-1, m in 1:M^(L-l), k in 1:K
        nb_rand_var += d + nb_rand_var_eval_MLP(d, l, M, K) + indicator(l) * nb_rand_var_eval_MLP(d, l-1, M, K)
    end
    if !(L < 1)
        nb_rand_var += M^L * d
    end
    return nb_rand_var
end

# testing nb_rand_var_eval_MLP
nb_rand_var_eval_MLP(10, 1, 1, 9)
# testing nb_rand_var_eval_DS
nb_rand_var_eval_DS(10, 10, 1000, 1, 1)

###########
### DS ####
###########
# path_results = "../results/2022-10-04/explo_param_DS_x0_sample_T=0.2/"
# dict_results_DS =load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2")
# df_all_params_DS = vcat([dict_results_DS[k]["df_ds"] for k in keys(dict_results_DS)]...)
# if compute_nb_rand_var
#     # for calculating `nb_rand_var`
#     batch_size = 8000
#     maxiters = 1000
#     for k in keys(dict_results_DS)
#         df = dict_results_DS[k]["df_ds"]
#         println(df)
#         df[!,"nb_rand_var"] = nb_rand_var_eval_DS.(df[:, "d"], 
#                                                     df[:,"N"],
#                                                     maxiters,
#                                                     df[:, "K"],
#                                                     df[:, "batch_size"],)
#     end
#     save(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2", dict_results_DS)
# end
# df_all_params_DS = vcat([dict_results_DS[k]["df_ds"] for k in keys(dict_results_DS)]...)

scen = "explo_batch_size"
path_results = "../results/2022-12-12/explo_param_DS_x0_sample_T=0.2_U_1e0/"
df_all_params_DS = load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2", scen)["df_ds"]
batch_size = 8000
maxiters = 1000

df_all_params_DS[!,"nb_rand_var"] = nb_rand_var_eval_DS.(df_all_params_DS[:, "d"], 
                                            df_all_params_DS[:,"N"],
                                            maxiters,
                                            df_all_params_DS[:, "K"],
                                            df_all_params_DS[:, "batch_size"],)

###########
### MLP ###
###########
# path_results = "../results/2022-10-04/explo_param_MLP_T=0.5_explo_K_M_4/"
# dict_results_MLP =load(path_results*"dict_results_MLP_param_explo.jld2")
# if compute_nb_rand_var
#     for k in keys(dict_results_MLP)
#         df = dict_results_MLP[k]["df_ds"]

#         df[!,"nb_rand_var"] = nb_rand_var_eval_MLP.(df[:, "d"], 
#                                                     df[:,"L"],
#                                                     df[:,"M"],
#                                                     df[:, "K"])
#     end
#     save(path_results*"dict_results_MLP_param_explo.jld2", dict_results_MLP)
# end
# df_all_params_MLP = vcat([dict_results_MLP[k]["df_ds"] for k in keys(dict_results_MLP)]...)

scen = "explo_L"
path_results = "../results/2022-12-12/explo_param_MLP_T=0.5_with_K_explo_uniform_sampling_1e0/"
df_all_params_MLP = load(path_results*"dict_results_MLP_param_explo_Kexplo_L_cpu_5.jld2", scen)["df_ds"]

df_all_params_MLP[!,"nb_rand_var"] = nb_rand_var_eval_MLP.(df_all_params_MLP[:, "d"], 
                                                            df_all_params_MLP[:,"L"],
                                                            df_all_params_MLP[:,"M"],
                                                            df_all_params_MLP[:, "K"])

#######################################
# Plotting only for a single parameter
#######################################
fig, ax = subplots(1, figsize=(4,4))
## time vs error
# ax = axs[1]
# DS
data = df_all_params_DS
ax.errorbar(data.nb_rand_var, 
            (data."\$L^1-\$approx. error"), 
            yerr=(data."Std. dev. error"), 
            fmt="o", 
            color="tab:blue",
            ecolor="tab:blue", elinewidth=3, capsize=0, label="ML-based approx. method")
display(fig)
ax.set_xlabel("Number of random variables")
ax.set_ylabel(L"$L^1-$approx. error")
# ax.set_xscale("log")
display(fig)
# MLP
data = df_all_params_MLP
# ax = axs[2]
ax.errorbar(data.nb_rand_var, 
            data."\$L^1-\$approx. error", 
            yerr=data."Std. dev. error", 
            fmt="o", 
            color="tab:red",
            ecolor="tab:red", elinewidth=3, capsize=0, label="MLP approx. method")
display(fig)
# ax.set_xlabel("Time (s)")
# ax.set_ylabel(L"$L^1-$approx. error")
ax.set_xscale("log")
ax.set_yscale("log")
fig.legend(loc="lower center", bbox_to_anchor=(0.55, 0.2,))
fig.tight_layout()
display(fig)
fig.savefig("nb_rand_var_vs_l1_err_x0_sample.pdf", dpi=100)