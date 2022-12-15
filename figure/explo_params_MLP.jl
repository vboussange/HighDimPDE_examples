#=
Plotting explo params MLP
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

# path_results = "../results/results_rev_T=0.5_2022-09-28/"
# path_results = "../results/2022-10-04/explo_param_MLP_T=0.5_explo_K_M_4/"
path_results = "../results/2022-12-12/explo_param_MLP_T=0.5_with_K_explo_uniform_sampling_1e0/"
dict_results =load(path_results*"dict_results_MLP_param_explo_Kexplo_L_cpu_5.jld2")

fig, axs = subplots(2,2, figsize = (6,4), sharex = "col")
[ax.set_yscale("log") for ax in axs[1,:]]

## explo K
ax = axs[1,1]
scen = "explo_K"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"M = n = %$(Int(df_ds.L[1])), T = %$(df_ds.T[1])")
println(df_ds)
for r in eachrow(df_ds)
    ax.errorbar(r.K, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"K")

ax = axs[2,1]
for r in eachrow(df_ds)
    ax.errorbar(r.K, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_yscale("log")
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"K")
ax.set_xticks(df_ds.K)
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = 0.1:0.1:1, numticks = 10)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
display(fig)


## explo M
# ax = axs[1,3]
# scen = "explo_M"
# @unpack df_ds, dfu_ds = dict_results[scen]
# ax.set_title(L"K = %$(Int(df_ds.K[1])), n = %$(Int(df_ds.L[1]))")
# println(df_ds)
# # fig, ax = subplots(1)
# for r in eachrow(df_ds)
#     ax.errorbar(r.M, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
# end
# ax.set_ylabel(L"$L^1$-approx. error")
# ax.set_xlabel(L"M")
# ax.set_xticks(df_ds.M)
# # ax.set_yscale("log")
# ax = axs[2,3]
# for r in eachrow(df_ds)
#     ax.errorbar(r.M, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
# end
# ax.set_yscale("log")
# ax.set_ylabel("avg. runtime (s)")
# ax.set_xlabel(L"M")
# ax.set_xticks(df_ds.M)
# display(fig)


## explo L
path_results = "../results/2022-12-14/explo_param_MLP_T=0.2_uniform_sampling_1e0/"
dict_results =load(path_results*"dict_results_MLP_param_explo_L_cpu_5.jld2")
ax = axs[1,2]
scen = "explo_L"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"M = n, K = %$(Int(df_ds.K[1])), T = %$(df_ds.T[1])")
println(df_ds)
# fig, ax = subplots(1)
# yerr = hcat(log.(df_ds."\$L^1-\$approx. error") .- log.(df_ds."Std. dev. error"), log.(df_ds."\$L^1-\$approx. error") .+ log.(df_ds."Std. dev. error")) .|> exp
# for (i,r) in enumerate(eachrow(df_ds))
#     ax.errorbar(r.L, r."\$L^1-\$approx. error", yerr = yerr[i:i,:]', c = "tab:blue", fmt = "o", ms = 4)
# end
for r in eachrow(df_ds)
    ax.errorbar(r.L, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"n")
ax.set_xticks(df_ds.L)
ax.set_yscale("log")
ax.set_ylim(1e-4,1e-1,)
display(fig)

ax = axs[2,2]
for r in eachrow(df_ds)
    ax.errorbar(r.L, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"n")
ax.set_xticks(df_ds.L)
ax.set_yscale("log")
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = 0.1:0.1:1, numticks = 10)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout()
display(fig)

fig.savefig(path_results*"/fig_MLP_param_explo.pdf", dpi=100)