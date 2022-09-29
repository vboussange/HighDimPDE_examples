#=
Plotting explo params MLP
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

path_results = "../results/results_rev_T=0.5_2022-09-28/"

dict_results =load(path_results*"dict_results_MLP_param_explo.jld2")

fig, axs = subplots(1,3, figsize = (8,2))

## explo K
ax = axs[1]
scen = "explo_K"
@unpack df_ds, dfu_ds = dict_results[scen]
println(df_ds)
for r in eachrow(df_ds)
    ax.errorbar(r.K, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"K")
ax.set_xticks(df_ds.K)

display(fig)


## explo M
ax = axs[2]
scen = "explo_M"
@unpack df_ds, dfu_ds = dict_results[scen]
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)
    ax.errorbar(r.M, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"M")
ax.set_xticks(df_ds.M)
ax.set_yscale("log")
display(fig)


## explo L
ax = axs[3]
scen = "explo_L"
@unpack df_ds, dfu_ds = dict_results[scen]
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)
    ax.errorbar(r.L, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"L")
ax.set_xticks(df_ds.L)

fig.tight_layout()
display(fig)

fig.savefig(path_results*"/fig_MLP_param_explo.pdf", dpi=100)