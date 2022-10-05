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
path_results = "../results/2022-10-04/explo_param_MLP_T=0.5_explo_K_M_4/"

dict_results =load(path_results*"dict_results_MLP_param_explo.jld2")

fig, axs = subplots(2,3, figsize = (8,4), sharex = "col")

## explo K
ax = axs[1,1]
scen = "explo_K"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"M = %$(Int(df_ds.M[1])), L = %$(Int(df_ds.L[1]))")
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
ax = axs[1,2]
scen = "explo_M"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"K = %$(Int(df_ds.K[1])), L = %$(Int(df_ds.L[1]))")
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)
    ax.errorbar(r.M, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"M")
ax.set_xticks(df_ds.M)
# ax.set_yscale("log")
ax = axs[2,2]
for r in eachrow(df_ds)
    ax.errorbar(r.M, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_yscale("log")
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"M")
ax.set_xticks(df_ds.M)
display(fig)


## explo L
ax = axs[1,3]
scen = "explo_L"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"M = %$(Int(df_ds.M[1])), K = %$(Int(df_ds.K[1]))")
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)
    ax.errorbar(r.L, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"L")
ax.set_xticks(df_ds.L)
ax = axs[2,3]
for r in eachrow(df_ds)
    ax.errorbar(r.L, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"L")
ax.set_xticks(df_ds.L)
ax.set_yscale("log")
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = 0.1:0.1:1, numticks = 10)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout()
display(fig)

fig.savefig(path_results*"/fig_MLP_param_explo.pdf", dpi=100)