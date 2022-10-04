#=
Plotting explo params DS
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

path_results = "../results/2022-09-29/explo_param_DS_T=0.2/"

dict_results =load(path_results*"dict_results_DeepSplitting_param_explo.jld2")

fig, axs = subplots(2,3, 
                    figsize = (8,4), 
                    sharex = "col", 
                    # sharey=true
                    )

## explo K
ax = axs[1,1]
scen = "explo_K"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), N = %$(Int(df_ds.N[1]))")
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
# ax.set_yscale("log")
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"K")
ax.set_xticks(df_ds.K)
ax.set_ylim(0,10)
display(fig)


## explo batch_size
ax = axs[1,2]
scen = "explo_batch_size"
@unpack df_ds = dict_results[scen]
ax.set_title(L"K = %$(Int(df_ds.K[1])), N = %$(Int(df_ds.N[1]))")
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)[2:2:end]
    ax.errorbar(r.batch_size, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"J_m")
ax = axs[2,2]
for r in eachrow(df_ds)[2:2:end]
    ax.errorbar(r.batch_size, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"J_m")
ax.set_xticks(df_ds.batch_size[2:2:end])
# ax.set_yscale("log")
ax.set_ylim(0,10)
display(fig)


## explo N
ax = axs[1,3]
scen = "explo_N"
@unpack df_ds = dict_results[scen]
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), K = %$(Int(df_ds.K[1]))")
println(df_ds[2:end,:])
# fig, ax = subplots(1)
for r in eachrow(df_ds[2:end,:])
    ax.errorbar(r.N, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"N")
# ax.set_yscale("log")
ax = axs[2,3]
for r in eachrow(df_ds[2:end,:])
    ax.errorbar(r.N, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_ylabel("avg. runtime (s)")
ax.set_xlabel(L"N")
ax.set_xticks(df_ds.N)
# ax.set_yscale("log")
display(fig)

# ax.set_ylim(0,0.1)
fig.tight_layout()
display(fig)

# fig.savefig(path_results*"/fig_DS_param_explo.pdf", dpi=100)