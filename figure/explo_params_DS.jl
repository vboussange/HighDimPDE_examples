#=
Plotting explo params DS
=#
cd(@__DIR__)
using HighDimPDE
using PyPlot, PyCall
using JLD2
using UnPack
using DataFrames

# path_results = "../results/2022-10-04/explo_param_DS_x0_sample_T=0.2_L1_err/"
# path_results = "../results/2022-10-04/explo_param_DS_x0_sample_T=0.2/"
# path_results = "../results/2022-10-04/explo_param_DS_T=0.2/"
path_results = "../results/2022-12-12/explo_param_DS_x0_sample_T=0.2_U_1e0/" # T = 1, U=1

dict_results =load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2")
# dict_results =load(path_results*"dict_results_DeepSplitting_param_explo.jld2")

fig = plt.figure(
            # constrained_layout=true,
            figsize = (10,10)
            )
outer = fig.add_gridspec(2,1, hspace = .3)
gs1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec = outer[1], hspace = 0.4, wspace=0.7)
gs2 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec = outer[2], hspace = 0.4, wspace=0.7)

ax1 = fig.add_subplot(py"$(gs1)[0,0:2]")
ax2 = fig.add_subplot(py"$(gs1)[0,2:4]")
ax3 = fig.add_subplot(py"$(gs1)[0,4:6]")
ax1b = fig.add_subplot(py"$(gs1)[1,0:2]")
ax2b = fig.add_subplot(py"$(gs1)[1,2:4]")
ax3b = fig.add_subplot(py"$(gs1)[1,4:6]")

ax4 = fig.add_subplot(py"$(gs2)[0,1:3]")
ax5 = fig.add_subplot(py"$(gs2)[0,3:5]")
ax4b = fig.add_subplot(py"$(gs2)[1,1:3]")
ax5b = fig.add_subplot(py"$(gs2)[1,3:5]")
display(fig)

axs = [(ax1,ax1b), (ax2,ax2b), (ax3,ax3b), (ax4,ax4b), (ax5,ax5b)]

[ax[1].set_yscale("log") for ax in axs[1,:]]
axs[1][1].set_ylim(1e-2,2e0)

## explo K
ax, axb = axs[1]
scen = "explo_K"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), N = %$(Int(df_ds.N[1])),  T = %$(df_ds.T[1])")
println(df_ds)
for r in eachrow(df_ds)
    ax.errorbar(r.K, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"K")

for r in eachrow(df_ds)
    axb.errorbar(r.K, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
ax.set_yscale("log")
axb.set_ylabel("avg. runtime (s)")
axb.set_xlabel(L"K")
axb.set_xticks(df_ds.K)
axb.set_ylim(0,50)
# ax.set_ylim(0,10)
display(fig)


## explo batch_size
ax, axb = axs[2]
scen = "explo_batch_size"
@unpack df_ds = dict_results[scen]
ax.set_title(L"K = %$(Int(df_ds.K[1])), N = %$(Int(df_ds.N[1])), T = %$(df_ds.T[1])")
println(df_ds)
# fig, ax = subplots(1)
for r in eachrow(df_ds)
    ax.errorbar(r.batch_size, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
# ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"J_m")
ax.set_yscale("log")
ax.set_xscale("log")

for r in eachrow(df_ds)
    axb.errorbar(r.batch_size, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
# axb.set_ylabel("avg. runtime (s)")
axb.set_xlabel(L"J_m")
axb.set_xticks(df_ds.batch_size)
axb.set_xscale("log")
axb.set_ylim(0,50)
# ax.set_ylim(0,10)
display(fig)


## explo N
ax, axb = axs[3]
scen = "explo_N"
path_results = "../results/2022-12-13/explo_param_DS_x0_sample_T=0.2_U_1e0/"
df_ds = load(path_results*"dict_resultsparam_explo_DS_x0_sample.jld2", scen)["df_ds"] # T = 1
# @unpack df_ds = dict_results[scen]
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), K = %$(Int(df_ds.K[1])), T = %$(df_ds.T[1])")
println(df_ds[2:end,:])
# fig, ax = subplots(1)
for r in eachrow(df_ds[:,:])
    ax.errorbar(r.N, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
# ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel(L"N")
# ax.set_yscale("log")

for r in eachrow(df_ds[:,:])
    axb.errorbar(r.N, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
# axb.set_ylabel("avg. runtime (s)")
axb.set_xlabel(L"N")
axb.set_xticks(df_ds.N)
# axb.set_yscale("log")
display(fig)

ax, axb = axs[4]
scen = "explo_hls"
@unpack df_ds, dfu_ds = dict_results[scen]
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), N = %$(Int(df_ds.N[1])), K = %$(Int(df_ds.K[1])),  T = %$(df_ds.T[1])")
println(df_ds)
for r in eachrow(df_ds)
    ax.errorbar(r.hls, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel("Hidden layer size")
ax.set_yscale("log")


for r in eachrow(df_ds)
    axb.errorbar(r.hls, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
# ax.set_yscale("log")
axb.set_ylabel("avg. runtime (s)")
axb.set_xlabel("Hidden layer size")
axb.set_xticks(df_ds.hls)
axb.set_ylim(0,50)
display(fig)

ax, axb = axs[5]
scen = "explo_nhlayers"
@unpack df_ds, dfu_ds = dict_results[scen]
df_ds.nhlayers .= df_ds.nhlayers .+ 2
ax.set_title(L"J_m = %$(Int(df_ds.batch_size[1])), N = %$(Int(df_ds.N[1])), K = %$(Int(df_ds.K[1])), T = %$(df_ds.T[1])")
println(df_ds)
for r in eachrow(df_ds)
    ax.errorbar(r.nhlayers, r."\$L^1-\$approx. error", yerr = r."Std. dev. error", c = "tab:blue", fmt = "o", ms = 4)
end
# ax.set_ylabel(L"$L^1$-approx. error")
ax.set_xlabel("Nb. of layers")
ax.set_yscale("log")

for r in eachrow(df_ds)
    axb.errorbar(r.nhlayers, r."avg. runtime (s)", c = "tab:red", fmt = "o", ms = 4)
end
# axb.set_ylabel("avg. runtime (s)")
axb.set_xlabel("Nb. of layers")
axb.set_xticks(df_ds.nhlayers)
axb.set_ylim(0,50)
display(fig)

# ax.set_ylim(0,0.1)
# fig.subplots_adjust(wspace = 0.4)
# fig.tight_layout()
display(fig)

fig.savefig(path_results*"/fig_DS_param_explo.pdf", dpi=100, bbox_inches="tight")