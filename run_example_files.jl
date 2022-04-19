cd(@__DIR__)
using CUDA
# CUDA.device!(6)
using Statistics
using HighDimPDE
using Flux
using Random
# Random.seed!(100)
# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV, JLD2
mydir = "results"
isdir(mydir) ? nothing : mkdir(mydir)

include("DeepSplitting_nonlocal_comp.jl")
include("DeepSplitting_nonlocal_sinegordon.jl")
include("DeepSplitting_fisherkpp_neumann.jl")
include("DeepSplitting_rep_mut.jl")
include("DeepSplitting_allencahn_neumann.jl")

include("MLP_nonlocal_comp.jl")
include("MLP_nonlocal_sinegordon.jl")
include("MLP_fisherkpp_neumann.jl")
include("MLP_rep_mut.jl")
include("MLP_allencahn_neumann.jl")

examples = [ 
            :nonlocal_comp, 
            :nonlocal_sinegordon,
            :fisherkpp_neumann,
            :rep_mut,
            :allencahn_neumann, 
            ]
ds = [1, 2, 5, 10]
Ts = [1/5, 1/2, 1]

# for testing:
# ds = [5] 
# Ts = [1/5] 

# Deepsplitting
N = 10

# MLP
L = 5

for ex in examples
    # try
        names_df = [L"d", L"T", L"N", "Mean", "Std. dev.", "Ref. value", L"L^1-"*"approx. error", "Std. dev. error", "avg. runtime (s)"]
        df_ds = DataFrame(); [df_ds[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
        dfu_ds = DataFrame(); [dfu_ds[!,c] = Float64[] for c in ["d","T","N","u","time_simu"]]; dfu_ds[!,"ref_value"] = []
        df_mlp = DataFrame(); [df_mlp[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
        dfu_mlp = DataFrame(); [dfu_mlp[!,c] = Float64[] for c in ["d","T","K","u","time_simu"]];
        for d in ds, T in Ts
                u_ds = DataFrame("value" => Float64[], "time" => Float64[], "ref_value" => [])
                u_mlp = DataFrame("value" => Float64[], "time" => Float64[])
                dt = T / N
                tspan = (0f0,T)
                # solving         
                for i in 1:5
                    ##################
                    # Deep Splitting #
                    ##################
                    println("Example ", String(ex))
                    println("d=",d," T=",T," i=",i)
                    println("DeepSplitting")
                    sol_ds = @timed eval(string("DeepSplitting_", ex) |> Symbol)(d, T, dt)
                    lossmax = sol_ds.value[2]
                    iter = 1
                    while (lossmax > 2e-4) && iter < 10 #this is to make sure that the approximation at the first step has converged
                        sol_ds = @timed eval(string("DeepSplitting_", ex) |> Symbol)(d, T, dt)
                        lossmax = sol_ds.value[2]
                        iter += 1
                    end
                    @show sol_ds.value[1]
                    push!(u_ds,[sol_ds.value[1],sol_ds.time,sol_ds.value[3]])
                    push!(dfu_ds,(d, T, N, u_ds[end,:]...))
                    CSV.write(mydir*"/$(String(ex))_ds.csv", dfu_ds)
                    JLD2.save(mydir*"/$(String(ex))_mlp.jld2", Dict("dfu_ds" => dfu_ds))
                    ################
                    ##### MLP ######
                    ################
                    println("MLP")
                    sol_mlp = @timed eval(string("MLP_", ex) |> Symbol)(d, T, L)
                    @show sol_mlp.value
                    push!(u_mlp, [sol_mlp.value, sol_mlp.time])
                    push!(dfu_mlp,(d, T, N, u_mlp[end,:]...))
                    CSV.write(mydir*"/$(String(ex))_mlp.csv", dfu_mlp)
                    JLD2.save(mydir*"/$(String(ex))_mlp.jld2", Dict("dfu_mlp" => dfu_mlp))
                end
                ismissing(u_ds.ref_value[1]) ? ref_v = mean(u_mlp.value) : ref_v = u_ds.ref_value[1]
                push!(df_ds, (d, T, N, mean(u_ds.value), std(u_ds.value), ref_v, mean(abs.((u_ds.value .- ref_v) / ref_v)), std(abs.((u_ds.value .- ref_v) / ref_v)), mean(u_ds.time)))
                # reference values are only returned by deep splitting function
                ismissing(u_ds.ref_value[1]) ? ref_v = mean(u_ds.value) : ref_v = u_ds.ref_value[1]
                push!(df_mlp, (d, T, L, mean(u_mlp.value), std(u_mlp.value), ref_v, mean(abs.((u_mlp.value .- ref_v) / ref_v)), std(abs.((u_mlp.value .- ref_v) / ref_v)), mean(u_mlp.time)))
        end
        sort!(df_ds, L"T"); sort!(df_mlp, L"T")
        #ds
        tab_ds = latexify(df_ds,env=:tabular,fmt="%.7f") #|> String
        io = open(mydir*"/$(String(ex))_ds.tex", "w")
        write(io,tab_ds);
        close(io)
        #mlp
        tab_mlp = latexify(df_mlp,env=:tabular,fmt="%.7f") #|> String
        io = open(mydir*"/$(String(ex))_mlp.tex", "w")
        write(io,tab_mlp);
        close(io)
    # catch e
    #     println("Error with example ", String(ex))
    #     println(e)
    # end
end
