#= 
run example ARGS[2] and save statistics as latex files.

* Version
- arg_parse : run simulations by parsing arguments (cuda_device and example)
=#
cd(@__DIR__)
cuda_device = parse(Int,ARGS[1])
example = Symbol(ARGS[2])
using Statistics
using HighDimPDE
using Flux
using Random
# Random.seed!(100)
# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV, JLD2, ProgressMeter
mydir = "results-25-04-2022-burnin"
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

ds = [1, 2, 5, 10]
if example == :rep_mut
    Ts = [1/10, 1/5, 1/2]
else
    Ts = [1/5, 1/2, 1.0]
end

deepsplitting_fun = eval(string("DeepSplitting_", example) |> Symbol)
mlp_fun = eval(string("MLP_", example) |> Symbol)
# Deepsplitting
N = 10

# MLP
L = 5

nruns = 5 #number of runs per example
progr = Progress( length(ds) * length(Ts) * nruns, showspeed = true, barlen = 10)
println("Experiment started with Ts = $Ts.")
# try
names_df = [L"d", L"T", L"N", "Mean", "Std. dev.", "Ref. value", L"L^1-"*"approx. error", "Std. dev. error", "avg. runtime (s)"]
df_ds = DataFrame(); [df_ds[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
dfu_ds = DataFrame(); [dfu_ds[!,c] = Float64[] for c in ["d","T","N","u","time_simu"]]; dfu_ds[!,"ref_value"] = []
df_mlp = DataFrame(); [df_mlp[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
dfu_mlp = DataFrame(); [dfu_mlp[!,c] = Float64[] for c in ["d","T","K","u","time_simu"]];

# running for precompilation
for _ in 1:nruns         
    deepsplitting_fun(ds[end], Ts[end], Ts[end] / N, cuda_device);
end
for _ in 1:nruns         
    mlp_fun(ds[end], Ts[end], L);
end

for _ in 1:2 #burnin
    for T in Ts, d in ds
            u_ds = DataFrame("value" => Float64[], "time" => Float64[], "ref_value" => [])
            u_mlp = DataFrame("value" => Float64[], "time" => Float64[])
            dt = T / N
            # sarting the timing
            for i in 1:nruns
                ##################
                # Deep Splitting #
                ##################
                println("Example ", String(example))
                println("d=",d," T=",T," i=",i)
                println("DeepSplitting")
                sol_ds = @timed deepsplitting_fun(d, T, dt, cuda_device)

                @show sol_ds.value[1]
                @show sol_ds.time

                push!(u_ds,[sol_ds.value[1],sol_ds.time,sol_ds.value[3]])
                push!(dfu_ds,(d, T, N, u_ds[end,:]...))
                CSV.write(mydir*"/$(String(example))_ds.csv", dfu_ds)
                JLD2.save(mydir*"/$(String(example))_mlp.jld2", Dict("dfu_ds" => dfu_ds))
                ################
                ##### MLP ######
                ################
                println("MLP")
                sol_mlp = @timed mlp_fun(d, T, L)

                @show sol_mlp.value
                @show sol_mlp.time


                push!(u_mlp, [sol_mlp.value, sol_mlp.time])
                push!(dfu_mlp,(d, T, N, u_mlp[end,:]...))
                CSV.write(mydir*"/$(String(example))_mlp.csv", dfu_mlp)
                JLD2.save(mydir*"/$(String(example))_mlp.jld2", Dict("dfu_mlp" => dfu_mlp))
                # logging
                next!(progr)
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
    io = open(mydir*"/$(String(example))_ds.tex", "w")
    write(io,tab_ds);
    close(io)
    #mlp
    tab_mlp = latexify(df_mlp,env=:tabular,fmt="%.7f") #|> String
    io = open(mydir*"/$(String(example))_mlp.tex", "w")
    write(io,tab_mlp);
    close(io)
end
# catch e
#     println("Error with example ", String(example))
#     println(e)
# end
println("All results saved in $mydir")