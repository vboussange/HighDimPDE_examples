#= 
Run DeepSplitting examples
with varying K, batch size, and N

We keep constant 
- T = 0.5
- d = 5

* Arguments
- `ARGS[1] = cuda_device::Int`
- `ARGS[2] = example::Symbol`
=#
cd(@__DIR__)
if !isempty(ARGS)
    cuda_device = parse(Int,ARGS[1])
else
    cuda_device = 1
end
using Statistics
using HighDimPDE
using Flux
using Random
using UnPack
Random.seed!(2)
# for post processes
using DataFrames
using Latexify # we could have used PrettyTables
using LaTeXStrings
using CSV, JLD2, ProgressMeter
using Dates

include("DeepSplitting_rep_mut_x0_sample.jl")

# common to all experiments
d = 5
T = 0.2
# overwritten for certain experiments
N = 2
K = 3
batch_size = 200

mydir = "results/$(today())/explo_param_DS_x0_sample_T=$(T)"
isdir(mydir) ? nothing : mkpath(mydir)

# Array of params to explore
Ns = 1:5
batch_sizes = [10^i for i in 1:4]
Ks = 1:5

default_settings = Dict{Symbol,Any}()
@pack! default_settings = d, T, N, batch_size, K

explo_all = Dict("explo_K" => Dict[], "explo_batch_size" => Dict[], "explo_N" => Dict[])
dict_results = Dict("explo_K" => Dict{String,Any}(), 
                    "explo_batch_size" => Dict{String,Any}(), 
                    "explo_N" => Dict{String,Any}())

for K in Ks
    dict_temp = copy(default_settings)
    dict_temp[:K] = K
    batch_size = 50
    push!(explo_all["explo_K"], dict_temp)
end
for N in Ns
    T = 10.0
    dict_temp = copy(default_settings)
    dict_temp[:N] = N
    push!(explo_all["explo_N"], dict_temp)
end
for batch_size in batch_sizes
    dict_temp = copy(default_settings)
    dict_temp[:batch_size] = batch_size
    push!(explo_all["explo_batch_size"], dict_temp)
end

# result containers
# summary stats table
df_ds_init = DataFrame("Mean" => Float64[],
                "Std. dev." => Float64[],
                "Ref. value" => Float64[],
                L"L^1-"*"approx. error" => Float64[],
                "Std. dev. error" => Float64[],
                "avg. runtime (s)" => Float64[], 
                (string.(keys(default_settings)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...)

# complete table
dfu_ds_init = DataFrame((string.(keys(default_settings)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...,
                "u" => Float64[],
                "time simu" => Float64[],
                "ref_value" => Float64[])

simul = DeepSplitting_rep_mut
# running for precompilation
for _ in 1:nruns         
    simul(; explo_all["explo_K"][1]..., cuda_device);
end

nruns = 5 #number of runs per example
progr = Progress( length(Ns) * length(batch_sizes) * length(Ks) * nruns, showspeed = true, barlen = 10)

println("Experiment started")

# for _ in 1:2 #burnin : first loop to heat up the gpu
for scen in keys(explo_all)
    dfu_ds = copy(dfu_ds_init)
    df_ds = copy(df_ds_init)

    for dict in explo_all[scen]
        u_ds = DataFrame("value" => Float64[], 
                        "time" => Float64[], 
                        "ref_value" => [])
        for i in 1:nruns
            ##################
            # Deep Splitting #
            ##################
            println(scen," i=",i)
            sol_ds = @timed simul(;dict..., cuda_device)

            @show sol_ds.value[1]
            @show sol_ds.time

            push!(u_ds,[sol_ds.value[1],sol_ds.time,sol_ds.value[2]])
            push!(dfu_ds,(values(dict)..., u_ds[end,:]...))
            CSV.write(mydir*"/$(scen)_DS_x0_sample.csv", dfu_ds)
            # logging
            next!(progr)
        end
        ref_v = u_ds.ref_value[1]
        push!(df_ds, (mean(u_ds.value), std(u_ds.value), ref_v, mean(abs.((u_ds.value .- ref_v) / ref_v)), std(abs.((u_ds.value .- ref_v) / ref_v)), mean(u_ds.time),values(dict)...))
    end
    @pack! dict_results[scen] = df_ds, dfu_ds
end

JLD2.save(mydir*"/dict_resultsparam_explo_DS_x0_sample.jld2", dict_results)
println("All results saved in $mydir")