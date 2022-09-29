#= 
Script to compare MLP and DS algo
with varying M, L, and N

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
Random.seed!(95)
# for post processes
using DataFrames
using Latexify # we could have used PrettyTables
using LaTeXStrings
using CSV, JLD2, ProgressMeter
using Dates


include("MLP_rep_mut.jl")
include("DeepSplitting_rep_mut.jl")

# common to all experiments
d = 5
T = 0.5
# MLP params
M = 5
K = 5
L = 4 # overwritten
Ls = 1:5

# DS params
N = 3
K = 1
batch_size = 200 # overwritten
batch_sizes = 1:50:250
maxiters = 500

mydir = "results/$(today())/explo_MLP_vs_DS"
isdir(mydir) ? nothing : mkpath(mydir)

settings_MLP = Dict{Symbol,Any}()
@pack! settings_MLP = d, T, M, K, L

settings_DS = Dict{Symbol,Any}()
@pack! settings_DS = d, T, N, K, batch_size

explo_all = Dict("DS" => Dict[], "MLP" => Dict[])
dict_results = Dict("DS" => Dict{String,Any}(), 
                    "MLP" => Dict{String,Any}())

for L in Ls
    dict_temp = copy(settings_MLP)
    dict_temp[:L] = L
    push!(explo_all["MLP"], dict_temp)
end
for batch_size in batch_sizes
    dict_temp = copy(settings_DS)
    dict_temp[:batch_size] = batch_size
    push!(explo_all["DS"], dict_temp)
end

# result containers
# summary stats table
df_DS_init = DataFrame("Mean" => Float64[],
                "Std. dev." => Float64[],
                "Ref. value" => Float64[],
                L"L^1-"*"approx. error" => Float64[],
                "Std. dev. error" => Float64[],
                "avg. runtime (s)" => Float64[], 
                (string.(keys(settings_DS)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...)
df_MLP_init = DataFrame("Mean" => Float64[],
                "Std. dev." => Float64[],
                "Ref. value" => Float64[],
                L"L^1-"*"approx. error" => Float64[],
                "Std. dev. error" => Float64[],
                "avg. runtime (s)" => Float64[], 
                (string.(keys(settings_MLP)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...)
# complete table
dfu_DS_init = DataFrame((string.(keys(settings_DS)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...,
                "u" => Float64[],
                "time simu" => Float64[],
                "ref_value" => Float64[])
dfu_MLP_init = DataFrame((string.(keys(settings_MLP)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...,
                "u" => Float64[],
                "time simu" => Float64[],
                "ref_value" => Float64[])

nruns = 5 #number of runs per example

println("MLP")
scen = "MLP"
dfu = copy(dfu_MLP_init)
df = copy(df_MLP_init)

for dict in explo_all[scen]
    u = DataFrame("value" => Float64[], 
                    "time" => Float64[], 
                    "ref_value" => [])
    for i in 1:nruns
        ##################
        # Deep Splitting #
        ##################
        println(scen," i=",i)
        sol_mlp = @timed MLP_rep_mut(;dict...)

        @show sol_mlp.value[1]
        @show sol_mlp.time

        push!(u,[sol_mlp.value[1],sol_mlp.time,sol_mlp.value[2]])
        push!(dfu,(values(dict)..., u[end,:]...))
        CSV.write(mydir*"/$(scen).csv", dfu)
    end
    ref_v = u.ref_value[1]
    push!(df, (mean(u.value), std(u.value), ref_v, mean(abs.((u.value .- ref_v) / ref_v)), std(abs.((u.value .- ref_v) / ref_v)), mean(u.time),values(dict)...))
end
@pack! dict_results[scen] = df, dfu


println("DS")
# running for precompilation
for _ in 1:nruns         
    simul(; explo_all["DS"][4]..., cuda_device, maxiters);
end
dfu = copy(dfu_DS_init)
df = copy(df_DS_init)
scen = "DS"
for dict in explo_all[scen]
    u = DataFrame("value" => Float64[], 
                    "time" => Float64[], 
                    "ref_value" => [])
    for i in 1:nruns
        ##################
        # Deep Splitting #
        ##################
        println(scen," i=",i)
        sol_mlp = @timed DeepSplitting_rep_mut(;dict...,cuda_device, maxiters)

        @show sol_mlp.value[1]
        @show sol_mlp.time

        push!(u,[sol_mlp.value[1],sol_mlp.time,sol_mlp.value[2]])
        push!(dfu,(values(dict)..., u[end,:]...))
        CSV.write(mydir*"/$(scen).csv", dfu)
    end
    ref_v = u.ref_value[1]
    push!(df, (mean(u.value), std(u.value), ref_v, mean(abs.((u.value .- ref_v) / ref_v)), std(abs.((u.value .- ref_v) / ref_v)), mean(u.time),values(dict)...))
end
@pack! dict_results[scen] = df, dfu

JLD2.save(mydir*"/dict_results_MLP_vs_DS.jld2", dict_results)
println("All results saved in $mydir")