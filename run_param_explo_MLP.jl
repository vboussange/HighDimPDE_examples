#= 
Run MLP examples
with varying M, L, and N

We keep constant 
- T = 0.5
- d = 5

=#
cd(@__DIR__)
using Statistics
using HighDimPDE
using Flux
using Random
using UnPack
Random.seed!(56)
# for post processes
using DataFrames
using Latexify # we could have used PrettyTables
using LaTeXStrings
using CSV, JLD2, ProgressMeter
using Dates


include("MLP_rep_mut.jl")

# common to all experiments
d = 5
T = 0.5
# overwritten for certain experiments
M = 5
K = 5
L = 4
mydir = "results/$(today())/explo_param_MLP_T=$(T)_explo_K_M_4"
isdir(mydir) ? nothing : mkpath(mydir)

# Array of params to explore
Ms = 1:5
Ks = 1:4:20
Ls = 1:5

default_settings = Dict{Symbol,Any}()
@pack! default_settings = d, T, M, K, L

explo_all = Dict("explo_K" => Dict[], "explo_M" => Dict[], "explo_L" => Dict[])
dict_results = Dict("explo_K" => Dict{String,Any}(), 
                    "explo_M" => Dict{String,Any}(), 
                    "explo_L" => Dict{String,Any}())

for K in Ks
    dict_temp = copy(default_settings)
    dict_temp[:K] = K
    dict_temp[:M] = 4
    push!(explo_all["explo_K"], dict_temp)
end
for M in Ms
    dict_temp = copy(default_settings)
    dict_temp[:M] = M
    push!(explo_all["explo_M"], dict_temp)
end
for L in Ls
    dict_temp = copy(default_settings)
    dict_temp[:L] = L
    push!(explo_all["explo_L"], dict_temp)
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

simul = MLP_rep_mut

nruns = 5 #number of runs per example
progr = Progress( length(Ms) * length(Ls) * length(Ks) * nruns, showspeed = true, barlen = 10)
# running for precompilation
for _ in 1:nruns         
    simul(;explo_all["explo_L"][1]...)
end

println("Experiment started")

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
            sol_mlp = @timed simul(;dict...)

            @show sol_mlp.value[1]
            @show sol_mlp.time

            push!(u_ds,[sol_mlp.value[1],sol_mlp.time,sol_mlp.value[2]])
            push!(dfu_ds,(values(dict)..., u_ds[end,:]...))
            CSV.write(mydir*"/$(scen)_MLP.csv", dfu_ds)
            # logging
            next!(progr)
        end
        ref_v = u_ds.ref_value[1]
        push!(df_ds, (mean(u_ds.value), std(u_ds.value), ref_v, mean(abs.((u_ds.value .- ref_v) / ref_v)), std(abs.((u_ds.value .- ref_v) / ref_v)), mean(u_ds.time),values(dict)...))
    end
    @pack! dict_results[scen] = df_ds, dfu_ds
end

JLD2.save(mydir*"/dict_results_MLP_param_explo.jld2", dict_results)
println("All results saved in $mydir")