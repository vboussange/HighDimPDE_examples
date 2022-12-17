#= 
Run DeepSplitting examples
with varying K, batch size, and N

We keep constant 
- T = 0.5
- d = 5

Similar to run_param_explo_DS_x0_sample, but we vary parameters together in order to increase the nb. of std. random var. evaluations.
* Arguments
- `ARGS[1] = cuda_device::Int`
- `ARGS[2] = example::Symbol`
=#
cd(@__DIR__)
if !isempty(ARGS)
    cuda_device = parse(Int,ARGS[1])
else
    cuda_device = 2
end
using Statistics
using HighDimPDE
using Flux
using Random
using UnPack
Random.seed!(58)
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
U = 1f0
# overwritten for certain experiments
K = 5
nhlayers = 1

mydir = "results/$(today())/explo_param_DS_x0_sample_T=$(T)_nb_rand_var"
isdir(mydir) ? nothing : mkpath(mydir)

# Array of params to explore
Ns = 1:4
batch_sizes = [10^i for i in 1:4]
hlss = 10:10:40 # hidden layer sizes

default_settings = Dict{Symbol,Any}()
@pack! default_settings = d, T, K, nhlayers, U

scenarios = ["explo_nb_rand_var"]
# scenarios = ["explo_nhlayers", "explo_hls"]

explo_all = Dict([ scen => Dict[] for scen in scenarios]...)
dict_results = Dict([ scen => Dict{String,Any}() for scen in scenarios]...)

for i in 1:length(Ns)
    dict_temp = copy(default_settings)
    dict_temp[:N] = Ns[i]
    dict_temp[:batch_size] = batch_sizes[i]
    dict_temp[:hls] = hlss[i]
    push!(explo_all["explo_nb_rand_var"], dict_temp)
end


# result containers
# summary stats table
df_ds_init = DataFrame("Mean" => Float64[],
                "Std. dev." => Float64[],
                "Ref. value" => Float64[],
                L"L^1-"*"approx. error" => Float64[],
                "Std. dev. error" => Float64[],
                "avg. runtime (s)" => Float64[], 
                (string.(keys(explo_all["explo_nb_rand_var"][1])) .=> [Int64[], Float64[], Int64[], Int64[], Int64[], Int64[], Int64[], Float64[],])...)

# complete table
dfu_ds_init = DataFrame((string.(keys(explo_all["explo_nb_rand_var"][1])) .=> [Int64[], Float64[], Int64[], Int64[], Int64[], Int64[], Int64[], Float64[],])...,
                "u" => Float64[],
                "time simu" => Float64[],
                "ref_value" => Float64[])

simul = DeepSplitting_rep_mut

nruns = 5 #number of runs per example
progr = Progress( length(Ns) * nruns, showspeed = true, barlen = 10)
# running for precompilation
for _ in 1:nruns         
    simul(; explo_all[scenarios[1]][1]..., cuda_device);
end

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

suffix_name_file = prod(scenarios)
JLD2.save(mydir*"/dict_resultsparam_explo_DS_x0_sample_$(suffix_name_file).jld2", dict_results)
println("All results saved in $mydir")