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
    example = Symbol(ARGS[2])
else
    cuda_device = 1
    example = :rep_mut
end
using Statistics
using HighDimPDE
using Flux
using Random
using UnPack
# Random.seed!(100)
# for post processes
using DataFrames
using Latexify # we could have used PrettyTables
using LaTeXStrings
using CSV, JLD2, ProgressMeter
using Dates
mydir = "results_rev_$(today())"
isdir(mydir) ? nothing : mkdir(mydir)

include("DeepSplitting_rep_mut_x0_sample.jl")

# common to all experiments
d = 5
T = 0.5
# overwritten for certain experiments
N = 5
K = 5
batch_size = 8000

# Array of params to explore
Ns = [1, 3, 5, 7, 9]
batch_sizes = [1000, 3000, 5000, 7000, 9000]
Ks = [1, 3, 5, 7, 9]

default_settings = Dict{Symbol,Any}()
@pack! default_settings = d, T, N, batch_size, K

explo_all = Dict("explo_K" => Dict[], "explo_batch_size" => Dict[], "explo_N" => Dict[])
for K in Ks
    dict_temp = copy(default_settings)
    dict_temp[:K] = K
    push!(explo_all["explo_K"], dict_temp)
end
for N in Ns
    dict_temp = copy(default_settings)
    dict_temp[:N] = N
    push!(explo_all["explo_N"], dict_temp)
end
for batch_size in batch_sizes
    dict_temp = copy(default_settings)
    dict_temp[:batch_size] = batch_size
    push!(explo_all["explo_batch_size"], dict_temp)
end


nruns = 5 #number of runs per example
progr = Progress( length(Ns) * length(batch_sizes) * length(Ks) * nruns, showspeed = true, barlen = 10)

# result containers
# summary stats table
df_ds = DataFrame("Mean" => Float64[],
                "Std. dev." => Float64[],
                "Ref. value" => Float64[],
                L"L^1-"*"approx. error" => Float64[],
                "Std. dev. error" => Float64[],
                "avg. runtime (s)" => Float64[], 
                (string.(keys(default_settings)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...)

# complete table
dfu_ds = DataFrame((string.(keys(default_settings)) .=> [Int64[], Float64[], Int64[], Int64[], Int64[]])...,
                "u" => Float64[],
                "time simu" => Float64[])


# running for precompilation
for _ in 1:nruns         
    DeepSplitting_rep_mut(; explo_all["explo_K"][1]..., cuda_device);
end

println("Experiment started")

for _ in 1:2 #burnin : first loop to heat up the gpu
    for T in Ts, d in ds
            u_ds = DataFrame("value" => Float64[], "time" => Float64[], "ref_value" => [])
            u_mlp = DataFrame("value" => Float64[], "time" => Float64[])
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
println("All results saved in $mydir")