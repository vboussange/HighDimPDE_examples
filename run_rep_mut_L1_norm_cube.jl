#= 
Run DeepSplitting with rep mut
with x0_sample
and computing L1 norm on the whole hypercube
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
Random.seed!(21)
# for post processes
using DataFrames
using Latexify # we could have used PrettyTables
using LaTeXStrings
using CSV, JLD2, ProgressMeter, Dates
mydir = "results/$(today())/results_rep_mut_x0_sample_L1"
isdir(mydir) ? nothing : mkpath(mydir)

include("DeepSplitting_rep_mut_x0_sample_L1.jl")

ds = [1, 2, 5, 10]

Ts = [1/10, 1/5, 1/2]

deepsplitting_fun = DeepSplitting_rep_mut_L1
# Deepsplitting
N = 10

nruns = 5 #number of runs per example
progr = Progress( length(ds) * length(Ts) * nruns, showspeed = true, barlen = 10)
println("Experiment started with Ts = $Ts.")

names_df = [L"d", L"T", L"N", L"L^1-"*"approx. error", "Std. dev. error", "avg. runtime (s)"]
df_ds = DataFrame(); [df_ds[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
dfu_ds = DataFrame(); [dfu_ds[!,c] = Float64[] for c in ["d","T","N","L1","time_simu"]];

# running for precompilation
for _ in 1:nruns         
    deepsplitting_fun(d = ds[end], T = Ts[end], N = N, cuda_device = cuda_device);
end


for _ in 1:2 #burnin : first loop to heat up the gpu
    for T in Ts, d in ds
            u_ds = DataFrame("L1" => Float64[], "time" => Float64[])
            for i in 1:nruns
                ##################
                # Deep Splitting #
                ##################
                println("Example ", String(example))
                println("d=",d," T=",T," i=",i)
                println("DeepSplitting")
                sol_ds = @timed deepsplitting_fun(;d, T, N, cuda_device)
                L1err = sol_ds.value[1]
                @show L1err
                @show sol_ds.time

                push!(u_ds,[sol_ds.value[1],sol_ds.time])
                push!(dfu_ds,(d, T, N, u_ds[end,:]...))
                CSV.write(mydir*"/$(String(example))_ds_x0_sample_L1.csv", dfu_ds)
                JLD2.save(mydir*"/$(String(example))_ds_x0_sample_L1.jld2", Dict("dfu_ds" => dfu_ds))
                next!(progr)
            end
            push!(df_ds, (d, T, N, mean(u_ds.L1), std(u_ds.L1), mean(u_ds.time)))
    end
    sort!(df_ds, L"T"); sort!(df_mlp, L"T")
    #ds
    tab_ds = latexify(df_ds,env=:tabular,fmt="%.7f") #|> String
    io = open(mydir*"/$(String(example))_ds_x0_sample_L1.tex", "w")
    write(io,tab_ds);
    close(io)
end

println("All results saved in $mydir")