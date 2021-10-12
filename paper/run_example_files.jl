cd(@__DIR__)
using CUDA
CUDA.device!(6)
using Statistics
using HighDimPDE
using Flux
using Random
# Random.seed!(100)
# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV
mydir = "results2"
isdir(mydir) ? nothing : mkdir(mydir)

include("DeepSplitting_nonlocal_comp.jl")
include("DeepSplitting_nonlocal_sinegordon.jl")

include("MLP_nonlocal_comp.jl")
include("MLP_nonlocal_sinegordon.jl")

examples = [ 
            # :nonlocal_comp, 
            :nonlocal_sinegordon,
            # :allen_cahn_nonlocal, 
            # :fisher_kpp_reflected,
            ]
ds = [5] # [1, 2, 5, 10]
Ts = [1/5] # [1/5, 1/2, 1]

# Deepsplitting
N = 3
maxiters = 2000
batch_size = 32000

for (i,ex) in enumerate(examples)
    try
        println("Example ", String(ex))
        names_df = [L"d",L"T",L"N","Mean","Std. dev.", "avg. runtime (s)"]
        df_ds = DataFrame(); [df_ds[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
        dfu_ds = DataFrame(); [dfu_ds[!,c] = Float64[] for c in ["d","T","N","u","time_simu"]]
        df_mlp = DataFrame(); [df_mlp[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
        dfu_mlp = DataFrame(); [dfu_mlp[!,c] = Float64[] for c in ["d","T","K","u","time_simu"]]
        for d in ds, T in Ts
                u_ds = []
                u_mlp = [ ]
                dt = T / N
                tspan = (0f0,T)
                # solving         
                for i in 1:5
                    ##################
                    # Deep Splitting #
                    ##################
                    println("d=",d," T=",T," i=",i)
                    println("DeepSplitting")
                    sol_ds = @timed eval(string("DeepSplitting_", ex) |> Symbol)(d, T, dt)
                    @show sol_ds.value
                    push!(u_ds,[sol_ds.value[3][end],sol_ds.time])
                    push!(dfu_ds,(d, T, N, u_ds[end][1],u_ds[end][2]))
                    CSV.write(mydir*"/$(String(ex))_ds.csv", dfu_ds)
                    ################
                    ##### MLP ######
                    ################
                    println("MLP")
                    sol_mlp = @timed eval(string("MLP_", ex) |> Symbol)(d, T, dt)
                    @show sol_mlp.value
                    push!(u_mlp, [sol_mlp.value[3][end], sol_mlp.time])
                    push!(dfu_mlp,(d, T, N, u_mlp[end][1], u_mlp[end][2]))
                    CSV.write(mydir*"/$(String(ex))_mlp.csv", dfu_mlp)
                end
                push!(df_ds, (d, T, N, mean(u_ds)[1], std(u_ds)[1], mean(u_ds)[2] ))
                push!(df_mlp, (d, T, L, mean(u_mlp)[1], std(u_mlp)[1],mean(u_mlp)[2] ))
        end
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
    catch e
        println("Error with example ", String(ex))
        println(e)
    end
end
