cd(@__DIR__)
using Statistics
using HighDimPDE
using Flux
using Random
using CUDA
Random.seed!(100)
# for post processes
using DataFrames
using Latexify
using LaTeXStrings
using CSV
mydir = "results2"
isdir(mydir) ? nothing : mkdir(mydir)

include("examples.jl")
examples = [#
            :fisher_kpp, 
            :hamel, 
            # :merton, 
            :mirrahimi, 
            :nonlocal_comp, 
            :sine_gordon,
            :allen_cahn_nonlocal, 
            ]
ds = [1,2,5,10] # [1]
Ts = [1/5, 1/2, 1] # [1/5]

# Deepsplitting
N = 10
maxiters = 8000
batch_size = 16000

# MLP
L = 5

for (i,ex) in enumerate(examples)
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

            ##################
            # Deep Splitting #
            ##################
            prob, mc_sample = eval(ex)(d, tspan, gpu)
            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                            Dense(hls,hls,tanh),
                            Dense(hls,1)) # Neural network used by the scheme

            opt = Flux.Optimiser(ExpDecay(0.1,
                                0.1,
                                2000,
                                1e-4),
                                ADAM() )#optimiser

            alg_ds = DeepSplitting(nn, K = 5, opt = opt, mc_sample = mc_sample )
            
            ################
            ##### MLP ######
            ################
            prob, mc_sample = eval(ex)(d, tspan, cpu)
            alg_mlp = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )

            for i in 1:5
                # solving
                println("d=",d," T=",T," i=",i)
                println("DeepSplitting")
                sol_ds = @timed solve(prob, alg_ds,
                                        dt=dt,
                                        verbose = false,
                                        abstol=1e-6,
                                        maxiters = maxiters,
                                        batch_size = batch_size)
                push!(u_ds,[sol_ds.value.u[end],sol_ds.time])
                push!(dfu_ds,(d, T, N, u_ds[end][1],u_ds[end][2]))
                CSV.write(mydir*"/$(String(ex))_ds.csv", dfu_ds)

                println("MLP")
                sol_mlp = @timed solve(prob, alg_mlp, multithreading=true)
                push!(u_mlp, [sol_mlp.value, sol_mlp.time])
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
end
