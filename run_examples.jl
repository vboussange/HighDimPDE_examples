cd(@__DIR__)
name_sim = split(splitpath(@__FILE__)[end],".")[1]
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
isdir("results") ? nothing : mkdir("results")

include("examples.jl")
examples = [:allen_cahn_nonlocal, 
            # :fisher_kpp, :hamel, :merton, :mirrahimi, :nonlocal_comp, :sine_gordon
            ]
ds = [1]#[1,2,5,10]
Ts = [1/5] #[1/5,1/2,1]

for (i,ex) in enumerate(examples)
    names_df = [L"d",L"T",L"N","Mean","Std. dev.", "avg. runtime (s)"]
    df_ds = DataFrame(); [df_ds[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
    dfu_ds = DataFrame(); [dfu_ds[!,c] = Float64[] for c in ["d","T","N","u","time_simu"]]
    df_mlp = DataFrame(); [df_mlp[!,names_df[i]] = [Int64[], Int64[], Int64[], Float64[], Float64[], Float64[] ][i] for i in 1:length(names_df)]
    dfu_mlp = DataFrame(); [dfu_mlp[!,c] = Float64[] for c in ["d","T","K","u","time_simu"]]
    for d in ds, T in Ts
            N = 10
            L = 5
            u_ds = []
            u_mlp = [ ]
            dt = T / N
            tspan = (0f0,T)
            prob, mc_sample = eval(ex)(d,tspan)

            ##################
            # Deep Splitting #
            ##################
            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                            Dense(hls,hls,tanh),
                            Dense(hls,1)) # Neural network used by the scheme

            opt = Flux.Optimiser(ExpDecay(0.1,
                                0.1,
                                2000,
                                1e-4),
                                ADAM() )#optimiser

            alg_ds = DeepSplitting(nn, K=1, opt = opt, mc_sample = mc_sample )
            
            ################
            ##### MLP ######
            ################
            alg_mlp = MLP(M=L, K=10, L = L, mc_sample = mc_sample )

            for i in 1:5
                # solving
                sol_ds = @timed solve(prob, alg_ds,
                            dt=dt,
                            verbose = false,
                            abstol=1e-6,
                            maxiters = 8000,
                            batch_size = 8000)
                sol_mlp = @timed solve(prob, alg_mlp, multithreading=true)

                push!(u_ds,[sol_ds.value.u[end],sol_ds.time])
                push!(dfu_ds,(d, T, N, u_ds[end][1],u_ds[end][2]))
                push!(u_mlp, [sol_mlp.value, sol_mlp.time])
                push!(dfu_mlp,(d, T, N, u_mlp[end][1], u_mlp[end][2]))
                CSV.write("results/$(name_sim)_ds.csv", dfu_ds)
                CSV.write("results/$(name_sim)_mlp.csv", dfu_mlp)
            end
            push!(df_ds, (d, T, N, mean(u_ds)[1], std(u_ds)[1], mean(u_ds)[2] ))
            push!(df_mlp, (d, T, L, mean(u_mlp)[1], std(u_mlp)[1],mean(u_mlp)[2] ))
    end
    #ds
    tab_ds = latexify(df_ds,env=:tabular,fmt="%.7f") #|> String
    io = open("results/$(name_sim)_ds.tex", "w")
    write(io,tab_ds);
    close(io)
    #mlp
    tab_mlp = latexify(df_mlp,env=:tabular,fmt="%.7f") #|> String
    io = open("results/$(name_sim)_mlp.tex", "w")
    write(io,tab_mlp);
    close(io)
end
