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

function allen_cahn_nonlocal(;d,tspan,L)
        X0 = fill(0e0,d)  # initial point
        u_domain = [-5e-1,5e-1]

        g(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2) / σ_sampling^2) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0e0 # advection coefficients
        σ_f(X,p,t) = sqrt(2f0) # diffusion coefficients
        mc_sample(x) = (rand(Float64,d) .- 0.5) * (u_domain[2]-u_domain[1]) .+ mean(u_domain) # uniform distrib in u_domain

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                u_domain = u_domain
                                )

        # using the Deep Splitting algorithm
        alg = MLP(M=L, K=10, L=L )

        # solving
        @time sol = solve(prob, alg, mc_sample)

        println("u1 = ", sol)

        sol
end

## Basic example
if false
        sol = allen_cahn_nonlocal(
                d = 3, # number of dimensions
                # one-dimensional heat equation
                tspan = (0.0e0,1e0),
                dt = 0.1e0,   # time step
                batch_size = 8192,
                train_steps = 8000,
                σ_sampling = 1e0,
                K = 5,
                )
        Plots.plot(sol)
end

########################
### For publication ####
########################
if true
        df = DataFrame(); [df[!,c] = Float64[] for c in [L"d",L"T",L"L","Mean","Std. dev."]]
        dfu = DataFrame(); [dfu[!,c] = Float64[] for c in ["d","T","N","u"]]
        for d in [1,2,5,10],T in [1/5,1/2,1]
                u = []
                L = 5
                for i in 1:5
                        sol = allen_cahn_nonlocal(L=L,
                                d = d, # dimension of the domain (D = \R^d)
                                # one-dimensional heat equation
                                tspan = (0.0e0,T),
                                )
                        push!(u,sol)
                        push!(dfu,(d,T,L,sol))
                        CSV.write("$(name_sim).csv", dfu)
                end
        push!(df,(d,T,L,mean(u),std(u)))
        end
        tab = latexify(df,env=:tabular,fmt="%.7f") #|> String
        io = open("$(name_sim).tex", "w")
        write(io,tab);
        close(io)
end
