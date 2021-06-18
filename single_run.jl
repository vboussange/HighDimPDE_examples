cd(@__DIR__)
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

function hamel(d, tspan)
    σ_sampling = 1f0
    X0 = fill(0f0,d) # initial point
    g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
    m(x) = - 5f-1 * sum(x.^2, dims=1)
    f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) - max.(0f0, v_z) .* m(z) * Float32((2f0 * π)^(d/2) * σ_sampling^d) .* exp.(5f-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
    μ_f(X,p,t) = 0.0f0 # advection coefficients
    σ_f(X,p,t) = 1f-1 # diffusion coefficients
    mc_sample = NormalSampling(σ_sampling) # normal

    # defining the problem
    prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                            )
    prob, mc_sample
end

function hamel64(d, tspan)
    σ_sampling = 1e-1
    X0 = fill(0f0,d) # initial point
    g(X) = 2e0^(d/2)* exp.(-2e0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
    m(x) = - 5e-1 * sum(x.^2, dims=1)
    f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0e0, v_y) .* ( m(y) - max.(0e0, v_z) .* m(z) * (2e0 * π)^(d/2) * σ_sampling^d .* exp.(5e-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
    μ_f(X,p,t) = 0e0 # advection coefficients
    σ_f(X,p,t) = 1e-1 # diffusion coefficients
    mc_sample = NormalSampling(σ_sampling) # normal

    # defining the problem
    prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                            )
    prob, mc_sample
end

T = 0.2
tspan = (0f0,T)
d = 1

if false
    # Deepsplitting
    N = 10
    maxiters = 8000
    batch_size = 32000


    hls = d + 50 #hidden layer size

    nn = Flux.Chain(Dense(d,hls,tanh),
                    Dense(hls,hls,tanh),
                    Dense(hls,1)) # Neural network used by the scheme

    opt = Flux.Optimiser(ExpDecay(0.1,
                        0.1,
                        500,
                        1e-4),
                        ADAM() )#optimiser

    prob, mc_sample = hamel(d, tspan)
    alg_ds = DeepSplitting(nn, K = 1, opt = opt, mc_sample = mc_sample )
    @time sol_ds = solve(prob, alg_ds,
                            dt=T/N,
                            verbose = true,
                            abstol=1e-6,
                            maxiters = maxiters,
                            batch_size = batch_size,
                            use_cuda = true)
    println("Solution deep splitting = ", sol_ds.u[end])
end 

if true
    # MLP
    L = 5
    prob, mc_sample = hamel64(d, tspan)
    alg_mlp = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )
    @time sol_mlp = solve(prob, alg_mlp, multithreading=true, verbose=true)
    println("Solution MLP = ", sol_mlp)
end