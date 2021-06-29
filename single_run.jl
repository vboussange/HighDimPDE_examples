cd(@__DIR__)
using Statistics
using HighDimPDE
using Flux
using Random
using DataFrames
using Latexify
using LaTeXStrings
using CSV
using Revise
# mydir = "results2"
# isdir(mydir) ? nothing : mkdir(mydir)

function hamel(d, tspan)
    T = Float64
    σ_sampling = T(1.)
    X0 = zeros(T, d) # initial point
    g(X) = T(2.0^(d/2))* exp(-T(2.0 * π)  * sum( X.^2 ))   # initial condition
    m(x) = - T(0.5) * sum(x.^2)
    f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max(zero(T), v_y) * ( m(y) - max(zero(T), v_z) * m(z) * T((2. * π)^(d/2) * σ_sampling^d) * exp(T(0.5) * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the
    μ_f(X,p,t) = 0. # advection coefficients
    σ_f(X,p,t) = 0.1 # diffusion coefficients
    mc_sample = NormalSampling(σ_sampling) # normal

    # defining the problem
    prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
    prob, mc_sample
end

function nonlocal_comp(d, tspan)
    σ_sampling = 1f-1 / sqrt(2f0)
    X0 = fill(0f0,d) # initial point
    g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
    f(y, z, v_y, v_z, ∇v_y ,∇v_z, t) =  max.(0f0, v_y) .* (1f0 .- max.(0f0, v_z) * Float32((2 * π )^(d/2) * σ_sampling^d))
    μ_f(X,p,t) = 0.0f0 # advection coefficients
    σ_f(X,p,t) = 1f-1 # diffusion coefficients
    mc_sample = NormalSampling(σ_sampling, false) # uniform distrib in u_domain

    # defining the problem
    prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
    prob, mc_sample
end



T = 1.0
tspan = (0f0,T)
d = 1

if true
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

    prob, mc_sample = nonlocal_comp(d, tspan)
    alg_ds = DeepSplitting(nn, K = 10, opt = opt, mc_sample = mc_sample )
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
    prob, mc_sample = hamel(d, tspan)
    alg_mlp = MLP(M = L, K = 10, L = L, 
                    mc_sample = mc_sample 
                    )
    @time sol_mlp = solve(prob, alg_mlp, multithreading = false, verbose = false)
    println("Solution MLP = ", sol_mlp)
end