################################
#=
Simulating and plotting results
for the replicator equation
=#
################################

cd(@__DIR__)
using HighDimPDE
using Random
using Test
import Flux: Chain, Dense, ADAM
using Revise
using PyPlot, Printf
using UnPack
using JLD2
plotting = true
simu = false

if simu
    tspan = (0f0,15f-2)
    dt = 5f-2 # time step
    μ(X,p,t) = 0f0 # advection coefficients
    σ(X,p,t) = 1f-1 # diffusion coefficients
    d = 5
    ss0 = 5f-2#std g0
    ∂ = fill(5f-1, d)
    x0_sample = UniformSampling(-∂, ∂)
    x0 = fill(0f0,d) # initial point

    ##############################
    ####### ML params #######
    ##############################
    maxiters = 2000
    batch_size = 8000
    K = 5

    hls = d + 50 #hidden layer size

    nn = Chain(Dense(d, hls, tanh),
                    Dense(hls,hls,tanh),
                    Dense(hls, 1, x->x^2)) # Neural network used by the scheme

    opt = ADAM(5e-3)#optimiser

    ##########################
    ###### PDE Problem #######
    ##########################
    g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
    m(x) = - 5f-1 * sum(x.^2, dims=1)
    vol = prod(2 * ∂)
    f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) =  v_y .* (m(y) .- vol * v_z .* m(z) ) # nonlocal nonlinear part of the

    # defining the problem
    alg = DeepSplitting(nn, K=K, opt = opt,
                            mc_sample = x0_sample )
    prob = PIDEProblem(g, f, μ, σ, x0, tspan, x0_sample = x0_sample)

    # solving
    @time sol = solve(prob, 
                    alg, 
                    dt, 
                    verbose = true, 
                    maxiters = maxiters,
                    batch_size = batch_size,
                    use_cuda = true,
                    verbose_rate = 1,
                    cuda_device = 2
                    )

    @save "solution_convergence.jld2" sol
end

@load "solution_convergence.jld2" sol;

legs = ["V"]

fig, ax = plt.subplots(1)
# loading simulation
for (i,loss) in enumerate(sol.losses[1:end-1])
    its = (1:length(loss))[1:4:end]
    loss = loss[1:4:end]
    ax.plot(1:length(loss), loss, label = L"\phi^{%$i,m,0}(\Theta_m^{%$i})")
end
ax.set_xlabel(L"m")
ax.set_yscale("log")
ax.set_ylabel("Loss")
ax.legend()
display(fig)

fig.savefig("figure_convergence.pdf", dpi=100)