#=
Replicator mutator example with Deep Splitting,
solving on the whole hypercube
=#

using HighDimPDE
using Random
using Test
import Flux
import Flux.tanh, Flux.relu, Flux.Dense
using Revise

function DeepSplitting_rep_mut(; d, T, N, batch_size = 8000, K = 5, cuda_device)
        dt = T / N
        tspan = (0f0,T)
        ##############################
        ####### ML params #######
        ##############################
        maxiters = 4000        

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d, hls, tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls, 1, x->x^2))

        opt = Flux.ADAM() # optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        U = 5f-1
        domain = (fill(-U, d), fill(U, d))
        x0_sample = UniformSampling(domain...)
        ss0 = 5f-2#std g0

        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = Float32((2f0*π)^(-d/2f0)) * ss0^(- Float32(d) * 5f-1) * 
                exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(domain[2] - domain[1])
        f(y, z, v_y, v_z, ∇u_x, ∇u_y, p, t) =  v_y .* (m(y) .- vol * v_z .* m(z))

        # reference solution
        function _SS(x, t, p)
                d = length(x)
                MM = σ(x, p, t) * ones(d)
                SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* 
                        cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
                return SSt
        end
        
        function rep_mut_anal(x, t, p)
                d = length(x)
                return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * 
                        exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
        end

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt, λs = [5e-3,1e-3],
                mc_sample = UniformSampling(domain[1], domain[2]) )
        prob = PIDEProblem(g, f, μ, σ, zeros(Float32,d), tspan, x0_sample = x0_sample)
        # solving
        sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=1f-99,
                maxiters = maxiters,
                batch_size = batch_size,
                use_cuda = true,
                cuda_device = cuda_device
                )
        return sol.us[end], rep_mut_anal(zeros(d), T, Dict())
end

if false
        d = 10
        dt = 1f-5 # time step
        T = 1f-5
        sol, lossmax, truesol = DeepSplitting_rep_mut(d, T, dt)
        println("True solution: $truesol, Deep splitting approximation = $(sol)")
end