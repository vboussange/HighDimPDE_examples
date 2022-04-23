using CUDA
# CUDA.device!(1)
using HighDimPDE
using Random
using Test
import Flux
import Flux.tanh, Flux.relu, Flux.Dense
using Revise

function DeepSplitting_rep_mut(d, T, dt, cuda_device)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 4000
        batch_size = 800
        K = 100

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d, hls, tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls, 1, x->x^2))

        opt = Flux.ADAM(1e-3) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        σ_sampling = 1f0
        x0 = fill(0f0,d) .+ eps()# initial point
        ss0 = 5f-2#std g0

        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = Float32((2f0*π)^(-d/2f0)) * ss0^(- Float32(d) * 5f-1) * 
                exp.(-5f-1 *sum(x .^2 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        int_scale = Float32((2 * π )^(d/2) * σ_sampling^d)
        f(y, z, v_y, v_z, p, t) =  v_y .* (m(y) .- int_scale * v_z .* m(z))

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
        alg = DeepSplitting(nn, K=K, opt = opt,
                mc_sample = NormalSampling(σ_sampling, false))
        prob = PIDEProblem(g, f, μ, σ, tspan, x=x0)
        # solving
        xs,ts,sol,lossmax = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=1f-99,
                maxiters = maxiters,
                batch_size = batch_size,
                use_cuda = true,
                cuda_device = cuda_device
                )
        return sol[end], lossmax, rep_mut_anal(zeros(d), T, Dict())
end

if true
        d = 10
        dt = 1f-1 # time step
        T = 1f-1
        sol, lossmax, truesol = DeepSplitting_rep_mut(d, T, dt, 1)
        println("True solution: $truesol, Deep splitting approximation = $(sol)")
end