using CUDA
# CUDA.device!(2)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function DeepSplitting_allencahn_neumann(d, T, dt, cuda_device)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 500
        batch_size = 8000
        K = 5

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        # Dense(hls,hls,relu),
                        Dense(hls, 1)) 

        opt = Flux.ADAM(1e-2) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        ∂ = fill(5f-1, d)
        x0_sample = UniformSampling(-∂, ∂)
        x0 = fill(0f0,d) # initial point

        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1)) # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, p, t) = a.(v_y) .- a.(v_z)

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt, mc_sample = x0_sample)
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, neumann_bc = [-∂, ∂], x0_sample = x0_sample)

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
        lossmax = maximum([loss[end] for loss in sol.losses[2:end]])
        return sol.us[end], lossmax, missing
end

if false
        d = 1
        dt = 1f-1 # time step
        T = 2f-1
        @show DeepSplitting_allencahn_neumann(d, T, dt, 6)
end