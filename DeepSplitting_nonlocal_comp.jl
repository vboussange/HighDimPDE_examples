using CUDA
# CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function DeepSplitting_nonlocal_comp(d, T, dt, cuda_device)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 500
        batch_size = 8000
        K = 5

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d,hls,tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls, 1, x->x^2)) 

        opt = Flux.ADAM(1e-2) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        σ_sampling = 1f-1 / sqrt(2f0)
        x0 = fill(0f0,d) # initial point
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
        _scale = Float32((2 * π )^(d/2) * σ_sampling^d)
        f(y, z, v_y, v_z, p, t) =  v_y .* (1f0 .-  v_z * _scale)

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt, 
                mc_sample = NormalSampling(σ_sampling, true))
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)
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
        lossmax = maximum(maximum.(sol.losses[2:end]))
        return sol.us[end],lossmax, missing
end

if false
        d = 10
        dt = 1f-1 # time step
        T = 2f-1
        @show DeepSplitting_nonlocal_comp(d, T, dt, 1)
end