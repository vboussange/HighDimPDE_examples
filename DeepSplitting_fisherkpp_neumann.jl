using CUDA
# CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function DeepSplitting_fisherkpp_neumann(d, T, dt, cuda_device)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 1000
        batch_size = 400
        K = 1

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d,hls,tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls, 1, x->x^2)) 

        opt = Flux.ADAM(1e-2) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        neumann_bc = [fill(-5f-1, d), fill(5f-1, d)]
        x0 = fill(0f0,d) # initial point
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
        f(y, z, v_y, v_z, p, t) = max.(0f0, v_y) .* ( 1f0 .- max.(0f0,v_y) ) 

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt)
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, neumann_bc = neumann_bc)

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
        return sol.us[end], lossmax, missing
end

if false
        d = 10
        dt = 1f-1 # time step
        T = 2f-1
        @show DeepSplitting_fisherkpp_neumann(d, T, dt, 1)
end