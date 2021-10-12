using CUDA
CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

function DeepSplitting_fisherkpp_neumann(d, T, dt)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 2000
        batch_size = 6000
        K = 10

        hls = d + 50 #hidden layer size


        nn_batch = Flux.Chain(Dense(d,hls,tanh),
                                Dense(hls,hls,tanh),
                                Dense(hls, 1, x->x^2)) # Neural network used by the scheme, with batch normalisation

        opt = Flux.ADAM(1e-2) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        u_domain = [-5f-1, 5f-1]
        x0 = fill(0f0,d) # initial point
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = max.(0f0, v_y) .* ( 1f0 .- max.(0f0,v_y) ) 

        # defining the problem
        alg = DeepSplitting(nn_batch, K=K, opt = opt)
        prob = PIDEProblem(g, f, μ, σ, tspan, neumann = u_domain, x = x0)

        # solving
        solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=1f-7,
                maxiters = maxiters,
                batch_size = batch_size,
                use_cuda = false,
                )
end

if false
        d = 5
        dt = 1f-1 # time step
        T = 2f-1
        xgrid,ts,sol = DeepSplitting_fisherkpp_neumann(d, T, dt)
        @show sol[end]
end