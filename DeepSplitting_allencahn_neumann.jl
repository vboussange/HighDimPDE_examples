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
        maxiters = 2000
        batch_size = 8000
        K = 1

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls, 1, x->x^2)) 

        opt = Flux.ADAM(1e-3) #optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        u_domain = (fill(-5f-1, d), fill(5f-1, d))
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1)) # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = a.(v_y) .- a.(v_z)

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt, 
                mc_sample=UniformSampling(u_domain[1],u_domain[2]))
        prob = PIDEProblem(g, f, μ, σ, tspan, neumann = u_domain, u_domain = u_domain)

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
        return sol[end](zeros(d))[], lossmax, missing
end

if false
        d = 10
        dt = 1f-1 # time step
        T = 1f-1
        @show DeepSplitting_allencahn_neumann(d, T, dt)
end