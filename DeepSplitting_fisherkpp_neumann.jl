using HighDimPDE
using Random
using Test
using Flux
using Revise

function DeepSplitting_fisherkpp_neumann(; d, T, N, batch_size = 8000, K = 1, cuda_device=5, maxiters = 500)
        dt = T / N
        tspan = (0f0,T)

        ##############################
        #######   ML params    #######
        ##############################

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
        x0 = fill(0f0,d) # point where u(x,t) is approximated
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
        f(y, z, v_y, v_z, ∇u_x, ∇u_y, p, t) = v_y .* ( 1f0 .- v_y ) 

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
        return sol.us[end], missing
end

if false
        d = 10
        dt = 1f-1
        T = 2f-1
        @show DeepSplitting_fisherkpp_neumann(d, T, dt, 1)
end