using CUDA
CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_fisherkpp_neumann(d, T, dt)
        tspan = (0e0,T)
        L = 5
        ##########################
        ###### PDE Problem #######
        ##########################
        U = 5f-1
        u_domain = (fill(-U, d), fill(U, d))
        x0 = fill(0e0,d) # initial point
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        g(x) = exp.(-0.25e0 * sum(x.^2))   # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = max.(0e0, v_y) .* ( 1e0 .- max.(0e0,v_y) )

        # defining the problem
        alg = MLP(M = L, K = 10, L = L)
        prob = PIDEProblem(g, f, μ, σ, tspan, neumann = u_domain, x = x0)

        # solving
        solve(prob, 
                alg, 
                multithreading=true
                )
end

if false
        d = 5
        dt = 1f-1 # time step
        T = 3f-1
        xgrid,ts,sol = MLP_fisherkpp_neumann(d, T, dt)
        @show sol[end]
end