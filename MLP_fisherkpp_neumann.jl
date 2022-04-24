using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_fisherkpp_neumann(d, T, L)
        tspan = (0e0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        ∂ = 5e-1
        neumann_bc = [fill(-∂, d), fill(∂, d)]
        x0 = fill(0e0,d) # initial point
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        g(x) = exp.(-0.25e0 * sum(x.^2))   # initial condition
        f(y, z, v_y, v_z, p, t) = max.(0e0, v_y) .* 
                ( 1e0 .- max.(0e0,v_y) )

        # defining the problem
        alg = MLP(M = L, K = 1, L = L)
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, neumann_bc = neumann_bc)

        # solving
        sol = solve(prob, 
                alg, 
                multithreading=true)
        return sol.us[end]
end

if false
        d = 5
        T = 2f0
        L = 5
        @show MLP_fisherkpp_neumann(d, T, L)
end