using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_allencahn_neumann(;d, T, M, L, K=1)
        tspan = (0f0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        neumann_bc = [fill(-5e-1, d), fill(5e-1, d)]
        x0 = fill(0e0,d) # initial point
        g_mlp(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f_mlp(y, z, v_y, v_z, ∇u_x, ∇u_y, p, t) = a.(max.(0f0, v_y)) .- a.(max.(0f0, v_z))
        μ_mlp(X,p,t) = 0.0e0 # advection coefficients
        σ_mlp(X,p,t) = 1e-1 # diffusion coefficients
        mc_sample = UniformSampling(neumann_bc...) # uniform distrib in x0_sample
        # defining the problem
        prob = PIDEProblem(g_mlp, f_mlp, μ_mlp, σ_mlp, x0, tspan,
                           neumann_bc = neumann_bc)
        alg = MLP(;M, K, L, mc_sample)
        # solving
        sol = solve(prob, alg, multithreading=true)
        return sol.us[end]
end

if false
        d = 1
        T = 3f-1
        L = 4
        @show MLP_allencahn_neumann(d, T, L)
end