using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_nonlocal_comp(;d, T, M, L, K=1)
        tspan = (0f0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        σ_sampling = 1e-1 / sqrt(2)
        x0 = fill(0e0,d) # initial point
        g_mlp(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        f_mlp(y, z, v_y, v_z, ∇u_x, ∇u_y, p, t) =  max.(0e0, v_y) .* 
                (1e0 .- max.(0e0, v_z) * Float64((2 * π )^(d/2) * σ_sampling^d))
        μ_mlp(X,p,t) = 0.0e0 # advection coefficients
        σ_mlp(X,p,t) = 1e-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling, true) # uniform distrib in x0_sample
        # defining the problem
        prob = PIDEProblem(g_mlp, f_mlp, μ_mlp, σ_mlp, x0, tspan)
        alg = MLP(;M, K, L, mc_sample)
        # solving
        sol = solve(prob, alg, multithreading=true)
        return sol.us[end]
end

if false
        d = 5
        T = 3f-1
        L = 4
        @show MLP_nonlocal_comp(d, T, L)
end