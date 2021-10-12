using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_nonlocal_comp(d, T, dt)
        tspan = (0f0,T)
        L = 4
        ##########################
        ###### PDE Problem #######
        ##########################
        σ_sampling = 1e-1 / sqrt(2e0)
        x0 = fill(0e0,d) # initial point
        g_mlp(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        f_mlp(y, z, v_y, v_z, ∇v_y ,∇v_z, p, t) =  max.(0e0, v_y) .* (1e0 .- max.(0e0, v_z) * Float64((2 * π )^(d/2) * σ_sampling^d))
        μ_mlp(X,p,t) = 0.0e0 # advection coefficients
        σ_mlp(X,p,t) = 1e-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling, true) # uniform distrib in u_domain
        # defining the problem
        prob = PIDEProblem(g_mlp, f_mlp, μ_mlp, σ_mlp, tspan, x = x0)
        alg = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )
        # solving
        solve(prob, 
                alg, 
                multithreading=true
                )
end

if false
        d = 1
        dt = 1f-1 # time step
        T = 3f-1
        xgrid,ts,sol = MLP_nonlocal_comp(d, T, dt)
end