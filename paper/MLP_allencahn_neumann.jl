using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_allencahn_neumann(d, T, dt, L)
        tspan = (0f0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        neumann = (fill(-5f-1, d), fill(5f-1, d))
        x0 = fill(0e0,d) # initial point
        g_mlp(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f_mlp(y, z, v_y, v_z, ∇v_y ,∇v_z, p, t) = a.(max.(0f0, v_y)) .- a.(max.(0f0, v_z))
        μ_mlp(X,p,t) = 0.0e0 # advection coefficients
        σ_mlp(X,p,t) = 1e-1 # diffusion coefficients
        mc_sample = UniformSampling(neumann...) # uniform distrib in u_domain
        # defining the problem
        prob = PIDEProblem(g_mlp, f_mlp, μ_mlp, σ_mlp, tspan, x = x0, 
                neumann = neumann)
        alg = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )
        # solving
        xs,ts,sol = solve(prob, 
                alg, 
                multithreading=true
                )
        return sol[end]
end

if false
        d = 1
        dt = 1f-1 # time step
        T = 3f-1
        L = 4
        @show MLP_allencahn_neumann(d, T, dt, L)
end