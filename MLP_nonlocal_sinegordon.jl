using CUDA
CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_nonlocal_sinegordon(d, T, L)
        tspan = (0e0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        σ_sampling = 1e-1 / sqrt(2e0)
        x0 = fill(0e0,d) # initial point
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        g(x) = exp.(-0.25e0 * sum(x.^2))   # initial condition
        f(y, z, v_y, v_z, p, t) = sin.(v_y) .- v_z * 
                Float64(2 * π^(d/2) * σ_sampling^d)

        # defining the problem
        mc_sample = NormalSampling(σ_sampling,true)
        alg = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)

        # solving
        sol = solve(prob, alg, multithreading=true)
        return sol.us[end]
end

if false
        d = 10
        T = 3f-1
        L = 4
        @show MLP_nonlocal_sinegordon(d, T, L)
end