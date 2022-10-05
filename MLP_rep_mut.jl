using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_rep_mut(;d, T, M, L, K=1)
        tspan = (0e0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        ss0 = 5e-2#std g0
        U = 5e-1
        mc_sample_∂ = (fill(-U, d), fill(U, d))
        x0 = fill(0e0,d) # initial point
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        g(x) = (2*π)^(-d/2) * ss0^(- d * 5e-1) * 
                exp.(-5e-1 *sum(x .^2e0 / ss0)) # initial condition
        m(x) = - 5e-1 * sum(x.^2)
        vol = prod(mc_sample_∂[2] - mc_sample_∂[1])
        f(y, z, v_y, v_z, ∇u_x, ∇u_y, p, t) = max(0.0, v_y) * 
                (m(y) -  vol * max(0.0, v_z) * m(z))

        # defining the problem
        alg = MLP(;M, K, L, mc_sample = UniformSampling(mc_sample_∂...))
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)

        # solving
        sol = solve(prob, alg, multithreading=true )

        # reference solution
        function _SS(x, t, p)
                d = length(x)
                MM = σ(x, p, t) * ones(d)
                SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* 
                        cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
                return SSt
        end
        
        function rep_mut_anal(x, t, p)
                d = length(x)
                return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * 
                        exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
        end

        return sol.us[end], rep_mut_anal(zeros(d), T, Dict())
end

if false
        d = 5
        T = 5e-1
        L = M = 4 
        K = 5

        @show MLP_rep_mut(;d, T, L, M, K)
        
        
        # Analytic sol
        ss0 = 5e-2#std g0
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        function _SS(x, t, p)
                d = length(x)
                MM = σ(x, p, t) * ones(d)
                SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* 
                        cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
                return SSt
        end

        function uanal(x, t, p)
                d = length(x)
                return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * 
                        exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
        end
        @show uanal(zeros(d), T, nothing)
end