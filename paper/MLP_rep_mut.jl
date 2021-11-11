using CUDA
CUDA.device!(6)
using HighDimPDE
using Random
using Test
using Flux
using Revise

function MLP_rep_mut(d, T, dt, L)
        tspan = (0e0,T)
        ##########################
        ###### PDE Problem #######
        ##########################
        ss0 = 5e-2#std g0
        U = 5f-1
        u_domain = (fill(-U, d), fill(U, d))
        x0 = fill(0e0,d) # initial point
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        g(x) = (2*π)^(-d/2) * ss0^(- d * 5e-1) * exp.(-5e-1 *sum(x .^2e0 / ss0)) # initial condition
        m(x) = - 5e-1 * sum(x.^2)
        vol = prod(u_domain[2] - u_domain[1])
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = max(0.0, v_y) * (m(y) -  vol * max(0.0, v_z) * m(z)) # nonlocal nonlinear part of the

        # defining the problem
        alg = MLP(M = L, K = 10, L = L, mc_sample = UniformSampling(u_domain...))
        prob = PIDEProblem(g, f, μ, σ, tspan, x = x0)

        # solving
        xs,ts,sol = solve(prob, 
                        alg, 
                        multithreading=true
                        )
        return sol[end]
end

if false
        d = 5
        dt = 1f-1 # time step
        T = 3f-1
        L = 4
        @show MLP_rep_mut(d, T, dt, L)
        
        
        # Analytic sol
        ss0 = 1e-2#std g0
        μ(X,p,t) = 0e0 # advection coefficients
        σ(X,p,t) = 1e-1 # diffusion coefficients
        function _SS(x, t, p)
                d = length(x)
                MM = σ(x, p, t) * ones(d)
                SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
                return SSt
        end

        function uanal(x, t, p)
                d = length(x)
                return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
        end
        @show uanal(zeros(d), T, nothing)
end