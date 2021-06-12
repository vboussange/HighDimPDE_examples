cd(@__DIR__)
using Revise
using HighDimPDE
using Random
# Random.seed!(100)

## Basic example
d = 1 # number of dimensions
# one-dimensional heat equation
tspan = (0.0,1)
σ_sampling = 0.1

X0 = fill(0.,d)  # initial point

g(X) = 2.0^(d/2)* exp(-2. * π  * sum( X.^2))   # initial condition
m(x) = - 0.5 * sum(x.^2)
f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients
mc_sample(x) = randn(d) * σ_sampling #montecarlo samples

# defining the problem
prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                    # u_domain=[-1f0,1f0]
                     )

# using the Deep Splitting algorithm
alg = MLP(M=4, K=10, L = 4 )


# solving
@time sol = solve(prob, alg,mc_sample, verbose = false)
println("u1 = ", sol)