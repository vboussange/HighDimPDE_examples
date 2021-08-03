cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot
using UnPack
plotting = true

# needed to get dimension for n dimensional tensors
function get_dim(y::AbstractArray, dim)
    return view(y,dim,ntuple(i->:,ndims(y)-1)...)
end

tspan = (0f0, 5f-1)
dt = 5f-2 # time step


d_spatial = 1
d = d_spatial + 1
γ = 1.8f-4
αs = 1.5f-7
ϕ = 1f-5
x0 = 0.5f0
dr = 1f0 # death rates
ss0 = 1f-2 # initial disribution
u_domain = repeat([0f0,1f0]', d, 1)

μ(x,p,t) = 0f0 # advection coefficients
σ(x,p,t) = [1f-1, 1f-1] |> gpu # diffusion coefficients
S(x) = 1.017f-6 * exp.(-4885f0 * ( sum(get_dim(x,2:d_spatial+1) .- x0, dims=1) .+ 1f-2).^2 ) # # Oxygen distribution
a(S) = γ * (S ./ (S .+ αs)) + ϕ ./ (ϕ .+ γ * (S ./ (S .+ αs)) )
b(S) = ϕ .+ γ * S ./ (αs .+ S)
h(S) = ϕ ./ (ϕ .+ γ * (S ./ (S .+ αs)) )
volx = 1f0
g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(- 5f-1 *sum((x .-  5f-1) .^2f0 / ss0, dims = 1)) # initial condition
# plt.plot(xgrid1, g.(xgrid))
# gcf()


function f(y, z, u, ∇u, t) 
        # integrate only on dimension 2 for ρ
        ρ = u(z)
        y1 = get_dim(y,1:1)
        _f = a(S(y)) - b(S(y)) .* (y1 - h(S(y)) ).^2f0 .- dr * ρ * volx
        return _f
end

# some tests
if false
    u(x) = sum(x,dims=1)
    x = randn(Float32,d,10)
    S(x)
    a(S(x))
    b(S(x))
    h(S(x))
    y = CUDA.randn(Float32,d,10)
    z =  CUDA.randn(Float32,d,10, 10)
    f(y, z, u, 0f0, 0f0)
end


##############################
####### Neural Network #######
##############################
batch_size = 2000
train_steps = 5000
K = 20

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(
                BatchNorm(d, affine=true, dim = 1),
                Dense(d, hls, tanh),
                BatchNorm(hls, affine=true, dim = 1),
                Dense(hls, hls, tanh),
                BatchNorm(hls, affine=true, dim = 1),
                Dense(hls, 1, relu)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(1f-2,
                10.0,
                1000,
                1f-6),
                ADAM() )#optimiser
mc_sample = UniformSampling(u_domain[1,:],u_domain[2,:], [1]) # not integrating first dimension
alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = mc_sample )

##########################
###### PDE Problem #######
##########################

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain
                    )
# solving
@time xgrid,ts,sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=5f-8,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = true,
                neumann = u_domain
                )

###############################
######### Plotting ############
###############################

if plotting
        using QuadGK
        using LaTeXStrings

        clf()
        x = range(0, 1, length = 100)
        solinf = sol[end]
        function ρ(x, sol)
                ρ, _ = quadgk(y -> sol(reshape([y,x],2,1))[], 0, 1)
                return ρ[]
        end

        function μ(x, sol)
                μ, _ = quadgk(y -> y * sol(reshape([y,x],2,1))[], 0, 1)
                return μ[]
        end

        function σ²(x, sol)
                σ², _ = quadgk(y -> y^2 * sol(reshape([y,x],2,1))[], 0, 1)
                return σ²[]
        end

        fig, ax = plt.subplots(3)

        ax[1].set_title(L"\rho(x)")
        ax[2].set_title(L"\mu(x)")
        ax[3].set_title(L"\sigma^2(x)")

        for i in 1:length(sol)
                try
                        ax[1].plot(x, ρ.(x, Ref(sol[i])), label="t = $(dt * (i-1))")
                        ax[2].plot(x, μ.(x, Ref(sol[i])), label="t = $(dt * (i-1))")
                        ax[3].plot(x, σ².(x, Ref(sol[i])), label="t = $(dt * (i-1))")
                catch e
                        println("did not work for i = $i")
                end
        end

        for _ax in ax[1:1]
                _ax.legend()
        end
        fig.tight_layout()
end
gcf()
fig.savefig("results_Chiara_2D.pdf")
