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
function get_dim(y::Array, dim)
    return view(y,dim,ntuple(i->:,ndims(y)-1)...)
end

tspan = (0f0,1f0)
dt = 1f-1 # time step


d_spatial = 1
d = d_spatial + 1
γ = 1.8f-4
αs = 1.5f-7
ϕ = 1f-5
x0 = fill(0.5f0,d_spatial)
dr = 1f0 # death rates
ss0 = 1f-2 # initial disribution
u_domain = repeat([0f0,1f0]', d, 1)

μ(x,p,t) = 0f0 # advection coefficients
σ(x,p,t) = [1f-1, 1f-1] # diffusion coefficients
S(x) = 1.017f-6 * exp.(-4885f0 * ( sum(get_dim(x,2:d_spatial) .- x0, dims=1) .+ 1f-2).^2 ) # # Oxygen distribution
a(S) = γ * (S ./ (S .+ αs)) + ϕ ./ (ϕ .+ γ * (S ./ (S .+ αs)) )
b(S) = ϕ .+ γ * S ./ (αs .+ S)
h(S) = ϕ ./ (ϕ .+ γ * (S ./ (S .+ αs)) )
volx = 1f0
g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(- 5f-1 *sum((x .-  5f-1) .^2f0 / ss0, dims = 1)) # initial condition
# plt.plot(xgrid1, g.(xgrid))
# gcf()


function f(y, z, u, ∇u, t) 
        # integrate only on dimension 2 for ρ
        z2 = get_dim(z, 2:d_spatial + 1)
        y1 = repeat(get_dim(y, 1:1), outer = (1, 1, size(z2,3)))
        y1z2 = reduce(vcat,[y1,z2])
        ρ = u(y1z2)
        _f = a(S(y)) - b(S(y)) .* (get_dim(y,1:1) - h(S(y)) ).^2f0 .- dr * ρ * volx
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
batch_size = 1000
train_steps = 2000
K = 100

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(
                BatchNorm(d, affine=true, dim = 1),
                Dense(d, hls, tanh),
                BatchNorm(hls, affine=true, dim = 1),
                Dense(hls, hls, tanh),
                BatchNorm(hls, affine=true, dim = 1),
                Dense(hls, 1, relu)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                10.0,
                1000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = UniformSampling(u_domain[1,:],u_domain[2,:]) )

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
                use_cuda = false
                )

###############################
######### Plotting ############
###############################
if plotting
        clf()
        fig, ax = plt.subplots(1, sharey = true)

        xgrid1 = collect((u_domain[1,1]:1f-2:u_domain[1,2]))
        xgrid = [vcat(x, fill(5f-1,d-1)) for x in xgrid1] 

        for i in 1:length(sol)
                ax.scatter(xgrid1, reduce(vcat,sol[i].(xgrid)), s = .2, label="t = $(dt * (i-1))")
        end
        gcf()

        ax.set_title("DeepSplitting")
        ax.legend()
        gcf()

        #####
        # other DimensionMismatch
        #####
        if false
                dx = 0.05
                x = u_domain[1,1]:dx:u_domain[1,2]
                plt.contourf(x,x,g.(repeat(x,2)))
        end
end
gcf()
