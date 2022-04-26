################################
#=
test file to find the right neural network architecture for rep mu
=#
################################

cd(@__DIR__)
using HighDimPDE
using CUDA
using Random
using Test
using Flux
using Revise
using PyPlot, Printf
using UnPack
plotting = true
simu = true

tspan = (0f0,1f-6)
dt = 1f-6 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients
d = 10
ss0 = 5f-2#std g0
U = 5f-1
x0_sample = (fill(-U, d), fill(U, d))

##############################
####### Neural Network #######
##############################
batch_size = 16000
train_steps = 4000
K = 10

hls = d + 50 #hidden layer size

const sf = Float32((2*π*ss0)^(-d/2))
nn = Flux.Chain(Dense(d, hls, tanh),
                # Dense(hls, hls, tanh),
                Dense(hls, hls, tanh),
                Dense(hls, 1, x -> x^2)) # Neural network used by the scheme
nn_batch = Flux.Chain(Dense(d, hls),
                        BatchNorm(hls, tanh),
                        Dense(hls,hls),
                        BatchNorm(hls, tanh),
                        Dense(hls, 1, x -> x^2))
opt = ADAM()
alg = DeepSplitting(nn, K=K, opt = opt, λs = [5e-3, 1e-3], mc_sample = UniformSampling(x0_sample[1],x0_sample[2]) )

##########################
###### PDE Problem #######
##########################
g(x) = sf * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
m(x) = - 5f-1 * sum(x.^2, dims=1)
const vol = prod(x0_sample[2] - x0_sample[1])
f(y, z, v_y, v_z, p, t) =  v_y .* (m(y) .- vol * v_z .* m(z)) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                x0_sample = x0_sample
                )
# solving
@time xgrid,ts,sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                # abstol = 1f0,
                maxiters = train_steps,
                batch_size = batch_size,
                use_cuda = true,
                cuda_device = 0
                )

###############################
######### Plotting ############
###############################
xgrid1 = collect((-U:5f-3:U))
xgrid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1] 
# Analytic sol
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

using DataFrames
df = DataFrame("T"=>Float64[], "y_anal" => [],  "y_approx" => [])
for (i,t) in enumerate(collect(tspan[1]: dt : tspan[2]))
        y_anal = uanal.(xgrid, t, Ref(Dict()))
        y_approx = reduce(vcat,sol[i].(xgrid))
        push!(df, (t,y_anal,y_approx))
end

# plotting simulation
fig, ax = plt.subplots(1,2, sharey = true)

for (i,r) in enumerate(eachrow(df))
        ax[2].plot(xgrid1, r.y_anal, label = latexstring("t_$(i-1) = $(@sprintf("%.2f",r.T))"))
end
ax[2].set_title("Exact solution")
#Deepsplitting sol
for (i,r) in enumerate(eachrow(df))
        ax[1].scatter(xgrid1, r.y_approx, s = 1., label = latexstring("t_$(i-1) = $(@sprintf("%.2f",r.T))"))
end
ax[1].set_title("Approximate solution")
for _a in ax[1:1]
        _a.legend()
end
ax[1].set_ylabel(L"u(t,(x,0,\dots,0))")
ax[1].set_xlabel(L"x")
ax[2].set_xlabel(L"x")

fig.tight_layout()
display(fig)