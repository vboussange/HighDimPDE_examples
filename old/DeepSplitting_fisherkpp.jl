cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot
using UnPack
plotting = true

##############################
####### Global parameters ####
##############################
tspan = (0f0,2f-1)
dt = 1f-1 # time step
d = 5
U = 5f-1
u_domain = repeat([-U,U]', d, 1)

##############################
####### Neural Network #######
##############################
batch_size = 5000
train_steps = 5000
K = 1

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(
        BatchNorm(d,affine = true, dim = 1),
        Dense(d, hls, tanh),
        BatchNorm(hls,affine = true, dim = 1),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine = true, dim = 1),
        # Dense(hls,hls,relu),
        Dense(hls, 1, relu)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(1e-1,
                1e-2,
                1000,
                1e-4),
                ADAM() )#optimiser

##########################
###### PDE Problem #######
##########################
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients
σ_sampling = 5f-2
g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))  # initial condition
f(y,z,v_y,v_z,∇v_y,∇v_z, t) = v_y .* ( 1f0 .- v_z * Float32(π^(d/2)) * σ_sampling^d)  # nonlocal nonlinear part of the

# defining the problem
alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = NormalSampling(σ_sampling, true)  )
prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain)
# solving
@time xgrid,ts,sol = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol = 1f-5,
                maxiters = train_steps,
                batch_size = batch_size,
                use_cuda = true,
                neumann = u_domain
                )

###############################
######### Plotting ############
###############################
if plotting
        clf()
        fig, ax = plt.subplots( sharey = true)

        xgrid1 = collect((-U:5f-3:U))
        xgrid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1] 

        #Deepsplitting sol
        for i in 1:length(sol)
                ax.scatter(xgrid1, reduce(vcat,sol[i].(xgrid)), s = .2, label="t = $(dt * (i-1))")
        end
        
        ax.set_title("DeepSplitting")
        ax.legend()
                
        savefig("fisher_kpp_$(d)d.pdf")
end
gcf()