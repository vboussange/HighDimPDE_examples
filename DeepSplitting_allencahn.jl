using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

tspan = (0f0,3f-1)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients

d = 5
u_domain = repeat([-5f-1,5f-1]',d,1)

##############################
####### Neural Network #######
##############################
batch_size = 3000
train_steps = 10000
K = 100

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        BatchNorm(hls, affine=true, dim=1),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine=true, dim=1),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                1e-2,
                2000,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = UniformSampling(u_domain[:,1], u_domain[:,2]) )

##########################
###### PDE Problem #######
##########################
g(x) = exp.(-0.25f0 * sum(x.^2, dims = 1))   # initial condition
a(u) = u - u^3

# for uniform sampling of nl term
vol = prod(u_domain[:,2] - u_domain[:,1])
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) * vol
# for random sampling of nl term
# f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) .* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2)

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                                        # u_domain = u_domain,
                                        x = fill(0f0,d))
# solving
@time xgrid, ts, sol = HighDimPDE.solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=1f-4,
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
        gcf()

        ax.set_title("DeepSplitting")
        ax.legend()
        
        gcf()
        savefig("hamel_$(d)d.pdf")

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