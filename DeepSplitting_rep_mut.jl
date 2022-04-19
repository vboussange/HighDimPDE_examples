using CUDA
CUDA.device!(6)
using HighDimPDE
using Random
using Test
import Flux
import Flux.tanh, Flux.relu, Flux.Dense
using Revise

function DeepSplitting_rep_mut(d, T, dt)
        tspan = (0f0,T)
        ##############################
        ####### Neural Network #######
        ##############################
        maxiters = 2000
        batch_size = 16000
        K = 10

        hls = d + 50 #hidden layer size

        # Neural network used by the scheme
        nn_batch = Flux.Chain(Dense(d,hls,relu),
                              Dense(hls,hls,relu),
                              Dense(hls, 1, x->x^2)) 

        opt = Flux.ADAM(5e-3) #optimiser



        ##########################
        ###### PDE Problem #######
        ##########################
        U = 25f-2
        u_domain = (fill(-U, d), fill(U, d))
        ss0 = 5f-2#std g0

        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        g(x) = Float32((2f0*π)^(-d/2f0)) * ss0^(- Float32(d) * 5f-1) * 
                exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(u_domain[2] - u_domain[1])
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) =  v_y .* (m(y) .- vol * v_z .* m(z))

        # reference solution
        function _SS(x, t, p)
                ss0 = 5f-2#std g0
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

        # defining the problem
        alg = DeepSplitting(nn_batch, K=K, opt = opt, 
                mc_sample = UniformSampling(u_domain[1], u_domain[2]) )
        prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain)
        # solving
        xs,ts,sol,lossmax = solve(prob, 
                alg, 
                dt, 
                verbose = true, 
                abstol=3f-6,
                maxiters = maxiters,
                batch_size = batch_size,
                use_cuda = true,
                )
        return sol[end](zeros(d))[], lossmax, rep_mut_anal(zeros(d), T, Dict())
end

if false
        d = 5
        dt = 5f-2 # time step
        T = 2f-1
        @show sol, lossmax = DeepSplitting_rep_mut(d, T, dt)

        ###############################
        ######### Plotting ############
        ###############################

        if false
                using PyPlot
                fig, ax = plt.subplots(1,2, sharey = true)
                map(a -> a.clear(), ax)

                xgrid1 = collect((-U:5f-3:U))
                xgrid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1] 

                # ax[2].plot(xgrid1, reduce(hcat,g.(xgrid))[:], label = "g(x)")

                for t in collect(tspan[1]: dt : tspan[2])
                        ys = uanal.(xgrid, t, Ref(Dict()))
                        ax[2].plot(xgrid1, reduce(hcat,ys)[:], label = "t = $t")
                end
                ax[2].set_title("Analytical solution")
                gcf()

                #Deepsplitting sol
                for i in 1:length(sol)
                        ax[1].scatter(xgrid1, reduce(vcat,sol[i].(xgrid)), 
                        s = .2, label = "t = $(dt * (i-1))")
                end
                gcf()

                ax[1].set_title("DeepSplitting")

                for _a in ax
                        _a.legend()
                end
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
end