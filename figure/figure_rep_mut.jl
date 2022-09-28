################################
#=
Simulating and plotting results
for the replicator equation
=#
################################

cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot, Printf
using UnPack
using JLD2
plotting = true
simu = true

if simu
        tspan = (0f0,15f-2)
        dt = 5f-2 # time step
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients
        d = 5
        ss0 = 5f-2#std g0
        ∂ = fill(5f-1, d)
        x0_sample = UniformSampling(-∂, ∂)
        x0 = fill(0f0,d) # initial point

        ##############################
        ####### ML params #######
        ##############################
        maxiters = 2000
        batch_size = 8000
        K = 5

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d, hls, tanh),
                                Dense(hls,hls,tanh),
                                Dense(hls, 1, x->x^2)) # Neural network used by the scheme

        opt = Flux.ADAM(5e-3)#optimiser

        ##########################
        ###### PDE Problem #######
        ##########################
        g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(2 * ∂)
        f(y, z, v_y, v_z, p, t) =  v_y .* (m(y) .- vol * v_z .* m(z) ) # nonlocal nonlinear part of the

        # defining the problem
        alg = DeepSplitting(nn, K=K, opt = opt,
                                mc_sample = x0_sample )
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, x0_sample = x0_sample)

        # solving
        @time sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = true, 
                        maxiters = maxiters,
                        batch_size = batch_size,
                        use_cuda = true
                        )

        ###############################
        ######### Plotting ############
        ###############################
        xgrid1 = collect((-∂[1]/2:5f-3:∂[1]/2))
        xgrid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1] 
        # Analytic solution
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
        df = DataFrame("T"=>Float64[], "u_anal" => [],  "u_approx" => [])
        for (i,t) in enumerate(collect(tspan[1]: dt : tspan[2]))
                u_anal = uanal.(xgrid, t, Ref(Dict()))
                u_approx = reduce(vcat,sol.ufuns[i].(xgrid))
                push!(df, (t,u_anal,u_approx))
        end

        # saving results
        data_paper = Dict("xgrid1" => xgrid1, "df" => df )
        using JLD2
        @save "data_paper.jld2" data_paper
end
if plotting
        # loading simulation
        @load "data_paper.jld2" data_paper;
        df = data_paper["df"]; xgrid1 = data_paper["xgrid1"];

        # plotting simulation
        fig, ax = plt.subplots(1,2, sharey = true)
        
        #Exact solution
        for (i,r) in enumerate(eachrow(df))
                ax[2].plot(xgrid1, r.u_anal, label = latexstring("t_$(i-1) = $(@sprintf("%.2f",r.T))"))
        end
        ax[2].set_title("Exact solution")

        #Deepsplitting approximation
        for (i,r) in enumerate(eachrow(df))
                ax[1].scatter(xgrid1, r.u_approx, s = 1., label = latexstring("t_$(i-1) = $(@sprintf("%.2f",r.T))"))
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

        # fig.savefig("figure_rep_mut_$(d)d.pdf", dpi=800)
end


