L = 4 # MLP level

function nonlocal_comp(d, tspan, algtype)
        if algtype == :DS
                σ_sampling = 1f-1 / sqrt(2f0)
                x0 = fill(0f0,d) # initial point
                g_ds(X) = exp.(-0.25f0 * sum(X.^2, dims=1))   # initial condition
                f_ds(y, z, v_y, v_z, ∇v_y ,∇v_z, p, t) =  max.(0f0, v_y) .* (1f0 .- max.(0f0, v_z) * Float32((2 * π )^(d/2) * σ_sampling^d))
                μ_ds(X,p,t) = 0.0f0 # advection coefficients
                σ_ds(X,p,t) = 1f-1 # diffusion coefficients
                mc_sample = NormalSampling(σ_sampling, true) # uniform distrib in u_domain
                hls = d + 50 #hidden layer size
                nn = Flux.Chain(Dense(d,hls,tanh),
                                        Dense(hls,hls,tanh),
                                        Dense(hls,1)) # Neural network used by the scheme
                # defining the problem
                prob = PIDEProblem(g_ds, f_ds, μ_ds, σ_ds, tspan, x = x0)
                opt = ADAM(1e-3)#optimiser   
                alg = DeepSplitting(nn, K = 2, opt = opt, mc_sample = mc_sample )
                return prob, mc_sample, alg
        elseif algtype == :MLP
                σ_sampling = 1e-1 / sqrt(2e0)
                x0 = fill(0e0,d) # initial point
                g_mlp(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
                f_mlp(y, z, v_y, v_z, ∇v_y ,∇v_z, p, t) =  max.(0e0, v_y) .* (1e0 .- max.(0e0, v_z) * Float64((2 * π )^(d/2) * σ_sampling^d))
                μ_mlp(X,p,t) = 0.0e0 # advection coefficients
                σ_mlp(X,p,t) = 1e-1 # diffusion coefficients
                mc_sample = NormalSampling(σ_sampling, true) # uniform distrib in u_domain
                # defining the problem
                prob = PIDEProblem(g_mlp, f_mlp, μ_mlp, σ_mlp, tspan, x = x0)
                alg = MLP(M = L, K = 10, L = L, mc_sample = mc_sample )
                return prob, mc_sample, alg
        else
                error("only MLP or DS as alg")
        end
end

function allen_cahn(d, tspan)
        X0 = fill(0.0f0,d) # initial point
        u_domain = [-5f-1,5f-1]
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = sqrt(2f0) # diffusion coefficients
        mc_sample = UniformSampling(u_domain[1], u_domain[2]) # uniform distrib in u_domain
        # defining the problem
        prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan, u_domain = u_domain)
        prob, mc_sample
end

function fisher_kpp(d, tspan)
        σ_sampling = 1f-1
        X0 = fill(0.0f0,d) # initial point
        g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))  # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = max.(0f0, v_y) .* ( 1f0 .- max.(0f0,v_z) * Float32(π^(d/2)) * σ_sampling^d ) 
        μ_f(X,p,t) = 0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling / sqrt(2f0), true) # uniform distrib in u_domain
        # defining the problem
        prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
        prob, mc_sample
end

function fisher_kpp_reflected(d, tspan)
        u_domain = [-1f-1,1f-1]
        X0 = fill(0.0f0,d) # initial point
        g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))  # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = max.(0f0, v_y) .* ( 1f0 .- max.(0f0,v_y) ) 
        μ_f(X,p,t) = 0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NoSampling() # uniform distrib in u_domain
        # defining the problem
        prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan, u_domain = u_domain)
        prob, mc_sample
end

function hamel(d, tspan)
        σ_sampling = 1f-1
        X0 = fill(0f0,d) # initial point
        g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) - max.(0f0, v_z) .* m(z) * Float32((2f0 * π)^(d/2) * σ_sampling^d) .* exp.(5f-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling) # normal

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                )
        prob, mc_sample
end

function hamel_uniform(d, tspan)
        σ_sampling = 1f-1
        X0 = fill(0f0,d) # initial point
        g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) - max.(0f0, v_z) .* m(z) * Float32((2f0 * π)^(d/2) * σ_sampling^d) .* exp.(5f-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling) # normal

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                )
        prob, mc_sample
end


function sine_gordon(d, tspan)
        X0 = fill(0.0f0,d) # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = sin.(v_y) .- v_z * Float32(π^(d/2) * σ_sampling^d) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling/sqrt(2f0)) # uniform distrib in u_domain

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
        prob, mc_sample
end

function mirrahimi(d, tspan)
        σ_sampling = 1f-1
        X0 = fill(0.0f0,d) # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(y) = 1f0 .- 1f-1 .* sum(y.^2,dims=1)
        f(y, z, v_y, v_z, dv_y, dv_z, t) = max.(v_y , 0f0) .* (a(y) - 0.1f0 * max.(v_z , 0f0) .* (Float32(π^(d/2) * σ_sampling^d) .* exp.( sum(z.^2,dims=1)) / σ_sampling^2)) # function from solved equation
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = sqrt(2f0) # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling/sqrt(2f0)) # uniform distrib in u_domain

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
        prob, mc_sample
end
