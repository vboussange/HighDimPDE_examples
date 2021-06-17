
function allen_cahn_nonlocal(d, tspan, device)
        X0 = fill(0.0f0,d) |> device  # initial point
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

function fisher_kpp(d, tspan, device)
        σ_sampling = 1f-1
        X0 = fill(0.0f0,d) |> device # initial point
        g(X) = exp.(-0.5f0 * sum(X.^2,dims=1))  # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = max.(0f0, v_y) .*( 1f0 .- max.(0f0,v_z) * Float32(π^(d/2)) * σ_sampling^d ) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0f0 # advection coefficients
        σ_f(X,p,t) = 1f-1 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling / sqrt(2f0), true) # uniform distrib in u_domain
        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
        prob,mc_sample
end

function hamel(d, tspan, device)
        σ_sampling = 1f-1
        X0 = fill(0f0,d) |> device # initial point
        g(X) = Float32(2f0^(d/2))* exp.(-2f0 * Float32(π)  * sum( X.^2, dims=1))   # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0f0, v_y) .* ( m(y) - max.(0f0, v_z) .* m(z) * Float32((2f0 * π)^(d/2) * σ_sampling^d) .* exp.(5f-1 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = sqrt(1f-1) # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling) # normal

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                )
        prob, mc_sample
end


function sine_gordon(d, tspan, device)
        X0 = fill(0.0f0,d) |> device # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = sin.(v_y) .- v_z * Float32(π^(d/2) * σ_sampling^d) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = 1f0 # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling/sqrt(2f0)) # uniform distrib in u_domain

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan,
                                )
        prob, mc_sample
end


function nonlocal_comp(d, tspan, device)
        σ_sampling = 1f-1
        X0 = fill(0f0,d) |> device # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, ∇v_y ,∇v_z, t) =  max.(0f0, v_y) .* (1f0 .- max.(0f0, v_z) * Float32((2 * π )^(d/2) * σ_sampling^d))
        μ_f(X,p,t) = 0.0f0 # advection coefficients
        σ_f(X,p,t) = sqrt(1f-1) # diffusion coefficients
        mc_sample = NormalSampling(σ_sampling,true) # uniform distrib in u_domain

        # defining the problem
        prob    = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
        prob, mc_sample
end


function mirrahimi(d, tspan, device)
        σ_sampling = 1f-1
        X0 = fill(0.0f0,d) |> device # initial point
        u_domain = [-5f-1,5f-1]
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