
function allen_cahn_nonlocal(d,tspan)
        X0 = fill(0.0f0,d)  # initial point
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
