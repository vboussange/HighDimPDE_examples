examples = ["allen_cahn_nonlocal", "fisher_kpp", "hamel", "merton", "mirrahimi", "nonlocal_comp", "sine_gordon"]
for (i,ex) in enumerate(examples)
    prob, mc_sample = ex
    alg_ds = DeepSplitting(nn, K=K, opt = opt, mc_sample = mc_sample )
    #evaluate ex with 
    @time sol = solve(prob, 
                alg, 
                dt=dt, 
                verbose = true, 
                abstol=2e-3,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = false)
end