#= 
Testing neural network structure 
in approximating an gaussian function
=#
cd(@__DIR__)
using Flux, PyPlot, LaTeXStrings
using CUDA
CUDA.device!(7)
for d in [1,2,5,10]
    # d = 5
    g(x) = 5. .- sum(x.^2, dims = 1) # function to be approximated

    # ss0 = 5f-2#std g0
    # const sf = Float32((2*π*ss0)^(-d/2))
    # g(x) = sf * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # function to be approximated

    hls = d+50 #hidden layer size
    nn = Flux.Chain(Dense(d, hls, tanh),
                    Dense(hls, hls, tanh),
                    # Dense(hls, hls, tanh),
                    Dense(hls, 1)) |> gpu # Neural network

    function loss(x)
        sum((nn(x) - g(x)).^2) / length(x)
    end 

    ########################
    # training parameters  #
    ########################
    ps = Flux.params(nn)
    maxiters = 5000
    batch_size = 1000
    optimizers = [ADAM(0.001), ADAM(0.0001)]
    losses = []

    for opt in optimizers
        println(opt)
        for epoch in 1:maxiters
            # x = randn(d, batch_size) * 0.05

            # generating uniformly distributed variables on the hypercube [-0.5,0.5]^2
            x = CUDA.rand(d, batch_size) .- 5e-1

            # computing gradient
            gs = Flux.gradient(ps) do
                loss(x)
            end

            # updating parameters of the neural network given the gradient
            Flux.Optimise.update!(opt, ps, gs) 

            # logging loss for plotting later on
            push!(losses,loss(x))

            if epoch % 100 == 0
                println("Current loss is:", losses[end])
            end
        end
    end


    xplot = -0.5:0.01:0.5
    xgrid = [[zeros(d-1);x] for x in xplot]

    nn_plot = nn |> cpu
    fig, axs = subplots(1,2, figsize= (8,5)); 
    axs[1].plot(xplot, g.(xgrid), label = "True function "*L"g(x) = 5 -||x||^2"); 
    axs[1].plot(xplot, nn_plot.(xgrid), label = "Neural network approximation"); 
    axs[1].set_xlabel(L"x_1"); axs[1].set_ylabel(L"f(x_1,0,\dots,0)")
    axs[1].legend()
    axs[1].set_title("d = $d")
    axs[2].plot(1:length(losses), losses); 
    axs[2].set_yscale("log"); 
    axs[2].set_ylabel("Loss function")
    axs[2].set_xlabel("Iterations")

    fig.tight_layout()
    display(fig)
    fig.savefig("nn_approx_g(x)=-||x||^2+5_d=$d.png", dpi = 500)
end
