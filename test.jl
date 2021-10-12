import Base.PermutedDimsArray
using Flux
using CUDA

BN_custom = Flux.Chain(args -> PermutedDimsArray(args, dims),
            BatchNorm(d),
            args -> PermutedDimsArray(args, invperm(dims)))

d = 10
x = randn(d,5,2)
dims = [1,3,2]
BN_custom(x, dims)