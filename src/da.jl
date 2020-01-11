# Delayed acceptance
# NOTE: we want delayed acceptance in combination with distributed computing
# this is slightly challenging, as ideally all processes should stop once
# we decide to reject. Naive implementation of DA would imply an accept-reject
# function that is executed in parallel independently, which could be wasteful.
# still, even in that case it could lead to speed-ups (but only when there is
# rejection across all processes?)
using CSV
using Random
using Distributed
using Distributions
addprocs(2)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Beluga

tree = readline("example/dicots/dicots.nw")
data = CSV.read("example/dicots/dicots-f01-100.csv")

# 13.498667 seconds (2.44 M allocations: 463.278 MiB, 0.38% gc time)
#  7.340287 seconds (1.57 M allocations: 265.833 MiB, 0.49% gc time)  prior factorized
Random.seed!(123)
model, profile = DLWGD(tree, data)
prior = IRRevJumpPrior(
    Ψ=[1 0.0 ; 0.0 1],
    X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
    πK=DiscreteUniform(0, 10),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=Beluga.treelength(model))
kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
chain = RevJumpChain(data=profile, model=model, prior=prior, kernel=kernel)
init!(chain)
@time rjmcmc!(chain, 100, trace=5, show=10)
