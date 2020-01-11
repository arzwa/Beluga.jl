# Delayed acceptance
# NOTE: we want delayed acceptance in combination with distributed computing
# this is slightly challenging, as ideally all processes should stop once
# we decide to reject. Naive implementation of DA would imply an accept-reject
# function that is executed in parallel independently, which could be wasteful.
# still, even in that case it could lead to speed-ups (but only when there is
# rejection across all processes?)
using CSV
using Distributed
addprocs(2)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Beluga

tree = readline("example/dicots/dicots.nw")
data = CSV.read("example/dicots/dicots-f01-1000b.csv")
model, profile = DLWGD(tree, data)

@time logpdf!(model, profile)
# single core ~ 0.17-0.20
