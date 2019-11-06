using Test, Beluga, PhyloTrees, Parameters, Random
Random.seed!(333)

# example
t, x = Beluga.example_data1()
d1 = DuplicationLossWGD(t, rand(7), rand(7), Float64[], 1/1.5, maximum(x))

@test logpdf(d1, x) ≈ -16.100841297963
gradient(d1, x)

using Flux.Tracker
f = (u) -> logpdf(d1(u), x)
df(θ) = Tracker.gradient(f, θ; nest = true)[1]; # df/dx = 6x + 2
df(rand(15))
