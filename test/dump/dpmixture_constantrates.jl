using CSV
using DataFrames
using Beluga
using StatsBase
using Distributions
using Plots
using JLD
using AdaptiveMCMC
include("../src/mcmc_dp.jl")

s = SpeciesTree("test/data/tree1.nw")
Beluga.set_constantrates!(s)
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
M = profile(s, df)

G0 = MvLogNormal(log.([0.2, 0.2]), [1. 0.9 ; 0.9 1.])
prior = ConstantDPModel(G0, .1)
chain = init(prior, s, M)

# operator_switchmode_constantrates_nowgd!(M, chain);
mcmc!(chain, 5000, show_every=10)

l = vcat([[d[2][1] d[2][2] d[3][1] d[3][2]] for d in chain.trace]...)
p1 = plot(l[100:end, :], labels=[:l1, :l2, :m1, :m2]);
p2 = scatter(1:1000, [d[1] for d in chain.trace], color="black",
    alpha=0.5, markersize=1, legend=false;)
plot(p1, p2, size=(300,400), layout=(2,1))


# Branch-wise
s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts3.tsv", delim="\t")
deletecols!(df, :Orthogroup)
M = profile(s, df)

n = length(s)
a = 0.9
Σ = [I+zeros(n, n) a*I+zeros(n,n);z a* I+zeros(n, n) I+zeros(n,n)]
G0 = MvLogNormal(log.(repeat([0.2], 2*length(s))), Σ)
prior = ConstantDPModel(G0, 1.)
chain = init(prior, s, M)

#operator_switchmode_constantrates_nowgd!(M, chain);
mcmc!(chain, 5000, show_every=10)

p = plot([x[2][3, 2] for x in chain.trace][500:end])
for i=3:9
    plot!([x[2][i, 2] for x in chain.trace][500:end])
end
plot(p)

l = [x[2][4,:] for x in chain.trace]
histogram(vcat(l...))
