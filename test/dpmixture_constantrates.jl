using Test
using CSV
using DataFrames
using Beluga
using StatsBase
using Distributions
using Plots

s = SpeciesTree("test/data/tree1.nw")
Beluga.set_constantrates!(s)
df = CSV.read("test/data/counts2.tsv", delim="\t"); deletecols!(df, :Orthogroup)
M = profile(s, df)

G0 = MvLogNormal(log.([0.2, 0.2]), [1. 0.9 ; 0.9 1.])
prior = ConstantDPModel(G0, 1.)
chain = init(prior, s, M)

# operator_switchmode_constantrates_nowgd!(M, chain);
mcmc!(chain, 1000, show_every=10)

l = vcat([[d[2][1] d[2][2] d[3][1] d[3][2]] for d in chain.trace]...)
p1 = plot(l[100:end, :], labels=[:l1, :l2, :m1, :m2]);
p2 = scatter(1:1000, [d[1] for d in chain.trace], color="black",
    alpha=0.5, markersize=1, legend=false;)
plot(p1, p2, size=(900,300))
