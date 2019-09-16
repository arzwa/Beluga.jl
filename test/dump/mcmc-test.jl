using CSV
using DataFrames
#using StatsBase, Plots, JLD
using PhyloTrees
using Distributed
using Distributions
using AdaptiveMCMC
using Parameters
addprocs(2)
@everywhere using DistributedArrays
@everywhere using Beluga
include("../../src/mcmc.jl")

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)

#M = distribute(profile(s, df))
#M = zeros(Int64, 0, 0)

prior = GBMRatesPrior(
    InverseGamma(5,1),
    Exponential(0.2),
    Exponential(0.2),
    Beta(1,1),
    Beta(10,1))

chain = DLChain(p, prior, s, m)

chain = mcmc!(chain, 11000, show_every=10)
CSV.write("beluga-amcmc-test.csv", chain.trace)


# AMM
prop = Proposals(
    :ν=>AdaptiveScaleProposal(0.5),
    :η=>AdaptiveUnitProposal(0.2),
    :θ=>AdaptiveMixtureProposal(17*2, σ=0.01))

chain = DLChain(M, prior, s)
chain.proposals = prop

for i=1:1000
    move_ν!(chain)
    move_η!(chain)
    amm_mcmc!(chain, :θ)
end
