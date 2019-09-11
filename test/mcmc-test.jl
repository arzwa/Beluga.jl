using CSV
using DataFrames
using StatsBase
using Plots
using JLD
using PhyloTrees
using Distributed
using Distributions
using AdaptiveMCMC
#addprocs(2)
@everywhere using DistributedArrays
@everywhere using Beluga

s = SpeciesTree("test/data/tree1.nw")
Beluga.set_constantrates!(s)
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
M = distribute(profile(s, df))

prior = GBMRatesPrior(
    InverseGamma(5,1), Exponential(0.2), Exponential(0.2), Beta(10,1))
chain = DLChain(M, prior, s)

chain = mcmc!(chain, 11000, show_every=1)
CSV.write("beluga-amcmc-test.csv", chain.trace)
