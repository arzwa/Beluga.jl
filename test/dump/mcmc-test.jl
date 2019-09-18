using CSV
using DataFrames
using Distributed
using Distributions
using Parameters
@everywhere using DistributedArrays
@everywhere using Beluga

s = SpeciesTree("test/data/tree1.nw")

#s = SpeciesTree("test/data/tree2.nw")
#s[3, :q] = 1
#Beluga.set_wgdrates!(s)

df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)

#M = distribute(profile(s, df))
#M = zeros(Int64, 0, 0)

prior = GBMRatesPrior(
    InverseGamma(5,1),
    Exponential(0.2),
    Exponential(0.2),
    Beta(1,1),
    Beta(10,1))



# NB always re-initialize p when starting a new chain!
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
chain = mcmc!(chain, 11000, :η, show_every=10)
CSV.write("beluga-amcmc-test.csv", chain.trace)

# setting equal rates in branches from the root
s.bindex[17, :θ] = 2
chain.model.λ = chain[:λ] = chain.model.λ[1:16]
chain.model.μ = chain[:μ] = chain.model.μ[1:16]
