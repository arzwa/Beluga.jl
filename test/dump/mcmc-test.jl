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

#n = 1000
#idx = rand(1:size(df)[1], n)
#df = df[idx, :]

prior = GBMRatesPrior(
    InverseGamma(5,1),
    Exponential(0.5),
    Exponential(0.5),
    Beta(1,1),
    Beta(8,2))

prior = GBMRatesPrior(
    InverseGamma(5,1),
    LogUniform(-8,3),
    LogUniform(-8,3),
    Beta(1,1),
    Beta(8,2))

# NB always re-initialize p when starting a new chain!
#p = Profile()

# fix eta
#chain[:η] = 0.8
#chain.model.η = 0.8
# chain[:ν] = 0.1


p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
#s[17, :θ] = 2
chain.model.λ = chain[:λ] = rand(17)
chain.model.μ = chain[:μ] = rand(17)
chain = mcmc!(chain, 11000, show_every=10)
#CSV.write("beluga-amcmc-test.csv", chain.trace)

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
Beluga.set_constantrates!(s)
p, m = Profile(df, s)
prior = ConstantRatesPrior(
    LogUniform(-8,3), LogUniform(-8,3), Beta(8,1), Beta(8,2))
chain = DLChain(p, prior, s, m)
chain = mcmc!(chain, 11000, :η, show_every=10)

p, m = Profile(df, s)
prior = ConstantRatesPrior(
    Exponential(1), Exponential(1), Beta(1,1), Beta(6,2))
chain2 = DLChain(p, prior, s, m)
chain2 = mcmc!(chain2, 11000, show_every=10)
