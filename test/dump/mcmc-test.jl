using CSV
using DataFrames
using Distributed
using Distributions
@everywhere using DistributedArrays
@everywhere using Beluga

prior = GBMRatesPrior(
    InverseGamma(5,1),
    LogUniform(-8,3),
    LogUniform(-8,3),
    Beta(1,1),
    Beta(8,2))


prior = GBMRatesPrior(
    Exponential(0.1),
    Exponential(),
    Exponential(),
    Beta(1,1),
    Beta(8,2))


# NB always re-initialize p when starting a new chain!
#p = Profile()

# fix eta
#chain[:η] = 0.8
#chain.model.η = 0.8
# chain[:ν] = 0.1

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
chain.model.λ = chain[:λ] = rand(17)
chain.model.μ = chain[:μ] = rand(17)
chain = mcmc!(chain, 11000, show_every=10)


s = SpeciesTree("test/data/tree2.nw")
s[3, :q] = 1
Beluga.set_wgdrates!(s)
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
chain = mcmc!(chain, 11000, show_every=10)


s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
s[17, :θ] = 2
s[12, :θ] = 2
s[5, :θ] = 2
chain.model.λ = chain[:λ] = rand(14)
chain.model.μ = chain[:μ] = rand(14)
chain = mcmc!(chain, 11000, show_every=10)


s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
Beluga.set_constantrates!(s)
p, m = Profile(df, s)
prior = ConstantRatesPrior(
    Exponential(1), Exponential(1), Beta(1,1), Beta(6,2))
chain2 = DLChain(p, prior, s, m)
chain2 = mcmc!(chain2, 11000, show_every=10)


function plot_chain(chain, s, burnin=1000)
    cols = [x for x in names(chain.trace) if startswith(string(x), string(s))]
    p = [plot(chain.trace[burnin:end, cols[i]], legend=false) for
            i=1:length(cols)]
    plot(p...)
end
