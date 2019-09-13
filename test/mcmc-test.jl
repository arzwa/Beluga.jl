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
#@everywhere using DistributedArrays
@everywhere using Beluga

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
M = profile(s, df)
#M = distribute(profile(s, df))
#M = zeros(Int64, 0, 0)

prior = GBMRatesPrior(
    InverseGamma(5,1), Exponential(0.2), Exponential(0.2), Beta(10,1))
chain = DLChain(M, prior, s)

chain = mcmc!(chain, 11000, show_every=10)
CSV.write("beluga-amcmc-test.csv", chain.trace)


# AMM
prior = LogRatesPrior(InverseGamma(5,1), Normal(), Normal(), Beta(10,1))
prop = Proposals(
    :ν=>AdaptiveScaleProposal(0.5),
    :η=>AdaptiveUnitProposal(0.2),
    :θ=>AdaptiveMixtureProposal(17*2, σ=0.001, start=100))

chain = DLChain(M, prior, s)
chain.proposals = prop
chain[:θ] = [chain[:λ] ; chain[:μ]]

for i=1:1000
    chain.gen +=1
    Beluga.move_ν!(chain)
    Beluga.move_η!(chain)
    amm_mcmc!(chain, :θ)
    Beluga.log_mcmc(chain, stdout, true, 1)
end
