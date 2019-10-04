using Test
using StatsBase, Parameters
using Beluga, DataFrames, PhyloTrees, DistributedArrays, Distributions, CSV
import Beluga: AbstractProfile, minfs, Chain, State, RatesPrior, Proposals
import Beluga: get_defaultproposals, log_mcmc!
import AdaptiveMCMC: consider_adaptation!
include("../../src/mprofile.jl")
include("../../src/mixture.jl")

s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-100.tsv", delim=",")
deletecols!(df, :Orthogroup)
K = 4
p, m = MixtureProfile(df, s, K)

# Mixture profile
# ===============
d = [DuplicationLoss(s, rand(19), rand(19), 1/1.5, m) for i=1:K]
for i = 1:K
    logpdf!(d[i], p, i, -1)
    set_L!(p, i)
end
@test logpdf!(d[1], p, 1, -1) ==
    logpdf!(d[2], p, 2, -1) == logpdf!(d[3], p, 3, -1)
@test logpdf!(d[1], p, 1, -1) == logpdf!(d, p, -1)
logpdf_allother!(d[1], p, 1, -1)


# Mixtures
# ========
prior = Beluga.IIDRatesPrior(
    Exponential(1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.25 ; 0.25 .5]),
    Beta(1,1),
    Beta(8,2))

K = 1
p, m = MixtureProfile(df, s, K)
init = init_finitemixture(prior, s, K, m)
mchain = MixtureChain(p, prior, s, K, m)

mcmc!(mchain, 1000)
