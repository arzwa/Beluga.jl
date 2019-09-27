using CSV
using DataFrames
using Distributed
using Distributions
using DistributedArrays
using Beluga

# NB always re-initialize p when starting a new chain!

prior1 = GBMRatesPrior(
    InverseGamma(5,1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.25 ; 0.25 .5]),
    Beta(1,1),
    Beta(8,2))

prior2 = Beluga.IIDRatesPrior(
    Exponential(0.1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.25 ; 0.25 .5]),
    Beta(1,1),
    Beta(8,2))

prior3 = Beluga.NhRatesPrior(
    MvLogNormal(log.([0.5, 0.5]), [.5 0.25 ; 0.25 .5]),
    Beta(1,1),
    Beta(8,2))


# DL model
# ========
s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-10.tsv", delim=",")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
#p = Profile()  # prior alone
chain = DLChain(p, prior2, s, m)
chn = mcmc!(chain, 11000, show_every=100)


# DL+WGD
# ======
s = SpeciesTree("test/data/hexapods1.nw")
addwgd!(s, [:bmo, :pra], 1.5, 1)
df = CSV.read("test/data/hexapods1-10.tsv", delim="\t")
deletecols!(df, :Orthogroup)

# iid prior
p, m = Profile(df, s)
chain = DLChain(p, prior2, s, m)
res = mcmc!(chain, 11000, show_every=10)

# gbm prior
p, m = Profile(df, s)
chain2 = DLChain(p, prior1, s, m)
res2 = mcmc!(chain2, 11000, show_every=10)

chns = chainscat(res, res2)
plot(chns)


# Constant rates model
# ====================
