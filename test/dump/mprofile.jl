using Test
using StatsBase, Parameters
using Beluga, DataFrames, PhyloTrees, DistributedArrays, Distributions, CSV

base = "/home/arzwa/Beluga.jl"
s = SpeciesTree("$base/test/data/plants1.nw")
df = CSV.read("$base/test/data/plants1-100.tsv", delim=",")
deletecols!(df, :Orthogroup)

# Mixtures
# ========
prior = Beluga.IIDRatesPrior(
    Exponential(0.1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.45 ; 0.45 .5]),
    Beta(1,1),
    Beta(8,2))

prior = ConstantRatesPrior(
    MvLogNormal(log.([0.5, 0.5]), [.5 0.45 ; 0.45 .5]),
    Beta(1,1),
    Beta(8,2))

s = SpeciesTree("$base/test/data/plants1.nw")
# Beluga.set_constantrates!(s)
K = 6
p, m = MixtureProfile(df, s, K)
# init = init_finitemixture(prior, s, K, m)
mchain = MixtureChain(p, prior, s, K, m)

# move_latent_assignment!(mchain); display(mchain)
# move_clusterparams!(mchain); display(mchain)
mcmc!(mchain, 1000, show_every=2, show_trace=false)
write("test-mixture-K$K.csv", mchain.trace)

function plot_mix_cr(chain)
    p1 = plot(chain.trace[:λ11])
    for k=2:chain.K
        plot!(chain.trace[Symbol("λ$(k)1")])
    end
    p2 = plot(chain.trace[:μ11])
    for k=2:chain.K
        plot!(chain.trace[Symbol("μ$(k)1")])
    end
    plot(p1, p2)
end


# # Mixture profile
# # ===============
# K = 4
# p, m = MixtureProfile(df, s, K)
# d = [DuplicationLoss(s, rand(19), rand(19), 1/1.5, m) for i=1:K]
# for i = 1:K
#     logpdf!(d[i], p, i, -1)
#     set_L!(p, i)
# end
# @test logpdf!(d[1], p, 1, -1) ==
#     logpdf!(d[2], p, 2, -1) == logpdf!(d[3], p, 3, -1)
# @test logpdf!(d[1], p, 1, -1) == logpdf!(d, p, -1)
# logpdf_allother!(d[1], p, 1, -1)

Beluga.MixturePrior(prior, 1.)
