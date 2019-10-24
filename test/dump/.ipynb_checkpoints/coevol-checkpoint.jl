using Test
using StatsBase, Parameters, LinearAlgebra
using Beluga, DataFrames, PhyloTrees, DistributedArrays, Distributions, CSV
using AdaptiveMCMC
import AdaptiveMCMC.Proposals, AdaptiveMCMC.CoevolProposals
import Beluga.Prior, Beluga.State, Beluga.AbstractProfile, Beluga.MPArray



# chain
base = "/home/arzwa/Beluga.jl"
s = SpeciesTree("$base/test/data/plants1.nw")
df = CSV.read("$base/test/data/plants1-100.tsv", delim=",")
deletecols!(df, :Orthogroup)

K = 1
ν = 1
p, m = MixtureProfile(df, s, K)
prior = MvBMPrior(
    [ν 0. ; 0. ν],
    [Normal(log(0.2), 0.5), Normal(log(0.2), 0.5)],
    Beta(10,2),
    Dirichlet(K, 10.))

#p = MixtureProfile(K, 10)
chain = MixtureChain(p, prior, s, K, m)
_mcmc!(chain, 1000)


function plotX(chain, k=:X1, nodes=collect(1:6); f=identity)
    n = pop!(nodes)
    p1 = plot(f.([x[k][n,1] for x in chain.t]))
    for m in nodes
        plot!(f.([x[k][m,1] for x in chain.t]))
    end
    p2 = plot(f.([x[k][n,2] for x in chain.t]))
    for m in nodes
        plot!(f.([x[k][m,2] for x in chain.t]))
    end
    plot(p1, p2)
end




move_X!(chain, 1); move_X!(chain, 2); @show chain[:X1]
logprior(chain)
@unpack Y, J, A, q, n = getpics(chain[:X2], s)


# prior
t, x = Beluga.example_data1()
ν = 0.01
p = MvBMPrior([ν 0.5ν ; 0.5ν ν], [Normal(), Normal()],
Beta(5,2), Dirichlet(2, 1))
θ = rand(p, t)
logpdf(p, θ)

# likelihood
r = Beluga.branchrates(exp.(θ.X), t.tree)
d = DuplicationLoss(t, r[:,1], r[:,2], θ.η, maximum(x))
logpdf(d, x)
