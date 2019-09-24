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
df = CSV.read("test/data/counts1.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
chain.model.λ = chain[:λ] = rand(17)
chain.model.μ = chain[:μ] = rand(17)
chain = mcmc!(chain, 11000, show_every=10)

chain2 = deepcopy(chain)

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


# AMM
mutable struct AMMChain <: Chain
    X::PArray
    model::DuplicationLossWGD
    Ψ::SpeciesTree
    state::State
    proposals::Proposals
    prior::Distribution
    trace::Array{Float64,2}
    gen::Int64
end

function Distributions.logpdf(chain::AMMChain, v)
    m = chain.model([exp.(v) ; 0.9])
    logpdf(chain.prior, v) + logpdf!(m, chain.X)
end


s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts2.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
chain = DLChain(p, prior, s, m)
chain.model.λ = chain[:λ] = rand(17)
chain.model.μ = chain[:μ] = rand(17)

v = Beluga.asvector(chain.model)[1:end-1]
state = State(:θ=>log.(v), :logp=>-Inf)
prop = Proposals(:θ=>AdaptiveMixtureProposal(length(v), start=100))
pr = MvNormal(repeat([log(0.5)], length(v)), 1.)
chain = AMMChain(p, chain.model, s, state, prop, pr,
    zeros(0,length(v)+1), 0)

function amm_mcmc!(chain, p)
    prop = chain.proposals[p]
    x = prop(chain[p])
    lp = logpdf(chain, x)
    mhr = lp - chain[:logp]
    if log(rand()) < mhr
        chain[p] = x
        chain[:logp] = lp
        prop.accepted += 1
        set_L!(chain.X)
    else
        set_Ltmp!(chain.X)
    end
end


for i=1:10000
    amm_mcmc!(chain, :θ)
    chain.trace = [chain.trace ; [exp.(chain[:θ]') chain[:logp]]]
    i % 100 == 0 ?
        println(join(round.(chain.trace[end,:], digits=3), ",")) : nothing
end
