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


prior = Beluga.IIDRatesPrior(
    Exponential(0.1),
    Exponential(),
    Exponential(),
    Beta(1,1),
    Beta(8,2))


prior = Beluga.ExpRatesPrior(
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
#p = Profile()
chain = DLChain(p, prior, s, m)
chain.model.λ = chain[:λ] = rand(17)
chain.model.μ = chain[:μ] = rand(17)
#chain[:ν] = 0.1
chain = mcmc!(chain, 11000, :ν, show_every=10)

d = DuplicationLoss(s,
    [0.609326, 0.0317002, 0.111777, 0.69181, 0.0461885, 0.562838, 0.187973, 0.154587, 0.126413, 0.374417, 0.367792, 0.0472703, 0.039286, 1.64182, 0.689439, 0.171681, 0.168145],
    [0.0134613, 1.50559, 1.8343, 0.169876, 0.361917, 0.282899, 0.0131346, 0.214978, 0.766745, 0.469999, 0.0635182, 0.187134, 0.711814, 0.0637716, 0.984191, 0.951305, 11.9708],  0.7235804910286543, m
     )

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
