

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
