# Adaptive MWG-MCMC for DL(+WGD) model
# TODO this is basically the same as in Whale; so a common abstraction layer
# would be nice (idem for the SpeciesTree & SLicedTree)
abstract type Chain end
abstract type RatesPrior end
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}
const State = Dict{Symbol,Union{Vector{Float64},Float64}}
Distributions.logpdf(x::Real, y) = 0.  # hack for constant priors

mutable struct DLChain <: Chain
    X::PArray
    model::DuplicationLossWGD
    Ψ::SpeciesTree
    state::State
    priors::RatesPrior
    proposals::Proposals
    trace::DataFrame
    gen::Int64
end

Base.getindex(w::Chain, s::Symbol) = w.state[s]
Base.getindex(w::Chain, s::Symbol, i::Int64) = w.state[s][i]
Base.setindex!(w::Chain, x, s::Symbol) = w.state[s] = x
Base.setindex!(w::Chain, x, s::Symbol, i::Int64) = w.state[s][i] = x
Base.display(io::IO, w::Chain) = print("$(typeof(w))($(w.state))")
Base.show(io::IO, w::Chain) = write(io, "$(typeof(w))($(w.state))")

function DLChain(X::PArray, prior::RatesPrior, tree::SpeciesTree, m::Int64)
    init = rand(prior, tree)
    proposals = get_defaultproposals(init)
    trace = DataFrame()
    gen = 0
    model = DuplicationLossWGD(tree, init[:λ], init[:μ], init[:q], init[:η], m)
    return DLChain(X, model, tree, init, prior, proposals, trace, gen)
end

function get_defaultproposals(x::State)
    proposals = Proposals()
    for (k, v) in x
        if k ∈ [:logπ, :logp]
            continue
        elseif k == :q
            proposals[k] = [AdaptiveUnitProposal(0.2) for i=1:length(v)]
        elseif typeof(v) <: AbstractArray
            proposals[k] = [AdaptiveScaleProposal(0.1) for i=1:length(v)]
        elseif k == :ν
            proposals[k] = AdaptiveScaleProposal(0.5)
        elseif k == :η
            proposals[k] = AdaptiveUnitProposal(0.2)
        end
    end
    return proposals
end


# priors
# ======
# this is the model without correlation of λ and μ
struct GBMRatesPrior <: RatesPrior
    dν::Prior
    dλ::Prior
    dμ::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::GBMRatesPrior, tree::Arboreal)
    # assumed single prior for q for now, easy to adapt though
    @unpack dν, dλ, dμ, dq, dη = d
    ν = rand(dν)
    η = rand(dη)
    λ0 = rand(dλ)
    μ0 = rand(dμ)
    # HACK not for general rate indices!
    n = [n for n in preorder(tree) if !(iswgd(tree, n) || iswgdafter(tree, n))]
    λ = rand(GBM(tree, λ0, ν))[n]
    μ = rand(GBM(tree, λ0, ν))[n]
    q = rand(dq, Beluga.nwgd(tree))
    return State(:ν=>ν, :η=>η, :λ=>λ, :μ=>μ, :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

"""
Example: `logpdf(gbm, (Ψ=t, ν=0.2, λ=rand(17), μ=rand(17), η=0.8))`
"""
function logprior(d::GBMRatesPrior, θ::NamedTuple)
    @unpack Ψ, ν, λ, μ, q, η = θ
    @unpack dν, dλ, dμ, dq, dη = d
    lp  = logpdf(dν, ν) + logpdf(dλ, λ[1]) + logpdf(dμ, μ[1]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    lp += logpdf(GBM(Ψ, λ[1], ν), λ)
    lp += logpdf(GBM(Ψ, μ[1], ν), μ)
    return lp
end

function logprior(chain::DLChain, args...)
    s = deepcopy(chain.state)
    for (k,v) in args
        if haskey(s, k)
            length(v) == 2 ? s[k][v[1]] = v[2] : s[k] = v
        end
    end
    θ = (Ψ=chain.Ψ, ν=s[:ν], λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η])
    logprior(chain, θ)
end

logprior(c::DLChain, θ::NamedTuple) = logprior(c.priors, θ)

struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

Distributions.logpdf(d::LogUniform, x::T) where T<:Real =
    logpdf(Uniform(d.a, d.b), log10(x))

Base.rand(d::LogUniform) = exp(rand(Uniform(d.a, d.b)))

# mcmc
# ====
function mcmc!(chain::DLChain, n::Int64, args...;
        show_every=100, show_trace=true)
    wgds = Beluga.nwgd(chain.Ψ) > 0
    for i=1:n
        chain.gen += 1
        :ν in args ? nothing : move_ν!(chain)  # could be more elegant
        :η in args ? nothing : move_η!(chain)
        move_rates!(chain)
        if wgds
            move_q!(chain)
            move_wgds!(chain)
        end
        log_mcmc(chain, stdout, show_trace, show_every)
    end
    return chain
end

function move_ν!(chain::DLChain)
    prop = chain.proposals[:ν]
    ν_, hr = prop(chain[:ν])
    p_ = logprior(chain,
        (Ψ=chain.Ψ, λ=chain[:λ], μ=chain[:μ], q=chain[:q], η=chain[:η], ν=ν_))
    mhr = p_ - chain[:logπ] + hr
    if log(rand()) < mhr
        chain[:logπ] = p_
        chain[:ν] = ν_
        prop.accepted += 1
    end
    consider_adaptation!(prop, chain.gen)
end

function move_η!(chain::DLChain)
    prop = chain.proposals[:η]
    η_, hr = prop(chain[:η])

    # prior
    p_ = logprior(chain,
        (Ψ=chain.Ψ, λ=chain[:λ], μ=chain[:μ], η=η_, q=chain[:q], ν=chain[:ν]))

    # likelihood
    d = deepcopy(chain.model)
    d[:η] = η_
    l_ = logpdf!(d, chain.X, 1)  # XXX assume root is node 1

    mhr = p_ + l_ - chain[:logπ] - chain[:logp]
    if log(rand()) < mhr
        set_L!(chain.X)    # update L matrix
        chain.model = d
        chain[:logp] = l_
        chain[:logπ] = p_
        chain[:η] = η_
        prop.accepted += 1
    else
        set_Ltmp!(chain.X)  # revert Ltmp matrix
    end
    consider_adaptation!(prop, chain.gen)
end

function move_rates!(chain::DLChain)
    tree = chain.Ψ
    for i in postorder(chain.Ψ)
        iswgdafter(chain.Ψ, i) || iswgd(chain.Ψ, i) ? continue : nothing
        idx = tree[i,:θ]
        prop = chain.proposals[:λ,idx]
        λi, hr1 = prop(chain[:λ,idx])
        μi, hr2 = prop(chain[:μ,idx])

        # likelihood
        d = deepcopy(chain.model)
        d[:λ, i] = λi   # NOTE: implementation of setindex! uses node indices!
        d[:μ, i] = μi   # NOTE: implementation of setindex! uses node indices!
        l_ = logpdf!(d, chain.X, i)

        # prior
        p_  = logprior(chain,
            (Ψ=chain.Ψ, λ=d.λ, μ=d.μ, q=chain[:q], η=chain[:η], ν=chain[:ν]))

        l = chain[:logp]
        p = chain[:logπ]

        mhr = l_ + p_ - l - p + hr1 + hr2
        if log(rand()) < mhr
            set_L!(chain.X)    # update L matrix
            chain.model = d
            chain[:λ, idx] = λi
            chain[:μ, idx] = μi
            chain[:logp] = l_
            chain[:logπ] = p_
            prop.accepted += 1
        else
            set_Ltmp!(chain.X)  # revert Ltmp matrix
        end
        consider_adaptation!(prop, chain.gen)
    end
end

function move_q!(chain::DLChain)
    tree = chain.Ψ
    for i in wgdnodes(tree)
        idx = tree[i,:q]
        prop = chain.proposals[:q,idx]
        qi, hr1 = prop(chain[:q,idx])

        # prior
        p_ = logprior(chain, :q=>(idx, qi), :bla=>9)

        # likelihood
        d = deepcopy(chain.model)
        d[:q, i] = qi
        l_ = logpdf!(d, chain.X, childnodes(tree,i)[1])

        mhr = p_ + l_ - chain[:logπ] - chain[:logp]
        if log(rand()) < mhr
            set_L!(chain.X)    # update L matrix
            chain.model = d
            chain[:logp] = l_
            chain[:logπ] = p_
            chain[:q, idx] = qi
            prop.accepted += 1
        else
            set_Ltmp!(chain.X)  # revert Ltmp matrix
        end
        consider_adaptation!(prop, chain.gen)
    end
end

function move_wgds!(chain::DLChain)
    tree = chain.Ψ
    for i in wgdnodes(tree)
        idx = tree[i,:q]
        jdx = tree[i,:θ]
        propq = chain.proposals[:q,idx]
        propr = chain.proposals[:λ,jdx]
        qi, hr1 = propq(chain[:q,idx])
        λi, hr2 = propr(chain[:λ,jdx])
        μi, hr3 = propr(chain[:μ,jdx])

        # prior
        p_ = logprior(chain, :q=>(idx, qi), :λ=>(jdx, λi), :μ=>(jdx, μi))

        # likelihood
        d = deepcopy(chain.model)
        d[:q,i] = qi
        d[:λ,i] = λi
        d[:μ,i] = μi
        l_ = logpdf!(d, chain.X, childnodes(tree,i)[1])

        mhr = p_ + l_ - chain[:logπ] - chain[:logp] + hr2 + hr3
        if log(rand()) < mhr
            set_L!(chain.X)    # update L matrix
            chain.model = d
            chain[:logp] = l_
            chain[:logπ] = p_
            chain[:q, idx] = qi
            chain[:λ, jdx] = λi
            chain[:μ, jdx] = μi
        else
            set_Ltmp!(chain.X)  # revert Ltmp matrix
        end
    end
end

function log_mcmc(chain, io, show_trace, show_every)
    if chain.gen == 1
        s = chain.state
        x = vcat("gen", [typeof(v)<:AbstractArray ?
                ["$k$i" for i in 1:length(v)] : k for (k,v) in s]...)
        chain.trace = DataFrame(zeros(0,length(x)), [Symbol(k) for k in x])
        show_trace ? write(io, join(x, ","), "\n") : nothing
    end
    x = vcat(chain.gen, [x for x in values(chain.state)]...)
    push!(chain.trace, x)
    if show_trace && chain.gen % show_every == 0
        write(io, join(x, ","), "\n")
    end
    flush(stdout)
end
