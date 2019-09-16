# Adaptive MWG-MCMC for DL(+WGD) model
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
    λ = rand(GBM(tree, λ0, ν))
    μ = rand(GBM(tree, λ0, ν))
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

logprior(c::DLChain, θ) = logprior(c.priors, θ)

function mcmc!(chain::DLChain, n::Int64, args...;
        show_every=100, show_trace=true)
    for i=1:n
        chain.gen += 1
        move_ν!(chain)
        move_η!(chain)
        move_rates!(chain)
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
    # TODO q update too!
    for i in postorder(chain.Ψ)
        prop = chain.proposals[:λ, i]
        λi, hr1 = prop(chain[:λ,i])
        μi, hr2 = prop(chain[:μ,i])

        # likelihood
        d = deepcopy(chain.model)
        d[:λ, i] = λi
        d[:μ, i] = μi
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
            chain[:λ, i] = λi
            chain[:μ, i] = μi
            chain[:logp] = l_
            chain[:logπ] = p_
            prop.accepted += 1
        else
            set_Ltmp!(chain.X)  # revert Ltmp matrix
        end
        consider_adaptation!(prop, chain.gen)
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
