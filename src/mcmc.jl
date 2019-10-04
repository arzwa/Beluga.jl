# Adaptive MWG-MCMC for DL(+WGD) model
# TODO this is basically the same as in Whale; so a common abstraction layer
# would be nice (idem for the SpeciesTree & SlicedTree)

abstract type Chain end
const State = Dict{Symbol,Union{Vector{Float64},Float64,Int64}}
Base.getindex(state::State, s::Symbol, i::Int64) = state[s][i]
Base.setindex!(state::State, x, s::Symbol, i::Int64) = state[s][i] = x


# DLChain
# =======
# NOTE: Might be better to have a parametric DLChain{RatesPrior} type
"""
    DLChain

Chain object; storing both the data, state, proposals, trace and priors.
"""
mutable struct DLChain <: Chain
    X::PArray
    tree::SpeciesTree
    trace::DataFrame
    model::DuplicationLossWGD
    state::State
    priors::RatesPrior
    proposals::Proposals
end

Base.getindex(w::Chain, s::Symbol) = w.state[s]
Base.getindex(w::Chain, s::Symbol, i::Int64) = w.state[s][i]
Base.setindex!(w::Chain, x, s::Symbol) = w.state[s] = x
Base.setindex!(w::Chain, x, s::Symbol, i::Int64) = w.state[s][i] = x
Base.display(io::IO, w::Chain) = print("$(typeof(w))($(w.state))")
Base.show(io::IO, w::Chain) = write(io, "$(typeof(w))($(w.state))")

"""
    DLChain(X::PArray, prior::RatesPrior, tree::SpeciesTree, m::Int64)

Initilialize a DLChain with default proposals and sample from the prior as
initial state.
"""
function DLChain(X::PArray, prior::RatesPrior, tree::SpeciesTree, m::Int64)
    init = rand(prior, tree)
    init[:gen] = 0
    proposals = get_defaultproposals(init)
    model = DuplicationLossWGD(tree, init[:λ], init[:μ], init[:q], init[:η], m)
    return DLChain(X, tree, DataFrame(), model, init, prior, proposals)
end

# get an MCMCChains chain (gives diagnostics etc.)
function MCMCChains.Chains(df::DataFrame, burnin=1000)
    if size(df)[1] < burnin
        @error "Trace not long enough to discard $burnin iterations as burn-in"
    end
    X = reshape(Matrix(df), (size(df)...,1))[burnin+1:end, 2:end, :]
    return Chains(X, [string(x) for x in names(df)][2:end])
end

# get an MCMCChains chain (gives diagnostics etc.)
MCMCChains.Chains(c::DLChain, burnin=1000) = Chains(c.trace, burnin)

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
    proposals[:ψ] = AdaptiveScaleProposal(0.1)  # all-rates proposal
    return proposals
end

"""
    logprior(chain::DLChain, args...)

Evaluate prior for the current state and arbitrary modified parameters
(args...). For example: `logprior(chain, :ν=>0.2, :λ=>rand(10))`.
"""
function logprior(priors, state, tree, args...)
    s = deepcopy(state)
    for (k,v) in args
        if haskey(s, k)
            typeof(v)<:Tuple ? s[k][v[1]] = v[2] : s[k] = v
        else
            @warn "Trying to set unexisting variable ($k)"
        end
    end
    logprior(priors, s, tree)
end

logprior(chain::DLChain, args...) =
    logprior(chain.priors, chain.state, chain.tree, args...)
logprior(c::DLChain, θ::NamedTuple) = logprior(c.priors, θ)


# MCMC
# ====
"""
    mcmc!(chain, niters, fixed_params...; show_trace=true, show_every=10)

Do `niters` generations of the MCMC algorithm associated with chain. Dispatches
to specific MCMC implementations based on the prior type.
"""
function mcmc!(chain, niters, args...;
        show_trace=true, show_every=10, burnin=1000)
    mcmc!(chain, chain.priors, niters, show_trace, show_every, args...)
    Chains(chain, burnin)
end

function mcmc!(chain::DLChain,
        priors::ConstantRatesPrior,
        n, show_trace, show_every, args...)
    wgds = Beluga.nwgd(chain.tree) > 0
    init_mcmc!(chain)
    for i=1:n
        :η in args ? nothing : move_η!(chain)
        move_constantrates!(chain)
        if wgds
            move_q!(chain)
        end
        log_mcmc!(chain, stdout, show_trace, show_every)
    end
end

function mcmc!(chain::DLChain,
        priors::Union{GBMRatesPrior,IIDRatesPrior},
        n, show_trace, show_every, args...)
    wgds = Beluga.nwgd(chain.tree) > 0
    init_mcmc!(chain)
    for i=1:n
        :ν in args ? nothing : move_ν!(chain)  # could be more elegant
        :η in args ? nothing : move_η!(chain)
        move_rates!(chain)
        if wgds
            move_q!(chain)
            move_wgds!(chain)
        end
        move_allrates!(chain)
        log_mcmc!(chain, stdout, show_trace, show_every)
    end
end

function mcmc!(chain::DLChain,
        priors::NhRatesPrior,
        n, show_trace, show_every, args...)
    wgds = Beluga.nwgd(chain.tree) > 0
    init_mcmc!(chain)
    for i=1:n
        :η in args ? nothing : move_η!(chain)
        move_rates!(chain)
        if wgds
            move_q!(chain)
            move_wgds!(chain)
        end
        move_allrates!(chain)
        log_mcmc!(chain, stdout, show_trace, show_every)
    end
end

function init_mcmc!(chain)
    l = logpdf!(chain.model, chain.X)
    p = logprior(chain)
    set_L!(chain.X)
    chain[:logp] = l
    chain[:logπ] = p
end

function log_mcmc!(chain, io, show_trace, show_every)
    gen = chain[:gen] += 1
    if gen == 1
        init_trace!(chain, io, show_trace)
    end
    x = vcat([x for x in values(chain.state)]...)
    push!(chain.trace, x)
    if show_trace && gen % show_every == 0
        write(io, join(x, ","), "\n")
    end
    flush(stdout)
end

function init_trace!(chain, io, show_trace)
    x = vcat([typeof(v)<:AbstractArray ?
            ["$k$i" for i in 1:length(v)] : k for (k,v) in chain.state]...)
    chain.trace = DataFrame(zeros(0,length(x)), [Symbol(k) for k in x])
    show_trace ? write(io, join(x, ","), "\n") : nothing
end


# Moves
# =====
# NB do not use unpack statements in moves (move_rates)- I don't understand why
function move_ν!(chain::DLChain)
    prop = chain.proposals[:ν]
    ν_, hr = prop(chain[:ν])
    p_ = logprior(chain, :ν=>ν_)
    mhr = p_ - chain[:logπ] + hr
    if log(rand()) < mhr
        chain[:logπ] = p_
        chain[:ν] = ν_
        prop.accepted += 1
    end
    consider_adaptation!(prop, chain[:gen])
end

function move_η!(chain::DLChain)
    prop = chain.proposals[:η]
    η_, hr = prop(chain[:η])
    p_ = logprior(chain, :η=>η_)  # prior
    d = deepcopy(chain.model)
    d.η = η_
    l_ = logpdf!(d, chain.X, 1)  # likelihood; XXX assumes root is node 1
    mhr = p_ + l_ - chain[:logπ] - chain[:logp] + hr
    if log(rand()) < mhr
        chain.model = d
        chain[:logp] = l_
        chain[:logπ] = p_
        chain[:η] = η_
        prop.accepted += 1
        # NB: changing η does not change L matrices
    end
    consider_adaptation!(prop, chain[:gen])
end

function move_constantrates!(chain::DLChain)
    prop = chain.proposals[:λ, 1]
    λ, hr1 = prop(chain[:λ, 1])
    μ, hr2 = prop(chain[:μ, 1])
    p_ = logprior(chain, :λ=>λ, :μ=>μ)  # prior
    d = deepcopy(chain.model)
    d[:λ, 1] = λ
    d[:μ, 1] = μ
    l_ = logpdf!(d, chain.X)
    mhr = l_ + p_ - chain[:logp] - chain[:logπ] + hr1 + hr2
    if log(rand()) < mhr
        set_L!(chain.X)    # update L matrix
        chain.model = d
        chain[:λ, 1] = λ
        chain[:μ, 1] = μ
        chain[:logp] = l_
        chain[:logπ] = p_
        prop.accepted += 1
    else
        set_Ltmp!(chain.X)  # revert Ltmp matrix
    end
    consider_adaptation!(prop, chain[:gen])
end

function move_rates!(chain::DLChain)
    seen = Set{Int64}()  # HACK store indices that were already done
    for i in chain.tree.order
        idx = chain.tree[i,:θ]
        idx in seen ? continue : push!(seen, idx)
        prop = chain.proposals[:λ,idx]
        λi, hr1 = prop(chain[:λ,idx])
        μi, hr2 = prop(chain[:μ,idx])
        p_ = logprior(chain, :λ=>(idx, λi), :μ=>(idx, μi))  # prior
        d = deepcopy(chain.model)
        d[:μ, i] = μi   # NOTE: implementation of setindex! uses node indices!
        d[:λ, i] = λi   # NOTE: implementation of setindex! uses node indices!
        l_ = logpdf!(d, chain.X, i)  # likelihood
        mhr = l_ + p_ - chain[:logp] - chain[:logπ] + hr1 + hr2
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
        consider_adaptation!(prop, chain[:gen])
    end
end

function move_allrates!(chain::DLChain)
    prop = chain.proposals[:ψ]
    λ_, hr1 = prop(chain[:λ])
    μ_, hr2 = prop(chain[:μ])
    @unpack tree, q, η, value = chain.model
    # d = deepcopy(chain.model)
    # d.λ = λ_ that was so stupid
    # d.μ = μ_
    d = DuplicationLossWGD(tree, λ_, μ_, q, η, value.m)
    p = logprior(chain, :λ=>λ_, :μ=>μ_)
    l = logpdf!(d, chain.X)
    a = p + l - chain[:logπ] - chain[:logp] + hr1 + hr2
    if log(rand()) < a
        set_L!(chain.X)    # update L matrix
        chain.model = d
        chain[:λ] = λ_
        chain[:μ] = μ_
        chain[:logp] = l
        chain[:logπ] = p
        prop.accepted += 1
    else
        set_Ltmp!(chain.X)  # revert Ltmp matrix
    end
    consider_adaptation!(prop, chain[:gen])
end

function move_q!(chain::DLChain)
    tree = chain.tree
    for i in wgdnodes(tree)
        idx = tree[i,:q]
        prop = chain.proposals[:q,idx]
        qi, hr1 = prop(chain[:q,idx])
        p_ = logprior(chain, :q=>(idx, qi))  # prior
        d = deepcopy(chain.model)
        d[:q, i] = qi
        l_ = logpdf!(d, chain.X, childnodes(tree,i)[1])  # likelihood
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
        consider_adaptation!(prop, chain[:gen])
    end
end

function move_wgds!(chain::DLChain)
    tree = chain.tree
    for i in wgdnodes(tree)
        idx = tree[i,:q]
        jdx = tree[i,:θ]
        propq = chain.proposals[:q,idx]
        propr = chain.proposals[:λ,jdx]
        qi, hr1 = propq(chain[:q,idx])
        λi, hr2 = propr(chain[:λ,jdx])
        μi, hr3 = propr(chain[:μ,jdx])
        p_ = logprior(chain, :q=>(idx, qi), :λ=>(jdx, λi), :μ=>(jdx, μi))# prior
        d = deepcopy(chain.model)
        d[:q,i] = qi
        d[:λ,i] = λi
        d[:μ,i] = μi
        l_ = logpdf!(d, chain.X, childnodes(tree,i)[1])  # likelihood
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
