const State = Dict{Symbol,Union{Float64,Int64}}

# proposals
abstract type Proposals end

Base.getindex(p::Proposals, i::Int64) = p.proposals[i]
Base.setindex!(p::Proposals, x::Vector{<:AdaptiveUvProposal}, i::Int64) = p.proposals[i] = x
Base.delete!(p::Proposals, i::Int64) = delete!(p.proposals, i)
reindex!(p::Proposals, i::Int64) = reindex!(p.proposals, i)

struct MWGProposals <: Proposals
    proposals::Dict{Int64,Vector{AdaptiveUvProposal}}
end

MWGProposals() =
    MWGProposals(Dict{Int64,Vector{AdaptiveUvProposal{T,V} where{T,V}}}())

struct AMMProposals <: Proposals
    rates    ::AdaptiveMixtureProposal
    shift    ::AdaptiveUvProposal
    proposals::Dict{Int64}
end

AMMProposals(d::Int64; σ=0.1, β=0.1) = AMMProposals(
    AdaptiveMixtureProposal(d=d, σ=σ, β=β),
    AdaptiveUnProposal(),
    Dict{Int64,Vector{AdaptiveUvProposal{T,V} where{T,V}}}())

# reversible jump kernels
abstract type RevJumpKernel end

"""
    SimpleKernel

Reversible jump kernel that only introduces a WGD while not chnging λ or μ.
"""
@with_kw mutable struct SimpleKernel <: RevJumpKernel
    qkernel ::Beta{Float64} = Beta(1,1)
    accepted::Int64 = 0
end

function forward(kernel::SimpleKernel)
    q = rand(kernel.qkernel)
    q, logpdf(kernel.qkernel, q)
end

reverse(kernel::SimpleKernel, q::Float64) = logpdf(kernel.qkernel, q)

"""
    DropKernel

Reversible jump kernel that introduces a WGD and decreases λ on the associated
branch.
"""
@with_kw mutable struct DropKernel <: RevJumpKernel
    qkernel ::Beta{Float64} = Beta(1,1)
    λkernel ::Exponential{Float64} = Exponential(0.1)
    accepted::Int64 = 0
end

function forward(kernel::DropKernel, λ::Float64, μ::Float64)
    q  = rand(kernel.qkernel)
    θ  = log(λ) - rand(kernel.λkernel)
    q, exp(θ), μ, logpdf(kernel.qkernel, q)
end

function reverse(kernel::DropKernel, λ::Float64, μ::Float64, q::Float64)
    θ = log(λ) + rand(kernel.λkernel)
    exp(θ), μ, logpdf(kernel.qkernel, q)
end

"""
    BranchKernel

Reversible jump kernel that introduces a WGD, decreases λ and increases μ
on the associated branch.
"""
@with_kw mutable struct BranchKernel <: RevJumpKernel
    qkernel ::Beta{Float64} = Beta(1,1)
    λkernel ::Exponential{Float64} = Exponential(0.1)
    μkernel ::Exponential{Float64} = Exponential(0.1)
    accepted::Int64 = 0
end

function forward(kernel::BranchKernel, λ::Float64, μ::Float64)
    q  = rand(kernel.qkernel)
    rand() < 0.5 ?
        λ = exp(log(λ) - rand(kernel.λkernel)) :
        μ = exp(log(μ) + rand(kernel.μkernel))
    q, λ, μ, logpdf(kernel.qkernel, q)
end

function reverse(kernel::BranchKernel, λ::Float64, μ::Float64, q::Float64)
    rand() < 0.5 ?
        λ = exp(log(λ) + rand(kernel.λkernel)) :
        μ = exp(log(μ) - rand(kernel.μkernel))
    λ, μ, logpdf(kernel.qkernel, q)
end

# reversible jump chain
"""
    RevJumpChain

Reversible jump chain struct for DLWGD model inference.

!!! note After construction, an explicit call to `init!` is required.
"""
@with_kw mutable struct RevJumpChain{T<:Real,V<:ModelNode{T},P<:Proposals,
        M<:RevJumpPrior,K<:RevJumpKernel}
    data  ::PArray{T}
    model ::DLWGD{T,V}
    prior ::M            = IRRevJumpPrior()
    props ::P            = MWGProposals()
    state ::State        = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace ::DataFrame    = DataFrame()
    kernel::K            = BranchKernel()
    da    ::Bool         = true
end

Base.rand(df::DataFrame) = df[rand(1:size(df)[1]),:]
cols(chain::RevJumpChain, args...) = cols(chain.trace, args...)
cols(df::DataFrame, s) = [n for n in names(df) if startswith(string(n), s)]

"""
    init!(chain::RevJumpChain)

Initialize the chain.
"""
function init!(chain::RevJumpChain; ninit=50)
    @unpack data, model, prior, state, props = chain
    find_initial!(chain, ninit)
    setstate!(state, chain.model)
    trace!(chain)
    set!(chain.data)
    setprops!(props, chain.model)
end

# draw some inits from the prior and keep the one with highest posterior
function find_initial!(chain::RevJumpChain, n)
    @unpack data, model, prior, state, props = chain
    best = -Inf
    bestmodel = nothing
    for i=1:n
        data_ = deepcopy(data)
        x = rand(prior, model)
        addwgds!(data_, x.wgds)
        l = logpdf!(x.model, data_)
        p = logpdf(prior, x.model)
        if l + p > best
            bestmodel = x
            best = l + p
            state[:logp] = l
            state[:logπ] = p
        end
    end
    model = bestmodel.model
    addwgds!(data, bestmodel.wgds)
    l = logpdf!(model, data)  # do not forget to set the right DP matrices!
    chain.model = model
    chain.data = data
end

function setstate!(state, model)
    K = 0
    for i=2:ne(model)+1
        state[Symbol("k$i")] = 0
    end
    for (i, n) in model.nodes
        if iswgm(n)
            K += 1
            b = Symbol("k$(nonwgdchild(n).i)")
            state[b] += 1
        end
        for (k, v) in n.x.θ
            k != :t ? state[id(n, k)] = v : nothing
        end
    end
    state[:k] = K
end

function setprops!(props::MWGProposals, model)
    for (i, n) in model.nodes
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            props.proposals[i] = [AdaptiveUnitProposal(); WgdProposals()]
        elseif isroot(n)
            props.proposals[0] = [AdaptiveUnitProposal()]
            props.proposals[i] = CoevolUnProposals()
        else
            props.proposals[i] = CoevolUnProposals()
        end
    end
end

function setprops!(props::AMMProposals, model)
    for (i, n) in model.nodes
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            props.proposals[i] = [AdaptiveUnitProposal(); WgdProposals()]
        elseif isroot(n)
            props.proposals[0] = [AdaptiveUnitProposal()]
        end
    end
end

function trace!(chain)
    @unpack state, trace, model = chain
    chain.trace = vcat(trace, DataFrame(;sort(state)...,
        :wgds=>tracewgds(chain)), cols=:union)
end

function update!(state::State, node::ModelNode, args...)
    for s in args
        state[id(node, s)] = node[s]
    end
end

function reindex!(d::Dict{Int64,T}, i::Int64) where T
    for j=i:2:maximum(keys(d))
        d[j-2] = deepcopy(d[j])
        delete!(d, j)
    end
end

function branchrates!(chain)
    @unpack model, trace = chain
    for (i,n) in model.nodes
        if isroot(n) || isawgd(n) ; continue ; end
        l = (trace[id(n, :λ)] .+ trace[id(nonwgdparent(n.p), :λ)]) / 2.
        m = (trace[id(n, :μ)] .+ trace[id(nonwgdparent(n.p), :μ)]) / 2.
        trace[Symbol("l$i")] = l
        trace[Symbol("m$i")] = m
    end
end

function tracewgds(chain)
    @unpack model = chain
    d = Dict{Int64,Array{Tuple}}()
    for i in getwgds(model)
        n = model.nodes[i]
        c = nonwgdchild(n)
        x = (parentdist(c, n), n[:q])   # (t, q) tuple
        haskey(d, c.i) ? push!(d[c.i], x) : d[c.i] = [x]
    end
    d
end

# MCMC
rjmcmc!(chain, n; kwargs...) = rjmcmc!(chain, chain.prior, n; kwargs...)
  mcmc!(chain, n; kwargs...) =   mcmc!(chain, chain.prior, n; kwargs...)

function rjmcmc!(chain, prior::IRRevJumpPrior, n::Int64;
        trace=1, show=10, rjstart=0)
    Tl = prior.Tl
    logheader(stdout)
    for i=1:n
        chain.state[:gen] += 1
        if i > rjstart
            rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        end
        move!(chain, chain.props)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, chain)
            flush(stdout)
        end
        # NOTE: just to be sure there are no rogue nodes in the tree
        @assert length(chain.model.nodes) == length(postwalk(chain.model[1]))
        L = treelength(chain.model)
        @assert isapprox(Tl, L) "Tree length $L != $Tl"
    end
end

function mcmc!(chain, prior::IRRevJumpPrior, n::Int64; trace=1, show=10)
    logheader(stdout)
    for i=1:n
        chain.state[:gen] += 1
        move!(chain, chain.props)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, chain)
            flush(stdout)
        end
    end
end

# delayed acceptance factorizing prior and likelihood
acceptreject(chain, f, q) =
    chain.da ? acceptreject_da(chain, f, q) : acceptreject_default(chain, f, q)

function acceptreject_da(chain, f, q)
    @unpack prior, model, data, state = chain
    π  = logpdf(prior, model)
    α1 = min(π - state[:logπ] + q, 0.)
    accept = log(rand()) < α1
    ℓ  = accept ? f() : -Inf
    α2 = min(ℓ - state[:logp], 0.)
    accept = log(rand()) < α2
    return accept, ℓ, π
end

function acceptreject_default(chain, f, q)
    @unpack prior, model, data, state = chain
    π  = logpdf(prior, model)
    ℓ  = f()
    α  = π + ℓ - state[:logπ] - state[:logp] + q
    accept = log(rand()) < α
    return accept, ℓ, π
end

function logmcmc(io::IO, chain)
    @unpack trace, kernel = chain
    x = last(trace)
    cols1 = Symbol.(["k", "k2", "k3"])
    cols2 = Symbol.(["λ1", "λ2", "λ3", "μ1", "μ2", "μ3"])
    gen   = x[:gen]
    pjump = kernel.accepted / gen
    write(io, "|", join(
        [@sprintf("%7d,%5.2f,%3d,%2d,%2d", gen, pjump, x[cols1]...);
        [@sprintf("%6.3f", x[i]) for i in cols2]], ","), " ⋯ ",
         @sprintf("%10.3f,%8.3f,%6.3f", x[:logp], x[:logπ], x[:η1]), "\n")
end

function logheader(io::IO)
    write(io, "|", @sprintf(
        "%7s,%5s,%3s,%2s,%2s,%6s,%6s,%6s,%6s,%6s,%6s … %10s,%8s,%6s\n",
        split("gen pjmp k k2 k3 λ1 λ2 λ3 μ1 μ2 μ3 logp logπ η")...))
end

# Metropolis-within-Gibbs sweep over branches
function move!(chain, props::MWGProposals)
    @unpack model, prior = chain
    for n in postwalk(model[1])
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            move_wgdtime!(chain, n)
            move_wgdrates!(chain, n)
        else
            if isroot(n)
                move_root!(chain, n)
                move_node!(chain, n)
            else
                move_node!(chain, n)
            end
        end
    end
    return
end

# Multivariate update of rates, sweep over WGDs
function move!(chain, props::AMMProposals; kwargs...)
    @unpack model, prior = chain
    move_allrates!(chain, props.shift)
    move_allrates!(chain, props.rates)
    move_root!(chain, model[1])
    for i in getwgds(model)
        n = model.nodes[i]
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            move_wgdtime!(chain, n)
            move_wgdrates!(chain, n)
        end
    end
    return
end

function move_allrates!(chain, prop)
    @unpack data, state, model, prior = chain
    v = getrates(model)
    w, r = prop(log.(vcat(v...)))
    setrates!(model, reshape(exp.(w), size(v)...))
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(model, data), r)
    if accept
        state[:logp] = ℓ
        state[:logπ] = π
        setstate!(state, chain.model)
        set!(data)
        prop.accepted += 1
    else
        setrates!(model, v)
        rev!(data)
    end
    return
end

function move_node!(chain, n, equal::Bool=false)
    @unpack data, state, model, props, prior = chain
    v = n[:λ, :μ]
    prop = rand(props[n.i])
    w::Vector{Float64}, r::Float64 = prop(log.(v))
    equal ? w[2] = w[1] : nothing
    update!(n, (λ=exp(w[1]), μ=exp(w[2])))
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(n, data), r)
    if accept
        update!(state, n, :λ, :μ)
        state[:logp] = ℓ
        state[:logπ] = π
        set!(data)
        prop.accepted += 1
    else
        update!(n, (λ=v[1], μ=v[2]))
        rev!(data)
    end
    return
end

function move_root!(chain, n)
    @unpack data, state, model, props, prior = chain
    if typeof(prior.πη)<:ConstantDistribution
        return
    end
    prop = props[0][1]
    η = n[:η]
    η_::Float64, r::Float64 = prop(η)
    update!(n, :η, η_)
    accept, ℓ, π = acceptreject(chain, ()->logpdfroot(n, data), r)
    if accept
        update!(state, n, :η)
        state[:logp] = ℓ
        state[:logπ] = π
        set!(data)
        prop.accepted += 1
    else
        update!(n, :η, η)
        rev!(data)
    end
    return
end

# move for both retention rate and time of WGD
function move_wgdtime!(chain, n)
    @unpack data, state, model, props, prior = chain
    prop = props[n.i][1]
    child = first(first(n))
    nextsp = nonwgdchild(n)
    q = n[:q]
    t1 = n[:t]
    t2 = child[:t]
    u = rand()
    r = 0.
    if u < 0.66  # move q
        q_::Float64, r::Float64 = prop(q)
        n[:q] = q_  # update from child below
    elseif u > 0.33  # move t
        t = rand()*(t1 + t2)
        n[:t] = t1 + t2 - t  # update from child below
        child[:t] = t
    end
    update!(nextsp)
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(nextsp, data), r)
    if accept
        update!(state, n, :q)
        state[:logp] = ℓ
        state[:logπ] = π
        set!(data)
        u < 0.66 ? prop.accepted += 1 : nothing
    else
        n[:q] = q
        n[:t] = t1
        child[:t] = t2
        update!(nextsp)
        rev!(data)
    end
    return
end

move_wgdrates!(chain, n) = move_wgdrates!(chain, chain.prior, n)

function move_wgdrates!(chain, prior::IRRevJumpPrior, n)
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    child = nonwgdchild(n)
    rates = child[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w::Vector{Float64}, r::Float64 = prop(v)
    n[:q] = w[1]
    update!(child, (λ=exp(w[2]), μ=exp(w[3])))
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(child, data), r)
    if accept
        update!(state, n, :q)
        update!(state, child, :λ, :μ)
        state[:logp] = ℓ
        state[:logπ] = π
        set!(data)
        prop.accepted += 1
    else
        n[:q] = q
        update!(child, (λ=rates[1], μ=rates[2]))
        rev!(data)
    end
end


# Reversible jump moves
# =====================
# dispatch on kernel
move_addwgd!(chain::RevJumpChain) = move_addwgd!(chain, chain.kernel)
move_rmwgd!( chain::RevJumpChain) = move_rmwgd!( chain, chain.kernel)

# simple move, independence samplers q and t
function move_addwgd!(chain, kernel::SimpleKernel)
    @unpack data, state, props, prior = chain
    @unpack Tl = prior
    if logpdf(prior.πK, nwgd(chain.model)+1) == -Inf; return; end
    n, t  = randpos(chain.model)
    child = nonwgdchild(n)
    q, lp = forward(kernel)
    wgdnode = addwgd!(chain.model, n, t, q)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(child, data), -lp+log(Tl))
    if accept
        state[:logp] = ℓ
        state[:logπ] = π
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
        kernel.accepted += 1
    else
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
    return
end

# deterministic down move for simple kernel
function move_rmwgd!(chain, kernel::SimpleKernel)
    @unpack data, state, props, prior = chain
    @unpack Tl = prior
    if nwgd(chain.model) == 0; return; end
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    n  = nonwgdchild(wgdnode)
    lp = reverse(kernel, wgdnode[:q])
    # println("-"^70)
    # @show sort(collect(keys(chain.model.nodes)))
    child = removewgd!(chain.model, wgdnode, false, true)
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(n, data), lp-log(Tl))
    # @show sort(collect(keys(chain.model.nodes)))
    # println("-"^70)
    if accept
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        state[:logp] = ℓ
        state[:logπ] = π
        kernel.accepted += 1
    else
        addwgd!(chain.model, child, wgdnode, wgdafter)
        rev!(data)
    end
end

# drop kernel, reverse move is non-deterministic
# from what I've observed so far, a WGD tends to correspond wit an increased λ
# and typicaly not a decreased μ, so the drop move makes sense
function move_addwgd!(chain, kernel::Union{DropKernel,BranchKernel})
    @unpack data, state, props, prior = chain
    @unpack Tl = prior
    if logpdf(prior.πK, nwgd(chain.model)+1) == -Inf; return; end
    n, t  = randpos(chain.model)
    child = nonwgdchild(n)
    λ, μ = child[:λ, :μ]
    q, λ_, μ_, lp = forward(kernel, λ, μ)
    child[:λ] = λ_
    child[:μ] = μ_
    wgdnode = addwgd!(chain.model, n, t, q)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(child, data), -lp+log(Tl))
    if accept
        state[:logp] = ℓ
        state[:logπ] = π
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        update!(state, child, :λ)
        update!(state, child, :μ)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
        kernel.accepted += 1
    else
        child[:λ] = λ
        child[:μ] = μ
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
    return
end

function move_rmwgd!(chain, kernel::Union{DropKernel,BranchKernel})
    @unpack data, state, props, prior = chain
    @unpack Tl = prior
    if nwgd(chain.model) == 0; return; end
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    n = nonwgdchild(wgdnode)
    λ, μ = n[:λ, :μ]
    λ_, μ_, lp = reverse(kernel, λ, μ, wgdnode[:q])
    # @show λ, λ_, μ, μ_
    n[:λ] = λ_
    n[:μ] = μ_
    child = removewgd!(chain.model, wgdnode, false, true)
    accept, ℓ, π = acceptreject(chain, ()->logpdf!(n, data), lp-log(Tl))
    if accept
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        state[:logp] = ℓ
        state[:logπ] = π
        kernel.accepted += 1
    else
        n[:λ] = λ
        n[:μ] = μ
        addwgd!(chain.model, child, wgdnode, wgdafter)
        rev!(data)
    end
end

function addrandwgds!(model::DLWGD, data::PArray, k, πq)
    for i=1:k
        n, t = Beluga.randpos(model)
        q = rand(πq)
        wgdnode = addwgd!(model, n, t, q)
        child = Beluga.nonwgdchild(wgdnode)
        extend!(data, n.i)
    end
end
