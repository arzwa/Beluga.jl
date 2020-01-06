const Proposals = Dict{Int64,Vector{AdaptiveUvProposal{T,V} where {T,V}}}
const State = Dict{Symbol,Union{Float64,Int64}}

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
@with_kw mutable struct RevJumpChain{T<:Real,V<:ModelNode{T},
        M<:RevJumpPrior,K<:RevJumpKernel}
    data  ::PArray{T}
    model ::DLWGD{T,V}
    prior ::M            = IRRevJumpPrior()
    props ::Proposals    = Proposals()
    state ::State        = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace ::DataFrame    = DataFrame()
    kernel::K            = BranchKernel()
end

Base.rand(df::DataFrame) = df[rand(1:size(df)[1]),:]
cols(chain::RevJumpChain, args...) = cols(chain.trace, args...)
cols(df::DataFrame, s) = [n for n in names(df) if startswith(string(n), s)]

"""
    init!(chain::RevJumpChain)

Initialize the chain.
"""
function init!(chain::RevJumpChain)
    @unpack data, model, prior, state, props = chain
    setstate!(state, model)
    state[:logp] = logpdf!(model, data)
    state[:logπ] = logpdf(prior, model)
    trace!(chain)
    set!(data)
    setprops!(props, model)
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

function setprops!(props, model)
    for (i, n) in model.nodes
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            props[i] = [AdaptiveUnitProposal(); WgdProposals()]
        elseif isroot(n)
            props[0] = [AdaptiveUnitProposal()]
            props[i] = CoevolUnProposals()
        else
            props[i] = CoevolUnProposals()
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
        trace=1, show=10, rjstart=0, rootequal=false)
    Tl = prior.Tl
    logheader(stdout)
    for i=1:n
        chain.state[:gen] += 1
        if i > rjstart
            rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        end
        move!(chain, rootequal=rootequal)
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
        move!(chain)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, chain)
            flush(stdout)
        end
    end
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

function move!(chain; rootequal::Bool=false)
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
                move_node!(chain, n, rootequal)
            else
                move_node!(chain, n)
            end
        end
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
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    # @show l_, p_, exp.(w), v
    if log(rand()) < hr
        update!(state, n, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdfroot(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :η)
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdf!(nextsp, data)  # this was n instead of child, does that work
                               # here? apparently not!
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdf!(child, data)  # this was n instead of child, but that doesn't
                               # work right? 16/12/2019
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        update!(state, child, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ] - lp + log(Tl)
    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
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
    child = removewgd!(chain.model, wgdnode, false, true)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + lp - log(Tl)
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ] - lp + log(Tl)  # × (1/T)^-1
    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
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
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + lp - log(Tl)
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        state[:logp] = l_
        state[:logπ] = p_
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
