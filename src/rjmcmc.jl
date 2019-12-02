# NOTE:
# * update! will be faster than using a deepcopy of the model
# * initial implementation, hardcoded for only two correlated characters
# * the proposal mechnism is not yet type stable...

const Proposals = Dict{Int64,Vector{AdaptiveUvProposal{T,V} where {T,V}}}
const State = Dict{Symbol,Union{Float64,Int64}}

# @with_kw mutable struct Props{T,U,V,W,X,Y}
#     rates::Dict{Int64,T}
#     wgdst::Dict{Int64,U}
#     wgdsr::Dict{Int64,V}
#     root ::W
#     rjump::Array{X,Y}
# end

@with_kw mutable struct RevJumpChain{T<:Real,V<:ModelNode{T},M<:RevJumpPrior}
    data ::PArray{T}
    model::DLWGD{T,V}
    prior::M            = IidRevJumpPrior()
    props::Proposals    = Proposals()
    state::State        = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace::DataFrame    = DataFrame()
end

Base.rand(df::DataFrame) = df[rand(1:size(df)[1]),:]

function init!(chain::RevJumpChain; rjump=(1., 10., 0.01))
    @unpack data, model, prior, state, props = chain
    setstate!(state, model)
    state[:logp] = logpdf!(model, data)
    state[:logπ] = logpdf(prior, model)
    trace!(chain)
    set!(data)
    setprops!(props, model, rjump)
end

function setstate!(state, model)
    K = 0
    for (i, n) in model.nodes
        if iswgd(n)
            K += 1
            b = Symbol("k$(nonwgdchild(n).i)")
            haskey(state, b) ? state[b] += 1 : state[b] = 1
        end
        for (k, v) in n.x.θ
            k != :t ? state[id(n, k)] = v : nothing
        end
    end
    state[:k] = K
end

function setprops!(props, model, rjump)
    for (i, n) in model.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            props[i] = [AdaptiveUnitProposal(); WgdProposals()]
        elseif isroot(n)
            props[0] = [AdaptiveUnitProposal(),
                AdaptiveUvProposal(
                    kernel=Beta(rjump[1], rjump[2]),
                    bounds=(0.,1.), tuneinterval=10^10, stop=0,
                    move=AdaptiveMCMC.independent),
                DecreaseProposal(rjump[3], 10^10)]
            # they should be correlated, a large q with a strong decrease in λ
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

# see Lartillot & Poujol 2010
function posterior_Σ!(chain)
    @unpack model, prior = chain
    @unpack Σ₀ = prior
    chain.trace[!,:var] .= NaN
    chain.trace[!,:cov] .= NaN
    for row in eachrow(chain.trace)
        m = model(row)
        @unpack A, q, n = scattermat(model, prior)
        Σ = rand(InverseWishart(q + n, Σ₀ + A))
        row[:var] = Σ[1,1]
        row[:cov] = Σ[1,2]
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


# Custom Proposals
# ================
# Extension of AdaptiveMCMC lib, proposal moves for vectors [q, λ, μ]
WgdProposals(ϵ=[1.0, 1.0, 1.0, 1.0], ti=25) = AdaptiveUvProposal[
    AdaptiveUvProposal(kernel=Uniform(-e, e), tuneinterval=ti, move=m)
        for (m, e) in zip([wgdrw, wgdrand, wgdiid, wgdqλ], ϵ)]

function wgdrw(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = x .+ rand(k)
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end

function wgdrand(k::AdaptiveUvProposal, x::Vector{Float64})
    i = rand(1:3)
    xp = copy(x)
    xp[i] = x[i] + rand(k)
    i == 1 ? xp[1] = reflect(xp[1], 0., 1.) : nothing
    return xp, 0.
end

function wgdiid(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = x .+ rand(k, 3)
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end

function wgdqλ(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = copy(x)
    r = rand(k)
    xp[1] += r
    xp[2] -= r
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end


# MCMC
# =====
function rjmcmc!(chain, n; trace=1, show=10, rjstart=0, rootequal=false)
    for i=1:n
        chain.state[:gen] += 1
        if i > rjstart
            rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        end
        move!(chain, rootequal=rootequal)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, last(chain.trace))
            flush(stdout)
        end
    end
end

function mcmc!(chain, n; trace=1, show=10)
    for i=1:n
        chain.state[:gen] += 1
        move!(chain)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, last(chain.trace))
            flush(stdout)
        end
    end
end

logmcmc(io::IO, df, n=15) = write(io, "", join([@sprintf("%d,%d",df[1:2]...);
    [@sprintf("%6.3f", x) for x in Vector(df[3:n])]], ","), " ⋯\n| ")

function move!(chain; rootequal::Bool=false)
    @unpack model, prior = chain
    for n in postwalk(model[1])
        if iswgdafter(n)
            continue
        elseif iswgd(n)
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
    q = n[:q]
    t1 = n[:t]
    t = t2 = child[:t]
    u = rand()
    r = 0.
    if u < 0.66  # move q
        q_::Float64, r::Float64 = prop(q)
        n[:q] = q_  # update from child below
    elseif u > 0.33  # move t
        t = rand()*(t1 + t2)
        n[:t] = t1 + t2 - t  # update from child below
    end
    update!(child, :t, t)
    l_ = logpdf!(n, data)
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
        update!(child, :t, t2)
        rev!(data)
    end
    return
end

move_wgdrates!(chain, n) = move_wgdrates!(chain, chain.prior, n)

function move_wgdrates!(chain, prior::CoevolRevJumpPrior, n)
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    parent = nonwgdparent(n)
    child = nonwgdchild(n)
    flank = rand([parent, child])
    rates = flank[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w::Vector{Float64}, r::Float64 = prop(v)
    n[:q] = w[1]
    flank[:λ] = exp(w[2])
    flank[:μ] = exp(w[3])
    update!(child)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        update!(state, flank, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        prop.accepted += 1
    else
        n[:q] = q
        flank[:λ] = rates[1]
        flank[:μ] = rates[2]
        update!(child)
        rev!(data)
    end
end

function move_wgdrates!(chain, prior::IidRevJumpPrior, n)
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    child = nonwgdchild(n)
    rates = child[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w::Vector{Float64}, r::Float64 = prop(v)
    n[:q] = w[1]
    update!(child, (λ=exp(w[2]), μ=exp(w[3])))
    l_ = logpdf!(n, data)
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
function move_addwgd!(chain)
    @unpack data, state, model, props, prior = chain
    if logpdf(prior.πK, nwgd(chain.model)+1) == -Inf
        return
    end
    n, t  = randpos(chain.model)
    child = nonwgdchild(n)
    propq = chain.props[0][2]
    propλ = chain.props[0][3]
    λn    = child[:λ]
    q::Float64, r1::Float64  = propq(0.)
    θ::Float64, r2::Float64  = propλ(log(λn))

    child[:λ] = exp(θ)
    wgdnode = insertwgd!(chain.model, n, t, q)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ]

    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        update!(state, child, :λ)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
        propλ.accepted += 1
        propq.accepted += 1
    else
        child[:λ] = λn
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
    return
end


function move_rmwgd!(chain)
    @unpack data, state, model, props, prior = chain
    if nwgd(chain.model) == 0
        return
    end
    propλ = chain.props[0][3]
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    n  = nonwgdchild(wgdnode)
    λn = n[:λ]
    θ::Float64, r1::Float64  = propλ(log(λn))
    θ  = log(λn) - (θ - log(λn))  # HACK to reverse decrease proposal
    n[:λ] = exp(θ)
    child = removewgd!(chain.model, wgdnode, false)

    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        s = Symbol("k$(n.i)")
        update!(state, n, :λ)
        state[s] -= 1
        state[:logp] = l_
        state[:logπ] = p_
        propλ.accepted += 1
    else
        n[:λ] = λn
        insertwgd!(chain.model, child, wgdnode, wgdafter)
        rev!(data)
    end
    return
end

# function _move_addwgd!(chain::RevJumpChain{T,}) where T
#     # XXX unpack copies the model or something??
#     @unpack data, state, model, props, prior = chain
#     n, t = randpos(chain.model)
#     wgdnode = insertwgd!(chain.model, n, t, rand(prior.πq)/10.)
#     length(data[1].x) == 0 ? nothing : extend!(data, n.i)
#     l_ = logpdf!(n, data)
#     p_ = logpdf(prior, chain.model)
#     # hr = state[:k] == 0 ? log(0.5) : 0.  # see Rabosky
#     # if we use the move where a removal or addition is chosen with P=0.5, this
#     # is not necessary as q(M0|M1) = q(M1|M0) = 0.5
#     hr = l_ + p_ - state[:logp] - state[:logπ]
#     if log(rand()) < hr
#         state[:logp] = l_
#         state[:logπ] = p_
#         state[:k] += 1
#         s = Symbol("k$(nonwgdchild(wgdnode).i)")
#         state[s] = haskey(state, s) ? state[s] + 1 : 1
#         update!(state, wgdnode, :q)
#         set!(data)
#         props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
#     else
#         removewgd!(chain.model, wgdnode)
#         rev!(data)
#     end
# end

# function _move_rmwgd!(chain::RevJumpChain{T,IidRevJumpPrior}) where T
#     @unpack data, state, model, props, prior = chain
#     if nwgd(chain.model) == 0
#         return
#     end
#     # _model = deepcopy(chain.model)  # TODO find solution
#     wgdnode = randwgd(chain.model)
#     wgdafter = first(wgdnode)
#     child = removewgd!(chain.model, wgdnode, false)
#     l_ = logpdf!(child, data)
#     p_ = logpdf(prior, chain.model)
#     hr = l_ + p_ - state[:logp] - state[:logπ]
#     if log(rand()) < hr
#         # upon acceptance; shrink and reindex
#         length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
#         delete!(props, wgdnode.i)
#         delete!(state, id(wgdnode, :q))
#         reindex!(chain.model, wgdnode.i+2)
#         reindex!(chain.props, wgdnode.i+2)
#         set!(data)
#         setstate!(state, chain.model)
#         s = Symbol("k$(nonwgdchild(child).i)")
#         state[s] -= 1
#         state[:logp] = l_
#         state[:logπ] = p_
#     else
#         # re-inserting instead of the deepcopy above is tricky, since we insert
#         # a node at the end, which may be a different index; if we refactor
#         # removewgd to not reindex, we could re-insert the original wgdnode
#         # upon rejection, re-insert the original node
#         insertwgd!(chain.model, child, wgdnode, wgdafter)
#         # chain.model = _model
#         rev!(data)
#     end
# end
