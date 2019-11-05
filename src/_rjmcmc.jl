# NOTE:
# * update! will be faster than using a deepcopy of the model
# * initial implementation, hardcoded for only two correlated characters

abstract type Model end

const Proposals_ = Dict{Int64,Vector{ProposalKernel}}

@with_kw mutable struct RevJumpChain{T<:Real,M<:Model}
    data ::PArray{T}
    model::DLWGD{T}
    prior::M          = RevJumpPrior()
    props::Proposals_ = Proposals_()
    state::State      = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace::DataFrame  = DataFrame()
end

function init!(chain::RevJumpChain)
    @unpack data, model, prior, state, props = chain
    setstate!(state, model, prior, data)
    trace!(chain)
    set!(data)
    setprops!(props, model)
end

function setstate!(state, model, prior, data)
    for (i, n) in model.nodes
        for (k, v) in n.x.θ
            k != :t ? state[id(n, k)] = v : nothing
        end
    end
    state[:logp] = logpdf!(model, data)
    state[:logπ] = logpdf(prior, model)
end

function setprops!(props, model)
    for (i, n) in model.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            props[i] = [AdaptiveUnitProposal()]
        elseif isroot(n)
            props[0] = [AdaptiveUnitProposal()]
            props[i] = CoevolUnProposals()
        else
            props[i] = CoevolUnProposals()
        end
    end
end

function trace!(chain)
    @unpack state, trace = chain
    chain.trace = vcat(trace, DataFrame(;sort(state)...), cols=:union)
end


# Prior
# =====
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}

Base.rand(x::Real) = x
logpdf(x::Real, y) = 0.

@with_kw struct RevJumpPrior <: Model
    Σ₀::Matrix{Float64}  = [500. 0. ; 0. 500.]
    X₀::Prior            = MvNormal([1.,1.])
    πη::Prior            = Beta(3., 0.33)
    πq::Prior            = Beta()
    πK::Prior            = Geometric(0.1)
end

# one-pass prior computation based on the model
function logpdf(prior::RevJumpPrior, d::DLWGD)
    @unpack Σ₀, X₀, πη, πq, πK = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(N, M)
    A = zeros(M,M)
    for (i, n) in d.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            p += logpdf(πq, n[:q])  # what about the time? it is also random?
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, log.(n[:λ, :μ]))
        else
            pt = nonwgdparent(n.p)
            Δt = parentdist(n, pt)
            Y[i-1,:] = (log.(n[:λ, :μ]) - log.(pt[:λ, :μ])) / √Δt
            A += Y[i-1,:]*Y[i-1,:]'
            J *= Δt
        end
    end
    p += logp_pics(Σ₀, (Y=Y, J=J^(-M/2), A=A, q=M+1, n=N))
    p += logpdf(πK, k)
    p::Float64  # type stability not entirely optimal
end

# p(Y|Σ₀,q), q = df, n = # of branches
function logp_pics(Σ₀, θ)
    @unpack J, A, q, n = θ
    # in our case the Jacobian is a constant (tree and times are fixed)
    log(J) + (q/2)*log(det(Σ₀)) - ((q + n)/2)*log(det(Σ₀ + A))
end

id(node::ModelNode, args::Symbol...) = [id(node, s) for s in args]
id(node::ModelNode, s::Symbol) = Symbol("$s$(node.i)")

function update!(state::State, node::ModelNode, args...)
    for s in args
        state[id(node, s)] = node[s]
    end
end

# Moves
# =====
function move!(chain)
    @unpack model = chain
    for (i, n) in model.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            move_wgd!(chain, n)
        else
            if isroot(n)
                 move_root!(chain, n)
            end
            move_node!(chain, n)
        end
    end
end

function move_node!(chain, n)
    @unpack data, state, model, props, prior = chain
    v = n[:λ, :μ]
    prop = rand(props[n.i])
    w, r = prop(log.(v))
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
end

function move_root!(chain, n)
    @unpack data, state, model, props, prior = chain
    prop = props[0][1]
    η = n[:η]
    η_, r = prop(η)
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
end

function move_wgd!(chain, n)
    @unpack data, state, model, props, prior = chain
    prop = props[n.i][1]
    child = first(first(n))
    q = n[:q]
    t1 = n[:t]
    t = t2 = child[:t]
    r = rand()
    if r < 0.66  # move q
        q_, r = prop(q)
        n[:q] = q_  # update from child below
    elseif r > 0.33  # move t
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
        prop.accepted += 1
    else
        n[:q] = q
        n[:t] = t1
        update!(child, :t, t2)
        rev!(data)
    end
end

function move_addwgd!(chain)
    # XXX unpack copies the model or something??
    @unpack data, state, model, props, prior = chain
    n, t = randpos(chain.model)
    wgdnode = insertwgd!(chain.model, n, t, rand(prior.πq)/20.)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    hr = state[:k] == 0 ? log(0.5) : 0.  # see Rabosky
    hr += l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] += 1
        update!(state, wgdnode, :q)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal()]
    else
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
end

function move_rmwgd!(chain)
    @unpack data, state, model, props, prior = chain
    if nwgd(chain.model) == 0
        return
    end
    # _model = deepcopy(chain.model)  # TODO find solution
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    child = removewgd!(chain.model, wgdnode, false)
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = state[:k] == 1 ? log(2.) : 0.  # see Rabosky
    hr += l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        Beluga.reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] -= 1
    else
        # re-inserting instead of the deepcopy above is tricky, since we insert
        # a node at the end, which may be a different index; if we refactor
        # removewgd to not reindex, we could re-insert the original wgdnode
        # upon rejection, re-insert the original node
        insertwgd!(chain.model, child, wgdnode, wgdafter)
        # chain.model = _model
        rev!(data)
    end
end

function randpos(model)
    l = length(model)
    v = zeros(l)
    for (i, n) in model.nodes
        v[i] = n[:t]
    end
    i = sample(1:l, Weights(v))
    t = rand(Uniform(0., model[i][:t]))
    model[i], t
end

nwgd(model) = length(getwgds(model))
randwgd(model) = model[rand(getwgds(model))]

function getwgds(model)
    wgds = Int64[]; i = length(model)
    while isawgd(model[i])
        iswgd(model[i]) ? push!(wgds, i) : nothing
        i -= 1
    end
    wgds
end

function reindex!(d::Dict{Int64,T}, i::Int64) where T
    for j=i:2:maximum(keys(d))
        d[j-2] = deepcopy(d[j])
        delete!(d, j)
    end
end
