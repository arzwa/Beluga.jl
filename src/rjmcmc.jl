# NOTE:
# * update! will be faster than using a deepcopy of the model
# * initial implementation, hardcoded for only two correlated characters

abstract type Model end
abstract type RevJumpPrior <: Model end

const Proposals_ = Dict{Int64,Vector{ProposalKernel}}

@with_kw mutable struct RevJumpChain{T<:Real,M<:RevJumpPrior}
    data ::PArray{T}
    model::DLWGD{T}
    prior::M            = CoevolRevJumpPrior()
    props::Proposals_   = Proposals_()
    state::State        = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace::DataFrame    = DataFrame()
end

Base.vec(chain::RevJumpChain) = Vector(chain.trace[end,:])

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
    for (i, n) in model.nodes
        K += iswgd(n) ? 1 : 0
        for (k, v) in n.x.θ
            k != :t ? state[id(n, k)] = v : nothing
        end
    end
    state[:k] = K
end

function setprops!(props, model)
    for (i, n) in model.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
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


# Coevol-like Prior
# =================
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}

Base.rand(x::Real) = x
logpdf(x::Real, y) = 0.

@with_kw struct CoevolRevJumpPrior <: RevJumpPrior
    Σ₀::Matrix{Float64}  = [500. 0. ; 0. 500.]
    X₀::Prior            = MvNormal([1.,1.])
    πη::Prior            = Beta(3., 0.33)
    πq::Prior            = Beta()
    πK::Prior            = Geometric(0.5)
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


# Independent rates pior
# ======================
# This works for both branch and node rates?
# I guess we can also exploit conjugacy in an independent rates prior for
# branch rates?
@with_kw struct IidRevJumpPrior <: RevJumpPrior
    Σ₀::Matrix{Float64} = [100 0. ; 0. 100]
    X₀::Prior = MvNormal([1.,1.])
    πη::Prior = Beta(3., 0.33)
    πq::Prior = Beta()
    πK::Prior = Geometric(0.5)
end

function logpdf(prior::IidRevJumpPrior, d::DLWGD)
    @unpack Σ₀, X₀, πη, πq, πK = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(N, M)
    A = zeros(M,M)
    X0 = log.(d[1][:λ, :μ])
    for (i, n) in d.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            p += logpdf(πq, n[:q])  # what about the time? it is also random?
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, X0)
        else
            Y[i-1,:] = log.(n[:λ, :μ]) - X0
            A += Y[i-1,:]*Y[i-1,:]'
        end
    end
    p += logp_pics(Σ₀, (Y=Y, J=1., A=A, q=M+1, n=N))
    p += logpdf(πK, k)
    p::Float64  # type stability not entirely optimal
end

# Moves
# =====
function rjmcmc!(chain, n; trace=1, show=10)
    for i=1:n
        chain.state[:gen] += 1
        rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        move!(chain)
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

ro(x, d=3) = round(x, digits=d)
ro(x::Missing) = NaN
logmcmc(io::IO, df, n=15) = write(io, "↘ ", join(ro.(Vector(df[1:n])), ", "), " ⋯\n")

function move!(chain)
    @unpack model, prior = chain
    for n in postwalk(model[1])
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            move_wgdtime!(chain, n)
            move_wgdrates!(chain, n)
        else
            if isroot(n) && !(typeof(prior.πη)<:Number)
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

# ove for both retention rate and time of WGD
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
        q_, r = prop(q)
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
end

function move_wgdrates!(chain::RevJumpChain{T,CoevolRevJumpPrior}, n) where T
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    parent = nonwgdparent(n)
    child = nonwgdchild(n)
    flank = rand([parent, child])
    rates = flank[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w, r = prop(v)
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

function move_wgdrates!(chain::RevJumpChain{T,IidRevJumpPrior}, n) where T
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    child = nonwgdchild(n)
    rates = child[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w, r = prop(v)
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

function move_addwgd!(chain)
    # XXX unpack copies the model or something??
    @unpack data, state, model, props, prior = chain
    n, t = randpos(chain.model)
    wgdnode = insertwgd!(chain.model, n, t, rand(prior.πq)/10.)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    # hr = state[:k] == 0 ? log(0.5) : 0.  # see Rabosky
    # if we use the move where a removal or addition is chosen with P=0.5, this
    # is not necessary as q(M0|M1) = q(M1|M0) = 0.5
    hr = l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
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
    # hr = state[:k] == 1 ? log(2.) : 0.  # see Rabosky
    # if we use the move where a removal or addition is chosen with P=0.5, this
    # is not necessary as q(M0|M1) = q(M1|M0) = 0.5
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
        s = Symbol("k$(nonwgdchild(child).i)")
        state[s] -= 1
        state[:logp] = l_
        state[:logπ] = p_
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


# This doesn't seem to be working
# function move_rmwgd!(chain)
#     @unpack data, state, model, props, prior = chain
#     if nwgd(chain.model) == 0
#         return
#     end
#     _model = deepcopy(chain.model)
#     wgdnode = randwgd(chain.model)
#     child = removewgd!(chain.model, wgdnode)
#     @show child.p
#     @show child
#     length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
#     l_ = logpdf!(child, data)
#     p_ = logpdf(prior, chain.model)
#     hr = l_ + p_ - state[:logp] - state[:logπ]
#     if log(rand()) < hr
#         println("removal")
#         # upon acceptance; shrink and reindex
#         delete!(props, wgdnode.i)
#         delete!(state, id(wgdnode, :q))
#         reindex!(chain.props, wgdnode.i+2)
#         set!(data)
#         state[:logp] = l_
#         state[:logπ] = p_
#         state[:k] -= 1
#     else
#         chain.model = _model
#         rev!(data)
#     end
# end


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
    wgds = Int64[]; i = maximum(keys(model.nodes))
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

#= for each branch trace the WGDs?
As we're mostly interested in questions like, "what is the marginal posterior
probability of 1 WGD on branch x?", It's perhaps best to trace WGDs as a dict-
like structure {branch: (q, t), ...}?
=#
function tracewgds(chain)
    @unpack model = chain
    d = Dict{Int64,Array{Tuple}}()
    for i in getwgds(model)
        n = model.nodes[i]
        c = nonwgdchild(n)
        t = (n[:q], parentdist(c, n))
        haskey(d, c.i) ? push!(d[c.i], t) : d[c.i] = [t]
    end
    d
end


# Custom Proposals
# ================
# Extension of AdaptiveMCMC lib, proposal moves for vectors [q, λ, μ]
WgdProposals(ϵ=[1.0, 1.0, 1.0, 1.0], ti=25) = [AdaptiveUvProposal(
    kernel=Uniform(-e, e), tuneinterval=ti, move=m)
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


# Bayes factors
# =============
# Prior probabilities for # of WGDs on each branch
function kbranch_prior(n, k, T, prior, kmax=100)
    # NOTE approximate: kmax is the truncation bound for the # of WGDs in total
    # k is the maximal number of WGDs on the branch of interest for which the
    # prior should be computed
    te = parentdist(n, nonwgdparent(n.p))
    p = Dict(j=>0. for j=0:k)
    for j=0:k
        q = (te/T)^j
        for i=0:kmax
            p[j] += binomial(i, j)*q*(1. - (te/T))^(i-j)*pdf(prior, i)
        end
    end
    p
end

function branch_bayesfactors(chain)
    @unpack trace, model, prior = chain
    branch_bayesfactors(trace, model, prior)
end

function branch_bayesfactors(trace::DataFrame, model::DLWGD, prior)
    T = treelength(model)
    for (i,n) in sort(model.nodes)
        if isawgd(n); break; end
        if !(Symbol("k$i") in names(trace)); continue; end
        l = join(string.(Beluga.clade(model, n)), ",")
        println("_"^55)
        println("Node $i: ($l) ")
        ps = freqmap(trace[Symbol("k$i")])
        πs = kbranch_prior(n, maximum(keys(ps)), T, prior.πK)
        bf = zeros(maximum(keys(ps)))
        for k in 1:length(bf)
            bf[k] = (ps[k]/ps[k-1])/(πs[k]/πs[k-1])
            bfk = round(log10(bf[k]), digits=3)
            pk = round.([ps[k], ps[k-1]], digits=3)
            πk = round.([πs[k], πs[k-1]], digits=3)
            print("→ $k vs. $(k-1) ")
            print("log₁₀($(pk[1])/$(pk[2]) ÷ $(πk[1])/$(πk[2])) = $bfk")
            bfk > 1/2 ? print(" *") : nothing
            bfk > 1   ? print("*") : nothing
            bfk > 2   ? print("*\n") : print("\n")
        end
    end
end
