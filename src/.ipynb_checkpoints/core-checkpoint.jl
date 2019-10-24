# This is the core; with as goal efficient methods to compute likelihood for
# gene count profiles under the birth-death process
# NOTE: assuming only linear BDPs
# NOTE: assuming geometric at root could be generailzed to uniform,
# Poisson, Negbinomial and geometric
# NOTE: only conditioning on one in both clades currently
# TODO: implement gain (~ Csuros & Miklos) as well

# helper struct for the Csuros & Miklos algorithm
mutable struct CsurosMiklos{T<:Real}
    W::Array{T,3}
    ϵ::Array{T,2}
    m::Int64
end


"""
    DuplicationLossWGD(tree, λ, μ, q, η, mmax)

Stochastic duplication-loss (DL) and whole-genome duplication (WGD) model for
gene content evolution. This models DL as a Linear Birth-death process, and WGD
using the Rabier model, which assumes a binomial distribution on the number of
genes retained after WGD.

Note: This implements an indexer, for an instance `m` of DuplicationLossWGD,
`m[s, i]` gives the parameter value of parameter `s` (Symbol) for node/branch
`i` (Int), e.g. `m[:λ, 17]` will return the duplication rate for branch 17
(which is by convention the branch leading to node 17).
"""
mutable struct DuplicationLossWGD{T<:Real,Ψ<:Arboreal} <: PhyloBDP
    tree::Ψ
    λ::Vector{T}  # AbstractVector when we want to use Tracker?
    μ::Vector{T}  # AbstractVector when we want to use Tracker?
    q::Vector{T}  # AbstractVector when we want to use Tracker?
    η::T  # geometric prior probability at the root
    value::CsurosMiklos{T}
end

function DuplicationLossWGD(tree::Ψ,
        λ::Vector{T}, μ::Vector{T}, q::Vector{T},
        η::T, mmax::Int) where {Ψ<:Arboreal,T<:Real}
    ϵ = ones(T, length(tree.order), 2)
    # last dimension of W one too large, unnecessary memory...
    W = minfs(T, mmax+1, mmax+1, maximum(tree.order))
    c = CsurosMiklos(W, ϵ, mmax)
    d = DuplicationLossWGD{T,Ψ}(tree, λ, μ, q, η, c)
    get_ϵ!(d)
    get_W!(d)
    return d
end

"""
    DuplicationLoss(tree, λ, μ, η, mmax)

Stochastic duplication-loss (DL) model for gene content evolution. This models
DL as a Linear Birth-death process. This is a special case of the
`DuplicationLossWGD` model without WGDs or all `q` equal to 0.
"""
DuplicationLoss(tree::Arboreal, λ::Vector{T}, μ::Vector{T}, η::T, mmax::Int
    ) where T<:Real = DuplicationLossWGD(tree, λ, μ, eltype(λ)[], η, mmax)

Base.show(io::IO, d::DuplicationLossWGD) = show(io, d, (:λ, :μ, :q, :η))

# NB: takes a node index!
Base.getindex(d::DuplicationLossWGD, s::Symbol, i::Int64) =
    getfield(d, s)[d.tree[i, _translate(s)]]

# efficiently changing parameters by only recomputing required branches
function Base.setindex!(d::DuplicationLossWGD{T,Ψ}, v::T, s::Symbol,
        i::Int) where {Ψ<:Arboreal,T<:Real}
    # NB: this takes a node index, not rate index!
    idx = d.tree[i, _translate(s)]
    getfield(d, s)[idx] = v
    bs = d.tree.pbranches[i]
    get_ϵ!(d, bs)
    get_W!(d, bs)
end

# vector based constructor (to core.jl?)
function (d::DuplicationLossWGD)(θ::AbstractArray{T,1}, η=θ[end]) where T<:Real
    @unpack tree, value = d
    n = nrates(tree)
    q = θ[2n+1:end-1]
    λ = θ[1:n]
    μ = θ[n+1:2n]
    DuplicationLossWGD(tree, λ, μ, q, η, value.m)
end

asvector(d::DuplicationLossWGD) = [d.λ ; d.μ ; d.q ; d.η ]


# logpdf
# ======
# returns L matrix; not generally of use; but good for testing
function _logpdf(d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    L = minfs(T, length(d.tree), maximum(x)+1)
    l = logpdf!(L, d, x, d.tree.order)
    (l=l, L=L)
end

function logpdf(d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    L = minfs(T, length(d.tree), maximum(x)+1)
    logpdf!(L, d, x, d.tree.order)  # returns only likelihood
end

function logpdf!(L::Matrix{T},
        d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64},
        branches::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    @unpack value, tree, η = d
    root = branches[end]
    if branches != [root]
        L = csuros_miklos!(L, x, value, tree, branches)
    end
    l = integrate_root(L[root,:], η, value.ϵ[root,2])
    l -= condition_oib(tree, η, value.ϵ)
    isinf(l) ? -Inf : l  # FIXME; shouldn't be necessary
end

# Different interface (than profile) to compute the accumulated logpdf over
# multiple phylogenetic profiles, note that it is in fact slower than the
# Profile approach
function logpdf(d::DuplicationLossWGD{T,Ψ},
        X::AbstractMatrix{Int64}) where {Ψ<:Arboreal,T<:Real}
    m = maximum(X)
    L = minfs(T, size(X)..., m+1)
    l = @inbounds @distributed (+) for i=1:size(X)[1]
        logpdf!(L[i,:,:], d, X[i,:], d.tree.order)
    end
    l
end


# extinction probabilities
# ========================
get_ϵ!(d::DuplicationLossWGD) = get_ϵ!(d, d.tree.order)

function get_ϵ!(model::DuplicationLossWGD, branches::Vector{Int64})
    # Extinction probabilities, ϵ[node, 2] is the extinction P at the end of the
    # branch leading to `node`, ϵ[node, 1] is the extinction P at the beginning
    # of the branch leading to `node`.
    # NOTE: it is probably not really necessary to work on a log-scale here
    # as the p's don't get *really* small in general; but it doesn't lead
    # to less efficient code anyway
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    ϵ[1,1] = NaN  # root has only one extinction P, NaN for safety
    for e in branches
        ϵ[e,:] .= 0.
        if isleaf(tree, e)
            ϵ[e, 2] = log(zero(eltype(ϵ)))
        elseif Beluga.iswgd(tree, e)
            qe = q[tree[e, :q]]
            f = childnodes(tree, e)[1]
            μf = model[:μ, f]
            λf = model[:λ, f]
            t = parentdist(tree, f)
            ϵ[f, 1] = log(ep(λf, μf, t, exp(ϵ[f, 2])))  # XXX pass log ϵ?
            ϵ[f, 1] = ϵ[e, 2] = logaddexp(
                log(qe)+2*ϵ[f, 1], log(1. - qe)+ϵ[f, 1])  # HACK?
        else
            for c in childnodes(tree, e)
                μc = model[:μ, c]
                λc = model[:λ, c]
                t = parentdist(tree, c)
                ϵ[c, 1] = log(ep(λc, μc, t, exp(ϵ[c, 2]))) # XXX pass log ϵ?
                ϵ[e, 2] += ϵ[c, 1]
            end
        end
    end
    if any(ϵ .> 0.)
        @error "Some non-valid extinction probabilities\n, $λ, $μ"
        ϵ[ϵ .> 0.] .= 0.
    end
end


# transition probabiities
# =======================
# W matrix for Csuros & Miklos algorithm
get_W!(d::DuplicationLossWGD) = get_W!(d, d.tree.order)

function get_W!(model::DuplicationLossWGD, branches::Vector{Int64})
    # XXX↓ WGD affects branch *below* a WGD!
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    m == 0 ? (return) : nothing  # HACK when sampling from prior
    for i in branches[1:end-1]  # excluding root node
        μi = model[:μ, i]
        λi = model[:λ, i]
        ϵi = ϵ[i, 2]
        t = parentdist(tree, i)
        if Beluga.iswgdafter(tree, i)  # XXX↑
            qi = q[Beluga.qparent(tree, i)]
            W[:, :, i] = _wstar_wgd(t, λi, μi, qi, ϵi, m)
        else
            W[:, :, i] = _wstar(t, λi, μi, ϵi, m)
        end
    end
end

function _wstar(t, λ, μ, ϵ, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = (ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n
    ψp = ψ*(1. - ϵ) / _n
    w = zeros(typeof(λ), mmax+1, mmax+1)
    w[1,1] = 1.
#     ϕ = log(getϕ(t, λ, μ))  # p
#     ψ = log(getψ(t, λ, μ))  # q
#     _n = log1mexp(ψ+ϵ)
#     ϕp = logaddexp(ϕ+log1mexp(ϵ), log1mexp(ψ)+ϵ) - _n
#     ψp = ψ+log1mexp(ϵ) - _n
#     w = minfs(typeof(λ), mmax+1, mmax+1)
#     w[1,1] = 0.
    for m=1:mmax, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
#         w[n+1, m+1] = logaddexp(ψp + w[n+1, m],
#             log1mexp(ϕp) + log1mexp(ψp) + w[n, m])
    end
    return log.(w)
end

function _wstar_wgd(t, λ, μ, q, ϵ, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = (ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n
    ψp = ψ*(1. - ϵ) / _n
    w = zeros(typeof(λ), mmax+1, mmax+1)
    w[1,1] = 1.
    w[2,2] = ((1. - q) + 2q*ϵ)*(1. - ϵ)
    w[2,3] = q*(1. - ϵ)^2
#     ϕ = log(getϕ(t, λ, μ))  # p
#     ψ = log(getψ(t, λ, μ))  # q
#     _n = log1mexp(ψ+ϵ)
#     ϕp = logaddexp(ϕ+log1mexp(ϵ), log1mexp(ψ)+ϵ) - _n
#     ψp = ψ+log1mexp(ϵ) - _n
#     w = minfs(typeof(λ), mmax+1, mmax+1)
#     w[1,1] = 0.
#     w[2,2] = logaddexp(log(1. - q), log(2q)+ϵ) + log1mexp(ϵ)
#     w[2,3] = log(q) + 2*log1mexp(ϵ)
    for i=1:mmax, j=2:mmax
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1]
        # w[i+1, j+1] = logaddexp(w[2,2] + w[i, j], w[2,3] + w[i, j-1])
    end
    return log.(w)
end


# root integration and conditioning
# =================================
# this integrates the conditional probability at the root over the prior
# i.e. P(X) = Σₙ P(X|nroot=n)P(nroot=n)
function integrate_root(L::Vector{T}, η::T, ϵ::T) where T<:Real
    p = -Inf
    for i in 2:length(L)
        f = (i-1)*log1mexp(ϵ) + log(η) + (i-2)*log(1. - η)
        f -= i*log1mexp(log(1. - η)+ϵ)
        p = logaddexp(p, L[i] + f)
    end
    return p
end

# conditioning factor, i.e. the probability that a family is not extinct in
# both lineages stemming from the root
function condition_oib(tree::Arboreal, η::T, ϵ::Matrix{T}) where T<:Real
    e = findroot(tree)
    f, g  = childnodes(tree, e)
    lη = log(η)
    left = geometric_extinctionp(ϵ[f, 1], lη)
    rght = geometric_extinctionp(ϵ[g, 1], lη)
    p = try
        log1mexp(left) + log1mexp(rght)
    catch
        # XXX I had numerical issues here (with some data sets?)
        @error "Invalid P's at `condition_oib` \nleft = $left, right = $rght"
        -Inf
    end
    return p
end

#geometric_extinctionp(ϵ::T, η::T) where T<:Real = η*ϵ/(1. - (1. - η)*ϵ)
geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
geometric_extinctionp(ϵ::T, η::T) where T<:Real = η + ϵ -log1mexp(log1mexp(η)+ϵ)


# BDP utilities
# =============
getϕ(t, λ, μ) = λ ≈ μ ?
    λ*t/(1. + λ*t) : μ*(exp(t*(λ-μ)) - 1.)/(λ*exp(t*(λ-μ)) - μ)
getψ(t, λ, μ) = λ ≈ μ ?
    λ*t/(1. + λ*t) : (λ/μ)*getϕ(t, λ, μ)
getξ(i, j, k, t, λ, μ) = binomial(i, k)*binomial(i+j-k-1,i-1)*
    getϕ(t, λ, μ)^(i-k)*getψ(t, λ, μ)^(j-k)*(1-getϕ(t, λ, μ)-getψ(t, λ, μ))^k
tp(a, b, t, λ, μ) = (a == b == 0) ? 1.0 :
    sum([getξ(a, b, k, t, λ, μ) for k=0:min(a,b)])
# NOTE: when μ >> λ, numerical issues sometimes result in p > 1.   
ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    approx1((μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ)
approx1(x) = x ≈ one(x) ? one(x) : x

# The Csuros & Miklos algorithm
# =============================
"""
    csuros_miklos!(L::Matrix{T},
        x::AbstractVector{Int64},
        matrices::CsurosMiklos{T},
        tree::Arboreal,
        branches::Array{Int64})

Csuros & Miklos algorithm for computing conditional survival likelihoods. This
is equivalent to a pruning (dynamic programming) algorithm using the
conditional survival likelihoods. This will return The full dynamic programming
matrix.
"""
function csuros_miklos!(L::Matrix{T},
        x::AbstractVector{Int64},
        matrices::CsurosMiklos{T},
        tree::Arboreal,
        branches::Array{Int64}) where T<:Real
    @unpack W, ϵ, m = matrices
    mx = maximum(x)

    for e in branches
        if isleaf(tree, e)
            L[e, x[e]+1] = 0.
        else
            children = childnodes(tree, e)
            Mc = [x[c] for c in children]
            _M = cumsum([0 ; Mc])
            #_ϵ = cumprod([1.; [ϵ[c, 1] for c in children]])
            _ϵ = cumsum([0.; [ϵ[c, 1] for c in children]])
            B = minfs(eltype(_ϵ), length(children), _M[end]+1, mx+1)
            A = minfs(eltype(_ϵ), length(children), _M[end]+1)
            for i = 1:length(children)
                c = children[i]
                Mi = Mc[i]

                #B[i, 1, :] = log.(W[1:mx+1, 1:mx+1, c] * exp.(L[c, 1:mx+1]))
                B[i, 1, :] = log.(exp.(W[1:mx+1,1:mx+1,c]) * exp.(L[c,1:mx+1]))
                # ↑ take this for W on log scale

                for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                    if s == Mi
                        # B[i,t+1,s+1] = B[i,t,s+1] + log(ϵ[c,1])
                        B[i,t+1,s+1] = B[i,t,s+1] + ϵ[c,1]
                    else
                        #B[i,t+1,s+1] = logaddexp(
                        #    B[i,t,s+2], log(ϵ[c,1])+B[i,t,s+1])
                        B[i,t+1,s+1] = logaddexp(
                           B[i,t,s+2], ϵ[c,1]+B[i,t,s+1])
                    end
                end
                if i == 1
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        #A[i,n+1] = B[i,1,n+1] - n*log(1. - _ϵ[2])
                        A[i,n+1] = B[i,1,n+1] - n*log1mexp(_ϵ[2])
                    end
                else
                    # XXX is this loop as efficient as it could?
                    for n=0:_M[i+1], t=0:_M[i]
                        s = n-t
                        if s < 0 || s > Mi
                            continue
                        else
                            # p = _ϵ[i]
                            p = exp(_ϵ[i])
                            if !(zero(p) <= p <= one(p))
                                @error "Invalid extinction probability ($p)"
                                p = one(p)
                            end
                            lp = logpdf(Binomial(n, p), s) +
                                A[i-1,t+1] + B[i,t+1,s+1]
                            A[i,n+1] = logaddexp(A[i,n+1], lp)
                        end
                    end
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        #A[i,n+1] -= n*log(1. - _ϵ[i+1])
                        A[i,n+1] -= n*log1mexp(_ϵ[i+1])
                    end
                end
            end
            for n=0:x[e]
                L[e, n+1] = A[end, n+1]
            end
        end
    end
    return L
end

# FIXME I don't like this
_translate(s::Symbol) = s == :λ || s == :μ ? :θ : s

# initialize matrix of dimensions `dims` with -Inf
minfs(::Type{T}, dims::Tuple{}) where T<:Real = Array{T}(fill(-Inf, dims))
minfs(::Type{T}, dims::Union{Integer, AbstractUnitRange}...) where T<:Real =
    Array{T}(fill(-Inf, dims))
