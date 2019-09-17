# NOTE: assuming only linear BDPs
# NOTE: assuming geometric at root could be generailzed to uniform,
# Poisson, Negbinomial and geometric
# NOTE: only conditioning on one in both clades currently

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
"""
mutable struct DuplicationLossWGD{T<:Real,Ψ<:Arboreal} <: PhyloBDP
    tree::Ψ
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}
    η::T  # geometric prior probability at the root
    value::CsurosMiklos{T}
end

function DuplicationLossWGD(tree::Ψ, λ::Vector{T}, μ::Vector{T}, q::Vector{T},
        η::T, mmax::Int) where {Ψ<:Arboreal,T<:Real}
    ϵ = ones(T, length(tree.order), 2)
    # last dimension of W one too large, unnecessary memory...
    W = zeros(eltype(λ), mmax+1, mmax+1, maximum(tree.order))
    c = CsurosMiklos(W, ϵ, mmax)
    d = DuplicationLossWGD{T,Ψ}(tree, λ, μ, q, η, c)
    get_ϵ!(d)
    get_W!(d, mmax)
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
    bs = get_parentbranches(d.tree, i)
    get_ϵ!(d, bs)
    get_W!(d, d.value.m, bs)
end

# non-vector parameters require recomputation at the root only
function Base.setindex!(d::DuplicationLossWGD{T,Ψ}, v::T,
        s::Symbol) where {Ψ<:Arboreal,T<:Real}
    setfield!(d, s, v)
    bs = [findroot(d.tree)]
    get_ϵ!(d, bs)
    get_W!(d, d.value.m, bs)
end


# logpdf
# ======
function Distributions.logpdf(d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    L = zeros(T, length(d.tree), maximum(x)+1)
    l = logpdf!(L, d, x, d.tree.order)
    l, L
end

function Distributions.logpdf!(L::Matrix{T},
        d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64},
        branches::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    L = csuros_miklos!(L, x, d.value, d.tree, branches)
    root = branches[end]
    l = integrate_root(L[root,:], d.η, d.value.ϵ[root,2])
    try
        return l - log(condition_oib(d.tree, d.η, d.value.ϵ))
    catch
        p = condition_oib(d.tree, d.η, d.value.ϵ)
        @error "log error: log($p) at condition_oib\n $d"
        return -Inf
    end
end

# Different interface (than profile) to compute the accumulated logpdf over
# multiple phylogenetic profiles
function Distributions.logpdf(d::DuplicationLossWGD, X::AbstractMatrix{Int64})
    m = maximum(X)
    L = zeros(size(X)..., m+1)
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
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    ϵ[1,1] = NaN  # root has only one extinction P, NaN for safety
    for e in branches
        ϵ[e,:] .= 1.
        if isleaf(tree, e)
            ϵ[e, 2] = zero(eltype(ϵ))
        elseif Beluga.iswgd(tree, e)
            qe = q[tree[e, :q]]
            f = childnodes(tree, e)[1]
            #@show parentdist(tree, f)
            #@show ep(λ[f], μ[f], parentdist(tree, f), ϵ[f, 2])
            μf = μ[tree[f, :θ]]
            λf = λ[tree[f, :θ]]
            ϵ[f, 1] = ep(λf, μf, parentdist(tree, f), ϵ[f, 2])
            ϵ[f, 1] = ϵ[e, 2] = qe*ϵ[f, 1]^2 + (1. - qe)*ϵ[f, 1]  # HACK?
        else
            for c in childnodes(tree, e)
                μc = μ[tree[c, :θ]]
                λc = λ[tree[c, :θ]]
                ϵ[c, 1] = ep(λc, μc, parentdist(tree, c), ϵ[c, 2])
                ϵ[e, 2] *= ϵ[c, 1]
            end
        end
    end
end


# transition probabiities
# =======================
# W matrix for Csuros & Miklos algorithm
get_W!(d::DuplicationLossWGD, mmax::Int64) = get_W!(d, mmax, d.tree.order)

function get_W!(model::DuplicationLossWGD, mmax::Int64, branches::Vector{Int64})
    # XXX↓ WGD affects branch *below* a WGD!
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    for i in branches[1:end-1]  # excluding root node
        μi = μ[tree[i, :θ]]
        λi = λ[tree[i, :θ]]
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
    for m=1:mmax, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
    return w
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
    for i=1:mmax, j=2:mmax
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1]
    end
    return w
end


# root integration and conditioning
# =================================
function integrate_root(L::Vector{T}, η::T, ϵ::T) where T<:Real
    # XXX L not in log scale !
    p = 0.
    for i in 2:length(L)
        f = (1. - ϵ)^(i-1)*η*(1. -η)^(i-2)/(1. -(1. -η)*ϵ)^i
        p += L[i] * f
    end
    try
        return log(p)
    catch
        @error "log error: log($p);\n L = $L\n at integrate_root"
    end
end

function condition_oib(tree::Arboreal, η::T, ϵ::Matrix{T}) where T<:Real
    e = findroot(tree)
    f, g  = childnodes(tree, e)
    #root = geometric_extinctionp(ϵ[e, 2], η)
    left = geometric_extinctionp(ϵ[f, 1], η)
    rght = geometric_extinctionp(ϵ[g, 1], η)
    #1. - left - rght + root
    p = (1. -left)*(1. -rght)
    #p = isapprox(p, zero(p), atol=1e-12) ? zero(p) : p  # XXX had some issues
    return p
end

geometric_extinctionp(ϵ::T, η::T) where T<:Real = η*ϵ/(1. - (1. - η)*ϵ)
geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)


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
ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ


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
            L[e, x[e]+1] = 1.0
        else
            children = childnodes(tree, e)
            Mc = [x[c] for c in children]
            _M = cumsum([0 ; Mc])
            _ϵ = cumprod([1.; [ϵ[c, 1] for c in children]])
            B = zeros(eltype(_ϵ), length(children), _M[end]+1, mx+1)
            A = zeros(eltype(_ϵ), length(children), _M[end]+1)
            for i = 1:length(children)
                c = children[i]
                Mi = Mc[i]
                B[i, 1, :] = W[1:mx+1, 1:mx+1, c] * L[c, 1:mx+1]
                for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                    if s == Mi
                        B[i,t+1,s+1] = B[i,t,s+1] * ϵ[c,1]
                    else
                        B[i,t+1,s+1] = B[i,t,s+2] + ϵ[c,1]*B[i,t,s+1]
                    end
                end
                if i == 1
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] = B[i,1,n+1]/(1. - _ϵ[2])^n
                    end
                else
                    # XXX is this loop as efficient as it could?
                    for n=0:_M[i+1], t=0:_M[i]
                        s = n-t
                        p = _ϵ[i]
                        #p = isapprox(p, one(p)) ? one(p) : p  # had some problem
                        if s < 0 || s > Mi
                            continue
                        else
                            A[i,n+1] += pdf(Binomial(n, p), s) *
                            A[i-1,t+1] * B[i,t+1,s+1]
                        end
                    end
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] /= (1. - _ϵ[i+1])^n
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
