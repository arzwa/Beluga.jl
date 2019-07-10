# Arthur Zwaenepoel 2019 - based on WGDgc (Cecile Ane) and Csuros & Miklos 2009
abstract type GeneContentCTMC <: DiscreteMultivariateDistribution end

# extensions
Distributions.insupport(d::GeneContentCTMC, x::AbstractVector) = all(x .> 0)

PhyloTrees.isleaf(d::GeneContentCTMC, n::Int64) = isleaf(d.tree, n)
PhyloTrees.childnodes(d::GeneContentCTMC, n::Int64) = childnodes(d.tree, n)
PhyloTrees.parentdist(d::GeneContentCTMC, n::Int64) = parentdist(d.tree, n)

Base.length(tree::Arboreal) = length(tree.tree.nodes)  # XXX to PhyloTrees!

# map from leaves to consecutive indices
leafmap(tree::Arboreal) = Dict(k=>i for (i,(k,v)) in enumerate(tree.leaves))

# DL model (branch-wise rates by default)
# NOTE: The only thing that differs between DL and DL+gain is w*[m|n], so this
# could be generalized, or maybe we need an abstract type for the two together.
# XXX: should use a `rindex`
struct DLModel{T<:Real,Ψ<:Arboreal} <: GeneContentCTMC
    tree::Ψ
    porder::Array{Int64,1}
    leafmap::Dict{Int64,Int64}
    b::Array{LinearBDP,1}
    ϵ::Array{T,2}  # extinction probabilities at beginning and end of branch
    ρ::DiscreteUnivariateDistribution  # distribution at root
end

function DLModel(t::Ψ, o::Array{Int64}, b::Array{LinearBDP{T},1},
        ρ::DiscreteUnivariateDistribution) where {T<:Real,Ψ<:Arboreal}
    d = DLModel{T,Ψ}(t, o, leafmap(t), b, ones(T, length(o), 2), ρ)
    get_ϵ!(d)
    return d
end

function DLModel(tree, x::Array{T}, η::T) where T<:Real
    n = length(x) ÷ 2
    n == 1 ? DLModel(tree, x[1], x[2], η) :
        DLModel(tree, x[1:n], x[n+1:end], η)
end

function DLModel(tree, λ::Array{T}, μ::Array{T}, η::T=0.9) where T<:Real
    o = postorder(tree)
    b = [LinearBDP(λ[i], μ[i]) for i in o]
    DLModel(tree, o, b, Geometric(η))
end

DLModel(tree, λ, μ, η=0.9) = DLModel(tree, promote(λ, μ, η)...)
DLModel(tree, λ::T, μ::T, η::T=0.9) where T<:Real = DLModel(tree, repeat([λ],
    length(tree)), repeat([μ], length(tree)))

# helpers for DLModel
Base.length(d::DLModel) = length(d.porder)
Base.show(io::IO, d::DLModel) = show(io, d, (:leafmap, :porder))
Base.getindex(d::DLModel, i::Int64, s::Symbol) = getfield(d.b[i], s)
Base.getindex(d::DLModel, i::Int64, j::Int64) = d.ϵ[i, j]
Base.getindex(d::DLModel, i::Int64) = d.b[i]
Base.getindex(d::DLModel, s::Symbol) = [getfield(d.b[i], s) for i in d.porder]
Base.setindex!(d::DLModel{T}, v::T, i::Int64, j::Int64) where T<:Real = d.ϵ[i, j] = v
Base.setindex!(d::DLModel{T}, v::T, i::Int64, s::Symbol) where T<:Real = getfield(d, s)[i] = v

lmap(d::DLModel, e::Int64) = d.leafmap[e]

# likelihood routines
function _logpdf(d::DLModel, x::AbstractArray{Int64}, cond=:oib)
    M = get_M(d, x)
    W = get_wstar(d, M)
    return _logpdf(d, M, W, cond)
end

function _logpdf(d::DLModel{T,Ψ}, M::AbstractVector{Int64},
        W::Array{T,3}, cond=:oib) where {T<:Real,Ψ<:Arboreal}
    root = d.porder[end]
    L = csuros_miklos(d, M, W)[root, :]
    L = integrate_root(L, d.ρ, d, root)
    return condition(L, d, cond)
end

_logpdf(d::DLModel{T,Ψ}, M::AbstractMatrix{Int64}, W::Array{T,3}, cond=:oib) where {T<:Real,Ψ<:Arboreal} = mapreduce((i)->
        _logpdf(d, M[i,:], W), +, collect(1:size(M)[1]))

_gradient(d::DLModel)

# extinction probabilities
function get_ϵ!(d::DLModel)
    for e in d.porder
        if isleaf(d, e)
            d[e, 2] = 0.0
        else
            for c in childnodes(d, e)
                d[c, 1] = ep(d[c], parentdist(d, c), d[c, 2])
                d[e, 2] *= d[c, 1]
            end
        end
    end
end

#XXX rename to something more sensible like `extend_profile`
# X could be more generally a data frame, and lmap an Int64=>Symbol dict
function get_M(d::DLModel, X::AbstractMatrix)
    M = zeros(Int64, size(X)[1], length(d.porder))
    for n in d.porder
        M[:, n] = isleaf(d,n) ? X[:, lmap(d,n)] :
            sum([M[:, c] for c in childnodes(d,n)])
    end
    return M
end

function get_M(d::DLModel, x::AbstractVector)
    M = zeros(Int64, length(d.porder))
    for n in d.porder
        M[n] = isleaf(d,n) ? x[lmap(d,n)] : sum([M[c] for c in childnodes(d,n)])
    end
    return M
end

function get_wstar(d::DLModel, M::Array{Int64,N}) where N
    mmax = maximum(M)
    w = zeros(mmax+1, mmax+1, maximum(d.porder))
    # last dimension of w one too large, unnecessary memory...
    for i in d.porder[1:end-1]
        w[:, :, i] = _wstar(d, i, mmax)
    end
    return w
end

function _wstar(d::DLModel, e::Int64, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(d[e], parentdist(d, e))  # p
    ψ = getψ(d[e], parentdist(d, e))  # q
    _n = 1. - ψ*d[e, 2]
    ϕp = (ϕ*(1. - d[e, 2]) + (1. - ψ)*d[e, 2]) / _n
    ψp = ψ*(1. - d[e, 2]) / _n
    w = zeros(mmax+1, mmax+1)
    w[1,1] = 1.
    # NOTE vectorize?
    for m=1:mmax, n=1:m
        w[n+1, m+1] = n == m ? (1. - ϕp)*(1. - ψp)*w[n, m] :
            ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
    return w
end

# Csuros & Miklos algorithm
# verified against WGDgc, type stable
function csuros_miklos(d::DLModel{U,Ψ}, M::T, W::Array{U,3}) where
        {T<:AbstractVector,U<:Real,Ψ<:Arboreal}
    mx = maximum(M)
    L = zeros(length(d.tree), mx+1)
    for e in d.porder
        if isleaf(d, e)
            L[e, M[e]+1] = 1.0
        else
            children = childnodes(d, e)
            Mc = [M[c] for c in children]
            _M = cumsum([0 ; Mc])
            _ϵ = cumprod([1.; [d[c, 1] for c in children]])
            B = zeros(length(children), _M[end]+1, mx+1)
            A = zeros(length(children), _M[end]+1)
            for i = 1:length(children)
                c = children[i]
                Mi = Mc[i]
                B[i, 1, :] = W[1:mx+1, 1:mx+1, c] * L[c, :]  # correct I believe
                for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                    if s == Mi
                        B[i,t+1,s+1] = B[i,t,s+1] * d[c,1]
                    else
                        B[i,t+1,s+1] = B[i,t,s+2] + d[c,1]*B[i,t,s+1]
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
                        s < 0 || s > Mi ? continue : A[i,n+1] += pdf(Binomial(
                            n, _ϵ[i]), s) * A[i-1,t+1] * B[i,t+1,s+1]
                    end
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] /= (1. - _ϵ[i+1])^n
                    end
                end
            end
            for n=0:M[e]
                L[e, n+1] = A[end, n+1]
            end
        end
    end
    log.(L)
end

function condition(L::Float64, d::DLModel, cond::Symbol)
    return if cond == :oib
        L - log(oib(d.ρ, d))
    elseif cond == :none
        L
    else
        @warn "`cond=$cond`: returning unconditional likelihood"
        L
    end
end

function integrate_root(L::Array{Float64,1}, ρ::Geometric, d::DLModel, e::Int64)
    η = ρ.p
    p = 0.
    for i in 2:length(L)
        f = (1. -d[e,2])^(i-1)*η*(1. -η)^(i-2)/(1. -(1. -η)*d[e,2])^i
        p += exp(L[i]) * f
    end
    log(p)
end

function oib(ρ::Geometric, d::DLModel)
    e = d.porder[end]
    η = ρ.p
    f, g  = childnodes(d.tree, e)
    #root = geometric_extinctionp(d[e, 2], η)
    left = geometric_extinctionp(d[f, 1], η)
    rght = geometric_extinctionp(d[g, 1], η)
    #1. - left - rght + root
    (1. -left)*(1. -rght)
end

geometric_extinctionp(ϵ::Float64, η::Float64) = η*ϵ/(1. - (1. - η)*ϵ)
