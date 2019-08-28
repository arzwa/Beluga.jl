# DLModel
struct DLModel{T<:Real,Ψ<:Arboreal} <: PhyloLinearBDP
    tree::Ψ
    order::Array{Int64,1}
    b::Array{LinearBDP{T},1}
    ϵ::Array{T,2}  # extinction probabilities at beginning and end of branch
    ρ::DiscreteUnivariateDistribution  # distribution at root
end

# constructors
function DLModel(Ψ::SpeciesTree, o::Array{Int64,1}, b::Array{LinearBDP{T},1},
        ρ::DiscreteUnivariateDistribution) where T<:Real
    d = DLModel{T,SpeciesTree}(Ψ, o, b, ones(T, length(o), 2), ρ)
    get_ϵ!(d)
    return d
end

function DLModel(d::DLModel, x::Vector{<:Real})
    n = length(x) ÷ 2
    if n == 1
        b = [LinearBDP(x[1], x[2])]
        return DLModel(d.tree, d.order, b, d.ρ)
    else
        λ = x[1:n]
        μ = x[n+1:end]
        b = [LinearBDP(λ[i], μ[i]) for i in 1:n]
        return DLModel(d.tree, d.order, b, d.ρ)
    end
end

DLModel(Ψ::SpeciesTree, λ::Real, μ::Real, η::Real=0.9) =
    DLModel(Ψ, promote(λ, μ, η)...)

function DLModel(Ψ::SpeciesTree, λ::T, μ::T, η::T) where {T<:Real}
    n = maximum([v[:θ] for (k,v) in Ψ.bindex])
    DLModel(Ψ, postorder(Ψ), [LinearBDP(λ, μ) for i=1:n], Geometric(η))
end

DLModel(Ψ::SpeciesTree, λ::Vector{T}, μ::Vector{T}, η::T) where {T<:Real} =
    DLModel(Ψ, postorder(Ψ), [LinearBDP(λ[i], μ[i]) for i=1:length(λ)],
        Geometric(η))

# helpers
function asvector(d::DLModel)
    n = length(d.b)
    x = zeros(2n)
    for e in d.order
        i = d.tree.bindex[e, :θ]
        p = d[e, :θ]
        x[i] = p.λ
        x[i+n] = p.μ
    end
    return x
end

Base.show(io::IO, d::DLModel) = show(io, d, (:tree, :b))
Base.length(d::DLModel) = length(d.order)

# indexers, should think about what's best
Base.getindex(d::DLModel, i::Int64) = d[i, :θ]
Base.getindex(d::DLModel, i::Int64, j::Int64) = d.ϵ[i, j]
Base.getindex(d::DLModel, i::Int64, s::Symbol) = d.b[d.tree.bindex[i, s]]
Base.setindex!(d::DLModel{T}, v::T, i::Int64, j::Int64) where T<:Real =
    d.ϵ[i, j] = v

# extinction probabilities
function get_ϵ!(d::DLModel{T}) where T<:Real
    for e in d.order
        if isleaf(d, e)
            d[e, 2] = zero(T)
        else
            for c in childnodes(d, e)
                d[c, 1] = ep(d[c, :θ], T(parentdist(d, c)), d[c, 2])
                d[e, 2] *= d[c, 1]
            end
        end
    end
end

function get_wstar(d::DLModel{T}, M::Array{Int64,N}) where {N,T<:Real}
    mmax = maximum(M)
    w = zeros(T, mmax+1, mmax+1, maximum(d.order))
    # last dimension of w one too large, unnecessary memory...
    for i in d.order[1:end-1]
        w[:, :, i] = _wstar(d, i, mmax)
    end
    return w
end

function _wstar(d::DLModel{T}, e::Int64, mmax::Int64) where {T<:Real}
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(d[e, :θ], parentdist(d, e))  # p
    ψ = getψ(d[e, :θ], parentdist(d, e))  # q
    _n = 1. - ψ*d[e, 2]
    ϕp = (ϕ*(1. - d[e, 2]) + (1. - ψ)*d[e, 2]) / _n
    ψp = ψ*(1. - d[e, 2]) / _n
    w = zeros(T, mmax+1, mmax+1)
    w[1,1] = 1.
    # NOTE vectorize?
    for m=1:mmax, n=1:m
        w[n+1, m+1] = n == m ? (1. - ϕp)*(1. - ψp)*w[n, m] :
            ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
    return w
end

# note these are (should be) more general than DLModel
function Distributions.logpdf(d::PhyloLinearBDP,
        M::AbstractMatrix{Int64}, cond=:oib)
    W = get_wstar(d, M)
    return _logpdf(d, M, W, cond)
end

function Distributions.logpdf(d::PhyloLinearBDP,
        M::AbstractVector{Int64}, cond=:oib)
    W = get_wstar(d, M)
    return _logpdf(d, M, W, cond)
end

_logpdf(d::PhyloLinearBDP, M::AbstractMatrix{Int64}, W::Array{<:Real,3},
    cond=:oib) = mapreduce(
        (i) -> _logpdf(d, M[i,:], W, cond), +, collect(1:size(M)[1]))

function _logpdf(d::DLModel, x::AbstractVector{Int64}, W::Array{<:Real,3},
        cond=:oib)
    root = d.order[end]
    L = csuros_miklos(d, x, W)[root, :]
    L = integrate_root(L, d.ρ, d, root)
    return condition(L, d, cond)
end

#= NOTE: This approach for computing the gradient has quite some overhead
since we compute W for each vector again. It is however the only way to get
ForwardDiff to work with the distributed approach.
=#
function gradient(d::PhyloLinearBDP, M::AbstractMatrix{Int64}, cond=:oib)
    return mapreduce((i) -> gradient(d, M[i,:], cond), +, collect(1:size(M)[1]))
end

function gradient(d::DLModel, x::AbstractVector{Int64}, cond=:oib)
    v = asvector(d)
    f = (u) -> logpdf(DLModel(d, u), x, cond)
    g = ForwardDiff.gradient(f, v)
    return g[:, 1]
end

function condition(L::T, d::DLModel{T}, cond::Symbol) where {T<:Real}
    return if cond == :oib
        L - log(oib(d.ρ, d))
    elseif cond == :none
        L
    else
        @warn "`cond=$cond`: returning unconditional likelihood"
        L
    end
end

function integrate_root(L::Array{<:Real,1}, ρ::Geometric, d::DLModel, e::Int64)
    η = ρ.p
    p = 0.
    for i in 2:length(L)
        f = (1. -d[e,2])^(i-1)*η*(1. -η)^(i-2)/(1. -(1. -η)*d[e,2])^i
        p += exp(L[i]) * f
    end
    log(p)
end

function oib(ρ::Geometric, d::DLModel)
    e = d.order[end]
    η = ρ.p
    f, g  = childnodes(d.tree, e)
    #root = geometric_extinctionp(d[e, 2], η)
    left = geometric_extinctionp(d[f, 1], η)
    rght = geometric_extinctionp(d[g, 1], η)
    #1. - left - rght + root
    p = (1. -left)*(1. -rght)
    p = isapprox(p, zero(p), atol=1e-12) ? zero(p) : p  # XXX had some issues
    return p
end

geometric_extinctionp(ϵ::T, η::T) where T<:Real = η*ϵ/(1. - (1. - η)*ϵ)
geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
