# DLModel
# We keep ϵ and W in the model struct, as they only change when λ or μ change;
struct DLModel{T<:Real,Ψ<:Arboreal} <: PhyloLinearBDP
    tree::Ψ
    b::Array{LinearBDP{T},1}
    ρ::DiscreteUnivariateDistribution  # distribution at root
    ϵ::Array{T,2}  # extinction probabilities at beginning and end of branch
    W::Array{T,3}  # auxiliary matrix Csuros & Miklos algorithm
    max::Int64
end

# constructors
DLModel(Ψ::SpeciesTree, mmax::Int64, λ::Vector{T}, μ::Vector{T},
    η::T) where {T<:Real} = DLModel(Ψ, mmax, [LinearBDP(λ[i], μ[i]) for
        i=1:length(λ)], Geometric(η))

DLModel(Ψ::SpeciesTree, mmax::Int64, λ::Real, μ::Real, η::Real=0.9) =
    DLModel(Ψ, mmax, promote(λ, μ, η)...)

function DLModel(Ψ::SpeciesTree, mmax::Int64, b::Array{LinearBDP{T},1},
        ρ::DiscreteUnivariateDistribution) where T<:Real
    W = zeros(T, mmax+1, mmax+1, maximum(Ψ.order))
    d = DLModel{T,SpeciesTree}(Ψ, b, ρ, ones(T, length(Ψ.order), 2), W, mmax)
    get_ϵ!(d)
    get_wstar!(d, mmax)
    return d
end

function DLModel(Ψ::SpeciesTree, mmax::Int64, λ::T, μ::T, η::T) where {T<:Real}
    n = maximum([v[:θ] for (k,v) in Ψ.bindex])
    DLModel(Ψ, mmax, [LinearBDP(λ, μ) for i=1:n], Geometric(η))
end

function DLModel(d::DLModel, λ::Vector{T}, μ::Vector{T}, η::T) where {T<:Real}
    return DLModel(d.tree, d.max, λ, μ, η)
end

function DLModel(d::DLModel, x::Vector{<:Real})
    n = length(x) ÷ 2
    return n == 1 ? DLModel(d.tree, d.max, [LinearBDP(x[1], x[2])], d.ρ) :
        DLModel(d, x[1:n], x[n+1:end])
end

# helpers
function asvector(d::DLModel)
    n = length(d.b)
    x = zeros(2n)
    for e in d.tree.order
        i = d.tree.bindex[e, :θ]
        p = d[e, :θ]
        x[i] = p.λ
        x[i+n] = p.μ
    end
    return x
end

Base.show(io::IO, d::DLModel) = show(io, d, (:tree, :b))
Base.length(d::DLModel) = length(d.tree.order)

# indexers, should think about what's best
rateindex(d::DLModel, branch::Int64) = d.tree.bindex[branch]
Base.getindex(d::DLModel, i::Int64) = d[i, :θ]
Base.getindex(d::DLModel, i::Int64, j::Int64) = d.ϵ[i, j]
Base.getindex(d::DLModel, i::Int64, s::Symbol) = d.b[d.tree.bindex[i, s]]
Base.setindex!(d::DLModel{T}, v::T, i::Int64, j::Int64) where T<:Real =
    d.ϵ[i, j] = v

# extinction probabilities
function get_ϵ!(d::DLModel{T}) where T<:Real
    for e in d.tree.order
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

# XXX should implement partial recomputation !
# note that wstar only requires the maximum of the profile
function get_wstar!(d::DLModel{T}, mmax::Int) where {N,T<:Real}
    # last dimension of w one too large, unnecessary memory...
    for i in d.tree.order[1:end-1]  # excluding root node
        d.W[:, :, i] = _wstar(d, i, mmax)
    end
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
    return _logpdf(d, M, cond)
end

function Distributions.logpdf(d::PhyloLinearBDP,
        M::AbstractVector{Int64}, cond=:oib)
    return _logpdf(d, M, cond)
end

_logpdf(d::PhyloLinearBDP, M::AbstractMatrix{Int64}, cond=:oib) =
    sum(mapslices((x)->_logpdf(d, x, cond), M, dims=2))

function _logpdf(d::DLModel, x::AbstractVector{Int64}, cond=:oib)
    root = d.tree.order[end]
    L = csuros_miklos(d, x, d.W)[root, :]
    L = integrate_root(L, d.ρ, d, root)
    return condition(L, d, cond)
end

#= NOTE: This approach for computing the gradient has quite some overhead
since we compute W for each vector again. It is however the only way to get
ForwardDiff to work with the distributed approach.
=#
function gradient(d::PhyloLinearBDP, M::AbstractMatrix{Int64}, cond=:oib)
    return sum(Array(mapslices((x)->gradient(d, x, cond),
        M, dims=2)'), dims=2)[:,1]
    #return mapreduce((i) -> gradient(d, M[i,:], cond), +, collect(1:size(M)[1]))
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
    e = d.tree.order[end]
    η = ρ.p
    f, g  = childnodes(d.tree, e)
    #root = geometric_extinctionp(d[e, 2], η)
    left = geometric_extinctionp(d[f, 1], η)
    rght = geometric_extinctionp(d[g, 1], η)
    #1. - left - rght + root
    p = (1. -left)*(1. -rght)
    #p = isapprox(p, zero(p), atol=1e-12) ? zero(p) : p  # XXX had some issues
    return p
end

geometric_extinctionp(ϵ::T, η::T) where T<:Real = η*ϵ/(1. - (1. - η)*ϵ)
geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
