# Utilities
# =========
minfs(::Type{T}, dims::Tuple{}) where T<:Real = Array{T}(fill(-Inf, dims))
minfs(::Type{T}, dims::Union{Integer, AbstractUnitRange}...) where T<:Real =
    Array{T}(fill(-Inf, dims))


# BDP utilities
# =============
getϕ(t, λ, μ) = λ ≈ μ ? λ*t/(1. + λ*t) : μ*(exp(t*(λ-μ))-1.)/(λ*exp(t*(λ-μ))-μ)
getψ(t, λ, μ) = λ ≈ μ ? λ*t/(1. + λ*t) : (λ/μ)*getϕ(t, λ, μ)

# NOTE: when μ >> λ, numerical issues sometimes result in p > 1.
ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    approx1((μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ)
approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x

geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
geometric_extinctionp(ϵ::T, η::T) where T<:Real = η + ϵ -log1mexp(log1mexp(η)+ϵ)


# Tree utilities
# ==============
# get distance from n to m, where it is guaranteed that m is above n
function parentdist(n::ModelNode{T}, m::ModelNode{T}) where T<:Real
    d = zero(T)
    while n != m
        d += n[:t]
        n = n.p
    end
    return d
end


id(node::ModelNode, args::Symbol...) = [id(node, s) for s in args]
id(node::ModelNode, s::Symbol) = Symbol("$s$(node.i)")
