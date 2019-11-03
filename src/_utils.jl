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

geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
geometric_extinctionp(ϵ::T, η::T) where T<:Real = η + ϵ -log1mexp(log1mexp(η)+ϵ)
