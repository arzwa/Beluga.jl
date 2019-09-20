# priors (i.e. rate models)
# =========================
# Note a prior should implement a logprior(d::RatesPrior, θ::NamedTuple)
# interface and a rand(d::RatesPrior, tree::Arboreal)

abstract type RatesPrior end
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}

Distributions.logpdf(x::Real, y) = 0.  # hack for constant priors


# Constant rates
# ==============
struct ConstantRatesPrior <: RatesPrior
    dλ::Prior
    dμ::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::ConstantRatesPrior, tree::Arboreal)
    @unpack dλ, dμ, dη, dq = d
    η = rand(dη)
    λ = rand(dλ)
    μ = rand(dμ)
    q = rand(dq, Beluga.nwgd(tree))
    return State(:η=>η, :λ=>[λ], :μ=>[μ], :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

"""
     logprior(d::ConstantRatesPrior, θ::NamedTuple)
"""
function logprior(d::ConstantRatesPrior, θ::NamedTuple)
    @unpack λ, μ, q, η = θ
    @unpack dλ, dμ, dq, dη = d
    lp  = logpdf(dλ, λ[1]) + logpdf(dμ, μ[1]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    return lp
end

logprior(d::ConstantRatesPrior, s::Dict, args...) = logprior(
    d, (λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))


# GBM rates
# =========
# this is a model without correlation of λ and μ
struct GBMRatesPrior <: RatesPrior
    dν::Prior
    dλ::Prior
    dμ::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::GBMRatesPrior, tree::Arboreal)
    # assumed single prior for q for now, easy to adapt though
    @unpack dν, dλ, dμ, dq, dη = d
    ν = rand(dν)
    η = rand(dη)
    λ0 = rand(dλ)
    μ0 = rand(dμ)
    # HACK not for general rate indices!
    n = [n for n in preorder(tree) if !(iswgd(tree, n) || iswgdafter(tree, n))]
    λ = rand(GBM(tree, λ0, ν))[n]
    μ = rand(GBM(tree, λ0, ν))[n]
    q = rand(dq, Beluga.nwgd(tree))
    return State(:ν=>ν, :η=>η, :λ=>λ, :μ=>μ, :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

"""
     logprior(d::GBMRatesPrior, θ::NamedTuple)
"""
function logprior(d::GBMRatesPrior, θ::NamedTuple)
    @unpack Ψ, ν, λ, μ, q, η = θ
    @unpack dν, dλ, dμ, dq, dη = d
    lp  = logpdf(dν, ν) + logpdf(dλ, λ[1]) + logpdf(dμ, μ[1]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    lp += logpdf(GBM(Ψ, λ[1], ν), λ)
    lp += logpdf(GBM(Ψ, μ[1], ν), μ)
    return lp
end

logprior(d::GBMRatesPrior, s::Dict, Ψ::Arboreal) = logprior(
    d, (Ψ=Ψ, ν=s[:ν], λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))

struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

Distributions.logpdf(d::LogUniform, x::T) where T<:Real =
    logpdf(Uniform(d.a, d.b), log10(x))

Base.rand(d::LogUniform) = exp(rand(Uniform(d.a, d.b)))
