# priors (i.e. rate models)
# =========================
# Note a prior should implement a logprior(d::RatesPrior, θ::NamedTuple)
# interface and a rand(d::RatesPrior, tree::Arboreal)

abstract type RatesPrior end
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}


# LogUniform
struct LogUniform{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

Distributions.logpdf(d::LogUniform, x::T) where T<:Real =
    logpdf(Uniform(d.a, d.b), log10(x))

Base.rand(d::LogUniform) = exp(rand(Uniform(d.a, d.b)))

# HACK for constant priors
Base.rand(x::Real) = x
Distributions.logpdf(x::Real, y) = 0.


# IID rates
# =========
"""
    IIDRatesPrior(ν, r, q, η)

Branch-wise duplication and loss rates are assumed to be iid distributed
according to a LogNormal distribution with mean λ0 (for the duplication rate)
and μ0 (for the loss rate) and standard deviation ν. Any bivariate distribution
can be chosen as prior for (λ0, μ0), and it makes sense to use a Multivariate
log-normal with non-ero covariance; as too different duplication and loss rates
generally result in computational issues.

e.g. `prior = IIDRatesPrior(Exponential(0.1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.45 ; 0.45 .5]), Beta(1,1), Beta(8,2))``
"""
struct IIDRatesPrior <: RatesPrior
    dν::Prior
    dr::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::IIDRatesPrior, tree::Arboreal)
    # assumed single prior for q for now, easy to adapt though
    @unpack dν, dr, dq, dη = d
    ν = rand(dν)
    η = rand(dη)
    λ0, μ0 = rand(dr)
    n = Beluga.nrates(tree)
    λ = rand(LogNormal(log(λ0), ν), n)
    μ = rand(LogNormal(log(μ0), ν), n)
    q = rand(dq, Beluga.nwgd(tree))
    return State(:ν=>ν, :η=>η, :λ=>λ, :μ=>μ, :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

function logprior(d::IIDRatesPrior, θ::NamedTuple)
    @unpack Ψ, ν, λ, μ, q, η = θ
    @unpack dν, dr, dq, dη = d
    lp  = logpdf(dν, ν) + logpdf(dr, [λ[1], μ[1]]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    lp += sum(logpdf.(LogNormal(log(λ[1]), ν), λ[2:end]))
    lp += sum(logpdf.(LogNormal(log(μ[1]), ν), μ[2:end]))
    return lp
end

logprior(d::IIDRatesPrior, s::Dict, Ψ::Arboreal) = logprior(
    d, (Ψ=Ψ, ν=s[:ν], λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))


# Constant rates
# ==============
struct ConstantRatesPrior <: RatesPrior
    dr::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::ConstantRatesPrior, tree::Arboreal)
    @unpack dr, dη, dq = d
    η = rand(dη)
    λ, μ = rand(dr)
    q = rand(dq, Beluga.nwgd(tree))
    return State(:η=>η, :λ=>[λ], :μ=>[μ], :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

function logprior(d::ConstantRatesPrior, θ::NamedTuple)
    @unpack λ, μ, q, η = θ
    @unpack dr, dq, dη = d
    lp  = logpdf(dr, [λ[1], μ[1]]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    return lp
end

logprior(d::ConstantRatesPrior, s::Dict, args...) = logprior(
    d, (λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))


# GBM rates
# =========
struct GBMRatesPrior <: RatesPrior
    dν::Prior
    dr::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::GBMRatesPrior, tree::Arboreal)
    # assumed single prior for q for now, easy to adapt though
    @unpack dν, dr, dq, dη = d
    ν = rand(dν)
    η = rand(dη)
    λ0, μ0 = rand(dr)
    # HACK not for general rate indices!
    n = [n for n in preorder(tree) if !(iswgd(tree, n) || iswgdafter(tree, n))]
    λ = rand(GBM(tree, λ0, ν))[n]
    μ = rand(GBM(tree, μ0, ν))[n]
    q = rand(dq, Beluga.nwgd(tree))
    return State(:ν=>ν, :η=>η, :λ=>λ, :μ=>μ, :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

function logprior(d::GBMRatesPrior, θ::NamedTuple)
    @unpack Ψ, ν, λ, μ, q, η = θ
    @unpack dν, dr, dq, dη = d
    lp  = logpdf(dν, ν) + logpdf(dr, [λ[1], μ[1]]) + logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    lp += logpdf(GBM(Ψ, λ[1], ν), λ)
    lp += logpdf(GBM(Ψ, μ[1], ν), μ)
    return lp
end

logprior(d::GBMRatesPrior, s::Dict, Ψ::Arboreal) = logprior(
    d, (Ψ=Ψ, ν=s[:ν], λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))


# non-hierarchical rates prior
# ============================
struct NhRatesPrior <: RatesPrior
    dr::Prior
    dq::Prior
    dη::Prior
end

function Base.rand(d::NhRatesPrior, tree::Arboreal)
    @unpack dr, dq, dη = d
    η = rand(dη)
    n = Beluga.nrates(tree)
    r = rand(dr, n)
    λ = r[1,:]
    μ = r[2,:]
    q = rand(dq, Beluga.nwgd(tree))
    return State(:η=>η, :λ=>λ, :μ=>μ, :q=>q, :logp=>-Inf, :logπ=>-Inf)
end

function logprior(d::NhRatesPrior, θ::NamedTuple)
    @unpack Ψ, λ, μ, q, η = θ
    @unpack dr, dq, dη = d
    lp  = logpdf(dη, η)
    lp += sum(logpdf.(dq, q))
    lp += sum(logpdf(dr, [λ μ]'))
    return lp
end

logprior(d::NhRatesPrior, s::Dict, Ψ::Arboreal) = logprior(
    d, (Ψ=Ψ, λ=s[:λ], μ=s[:μ], q=s[:q], η=s[:η]))
