# trying to see whether it makes sens to use `logdensityproblems`

using Statistics
using Parameters
using TransformVariables
using LogDensityProblems

# Example from LogDensityProblems ==============================================
struct NormalPosterior{T} # contains the summary statistics
    N::Int
    x̄::T
    S::T
end

# calculate summary statistics from a data vector
function NormalPosterior(x::AbstractVector)
    NormalPosterior(length(x), mean(x), var(x; corrected = false))
end

# define a callable that unpacks parameters, and evaluates the log likelihood
function (problem::NormalPosterior)(θ)
    @unpack μ, σ = θ
    @unpack N, x̄, S = problem
    loglikelihood = -N * (log(σ) + (S + abs2(μ - x̄)) / (2 * abs2(σ)))
    logprior = - abs2(σ)/8 - abs2(μ)/50
    loglikelihood + logprior
end

problem = NormalPosterior(randn(100))
lhood = problem((μ = 0.0, σ = 1.0))
tproblem = TransformedLogDensity(as((μ = asℝ, σ = asℝ₊)), problem)
LogDensityProblems.logdensity(tproblem, zeros(2))
# ==============================================================================

mutable struct DLModelPost{T}
    Ψ::SpeciesTree
    X::Matrix{T}
    m::T
end

function (problem::DLModelPost)(θ)  # θ are the parameters for inference
    @unpack λ, μ, η = θ
    @unpack Ψ, X, m = problem
    d = DLModel(Ψ, m, λ, μ, η)
    loglikelihood = logpdf(d, X)
end

posterior = DLModelPost{eltype(M)}(s, M, maximum(M))
@time posterior((λ=0.2, μ=0.3, η=0.9))

a = @with_kw (λ = 0.2, μ = 0.3)
x = a()

v = as((λ = as_positive_real, μ = as_positive_real, η = as_unit_interval))
x = randn(dimension(v))
@time y = TransformVariables.transform(v, x)

v = as((λ = as(Array, asℝ₊, 10), μ = as(Array, asℝ₊, 10), η = as_unit_interval))
x = randn(dimension(v))
@time y = TransformVariables.transform(v, x)
