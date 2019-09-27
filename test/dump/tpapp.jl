using TransformVariables, DynamicHMC, LogDensityProblems, Random, CSV,
    DataFrames, ForwardDiff, Beluga, Parameters, Distributions

struct DLWGDProblem
    Ψ::SpeciesTree
    X::PArray
    m::Int64
end

# independent, non-hierarchical prior
function (problem::DLWGDProblem)(θ)
    @unpack X, Ψ, m = problem
    @unpack λ, μ, q, η = θ
    # priors
    ll = logpdf(Beta(8,2), η)
    ll += sum(logpdf.(Beta(), q))
    ll += sum(logpdf.(Exponential(), λ))
    ll += sum(logpdf.(Exponential(), μ))
    # lhood
    d = DuplicationLossWGD(Ψ, λ, μ, q, η, m)
    ll += logpdf(d, X)
    ll
end

# data
s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts1.tsv", delim="\t")
deletecols!(df, :Orthogroup)
p, m = Profile(df, s)
n = Beluga.nrates(s)

# problem and inits
problem = DLWGDProblem(s, p, m)
θ = (λ=exp.(randn(n)), μ=exp.(randn(n)), q=rand(Beluga.nwgd(s)), η=0.8)
@show problem(θ)

# transformation
problem_transformation(p::DLWGDProblem) =
    as((λ=as(Array, asℝ₊, n),
        μ=as(Array, asℝ₊, n),
        q=as(Array, as𝕀, nwgd(p.Ψ)),
        η=as𝕀))
P = TransformedLogDensity(problem_transformation(problem), problem)
∇P = ADgradient(:ForwardDiff, P)

# mcmc
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 10)


d = DuplicationLoss(s, exp.(randn(n)), exp.(randn(n)), 0.8, m)
gradient(d, p) |> show
