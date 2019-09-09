#=
MCMC chain for the full models.

Thoughts:
    - The struct should probably be organized following the cluster structure. Storing the relevant DLModel instances. This makes the state dict redundant, so we should have an efficient way of modifying a DLModel temporarily, probably a constructor for the DLModel that takes an already operating instance of the DLModel and an arbitrary parameter value pair.
    - This also makes it possible to have an interface like LogDensityProblems, because all the possible 'custom' and finnicky speed-ups are in the construction of the DLModel (ϵ and W calculation). Actually this isn't true, and in principle we could further speed up the Csuros Miklos part by partial computations, but stuff gets quite intricate I fear...
=#

mutable struct DLChain
    X::Matrix{Int64}
    Ψ::SpeciesTree
    priors::Priors
    proposals::Proposals
    clusters::Array{DLModel,1}
    gen::Int64
end

struct RelativeRatesPrior{T}
    θ::T              # rates model, e.g. GBM or IID
    λ0::LogNormal     # root dup. rate prior
    μ0::LogNormal     # root loss rate prior
    η::Beta           # prior on geometric prior at the root
    rλ::Gamma         # prior on relative dup. rates
    rμ::Gamma         # prior on relative loss rates
end

function (prior::RelativeRatesPrior)(chain)
    @unpack θ, λ0, μ0, η, rλ, rμ = prior
    lp =  logpdf(θ, λ, ν)
    lp += logpdf(θ, μ, ν)
    lp += logpdf(λ0, λ[1])
    lp += logpdf(μ0, μ[1])
    lp += logpdf(η, λ[0])

end
