# MCMC with a DP mixture
abstract type Chain end
abstract type MixtureModel end
abstract type DPMixture end
const State = Dict{Symbol,Union{Vector{<:Real},<:Real}}


# Chain for the Mixture models
# ============================
# The main tricky thing is that parameters may be on two levels, either across
# the entire genome or cluster-specific.

mutable struct Cluster{T<:PhyloBDP}  # mutable, since the model will change
    n::Int64  # current number of assignments
    w::Float64  # cluster weight
    model::T
    state::State
    proposals::Proposals
end

# it would probably work better if either the cluster or the PArray stores z
mutable struct MixtureChain{T<:MixtureModel,V<:PhyloBDP} <: Chain
    X::PArray
    z::Array{Int64}    # latent assignments XXX should this be here?
    gen::Int64
    tree::SpeciesTree
    state::State           # state for hyperparameters
    priors::Priors         # priors
    proposals::Proposals   # proposals for hyperparameters
    mixturemodel::T
    clusters::Array{Cluster{V},1}
end

# NOTE: Profile should be different for mixture case; would be good if it
# stored L matrices and likelihoods for all clusters, as well as the current
# cluster assignment

# NOTE: in the finite mixture case; we can parallelize the Gibbs sampler for
# latent assignments easily; as there is conditional independence of the zᵢ
# and z₋ᵢ given the weights

# gibbs sampler for latent assignments (parallel)
function move_latent_assignment!(chain)
    # distributed computation of new latent assignments
    # XXX is this possible, or does this screw up something with the weights?
    map((x)->_gibbs_latent_assignment!(x, chain.clusters), chain.X)
end

# We have the conditional independence p(zᵢ|θ, z₋ᵢ, x) = p(zᵢ|θ, xᵢ)
# and  p(zᵢ|θ, xᵢ) ∝ p(zᵢ)p(xᵢ|θzᵢ) = wᵢ p(xᵢ|θzᵢ)
function _move_latent_assignment!(x, clusters)
    # this does not a lot of work, as we assume logpdfs are stored
    logps = x.logp .+ [log(c.w) for c in clusters]
    pvec  = exp(logps - maximum(logps))
    pvec /= sum(pvec)
    i = rand(Categorical(pvec))
    clusters[i].n += 1
    x.z = i
end

function move_hyperparams!(chain)
    move_ν!(chain)  # reuse from non-mixture MCMC
    move_η!(chain)  # partly reuse from non-mixture MCMC
end

function move_clusterparams!(chain)
    # for computing the MH ratio, we only need to compute the lhood for the
    # currently assigned families of a particular cluster
    # *upon acceptance* of an update we should recompute the logpdf for all
    # families, as we need this lhood in the next iteration of drawing latent
    # assignments.
    for cluster in chain.clusters
        # reuse from non-mixture MCMC, perhaps using a type union?
        for f in [move_rates!, move_wgds!]
            accepted = f(cluster)
            if accepted
                logpdf!(cluster, chain.X)
            end
        end
    end
end
