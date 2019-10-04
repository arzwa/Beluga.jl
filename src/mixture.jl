# MCMC with a (DP) mixture
abstract type MixtureModel end
abstract type DPMixture end
const ChainOrCluster = Union{Chain,Cluster}

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

Base.getindex(w::Cluster, s::Symbol) = w.state[s]
Base.getindex(w::Cluster, s::Symbol, i::Int64) = w.state[s, i]
Base.setindex!(w::Cluster, x, s::Symbol) = w.state[s] = x
Base.setindex!(w::Cluster, x, s::Symbol, i::Int64) = w.state[s, i] = x

function logpdf!(cluster::Cluster, X::MixtureProfile)
    # this function computes logpdf for `cluster` for all profiles in X
    logpdf!()
end

function logpdf!(clusters::Vector{Cluster}, X::MixtureProfile)
    # this function computes the logpdf for each family for the current mixture
    # based on its assignment
end

# it would probably work better if either the cluster or the PArray stores z
mutable struct MixtureChain{T<:MixtureModel,V<:PhyloBDP} <: Chain
    X::PArray
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
    # I think it might only screw up the weights in case of the DP mixture?
    map((x)->_gibbs_latent_assignment!(x, chain.clusters), chain.X)
end

# We have the conditional independence p(zᵢ|θ, z₋ᵢ, x) = p(zᵢ|θ, xᵢ)
# and  p(zᵢ|θ, xᵢ) ∝ p(zᵢ)p(xᵢ|θzᵢ) = wᵢ p(xᵢ|θzᵢ)
function _move_latent_assignment!(x, clusters)
    # this does not a lot of work, as we assume logpdfs are stored
    logp  = x.logp .+ [log(c.w) for c in clusters]
    pvec  = exp(logp - maximum(logp))
    pvec /= sum(pvec)
    i = rand(Categorical(pvec))
    clusters[i].n += 1
    x.z = i
end

function move_hyperparams!(chain)
    move_ν!(chain)  # reuse from non-mixture MCMC
    move_η!(chain)  # partly reuse from non-mixture MCMC TODO
end

# TODO weights!

function move_clusterparams!(chain)
    # for computing the MH ratio, we only need to compute the lhood for the
    # currently assigned families of a particular cluster
    # *upon acceptance* of an update we should recompute the logpdf for all
    # families, as we need this lhood in the next iteration of drawing latent
    # assignments.
    # Actually; we probably want to compute the logpdfs for all clusters just
    # once at the end of this function
    for (k, c) in enumerate(chain.clusters)
        move_rates!(c, chain.X, chain.priors, chain.tree, chain.state, k)
        logpdf_allother!(c.model, chain.X, -1, k)
        set_L!(chain.X, k)
        # TODO add wgds
    end
end

# initially perhaps better to keep it separated from non-mixture case...
function move_rates!(cluster, data, priors, tree, state, gen, k)
    seen = Set{Int64}()  # HACK store indices that were already done
    for i in tree.order
        idx = tree[i,:θ]
        idx in seen ? continue : push!(seen, idx)
        prop = cluster.proposals[:λ,idx]
        λi, hr1 = prop(cluster[:λ,idx])
        μi, hr2 = prop(cluster[:μ,idx])
        p_ = logprior(priors, cluster.state, tree,
            :λ=>(idx, λi), :μ=>(idx, μi))  # prior
        d = deepcopy(cluster.model)
        d[:μ, i] = μi   # NOTE: implementation of setindex! uses node indices!
        d[:λ, i] = λi   # NOTE: implementation of setindex! uses node indices!
        l_ = logpdf!(d, data, i, k)  # likelihood
        # NOTE: loglikelihood from global chain; logprior from cluster!
        mhr = l_ + p_ - state[:logp] - chain[:logπ] + hr1 + hr2
        if log(rand()) < mhr
            set_L!(data, k)    # update L matrix
            cluster.model = d
            cluster[:λ, idx] = λi
            cluster[:μ, idx] = μi
            cluster[:logπ] = p_
            state[:logp] = l_
            prop.accepted += 1
        else
            set_Ltmp!(data, k)  # revert Ltmp matrix
        end
        consider_adaptation!(prop, chain[:gen])
    end
end
