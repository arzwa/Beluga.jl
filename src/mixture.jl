# MCMC with a (DP) mixture

# Chain for the Mixture models
# ============================

mutable struct MixtureChain <: Chain
    X::MPArray
    tree::SpeciesTree
    state::State
    models::Array{DuplicationLossWGD,1}
    priors::RatesPrior
    proposals::Proposals
    trace::DataFrame
end

function MixtureChain(X::MPArray, prior::RatesPrior, tree::SpeciesTree,
        K::Int64, m::Int64)
    @unpack state, models, proposals = init_finitemixture(prior, s, K, m)
    mchain = MixtureChain(X, tree, state, models, prior, proposals, DataFrame())
    l = logpdf!(mchain.models, X, -1)
    for i=1:K ; set_L!(mchain.X, i); end
    set_logpdf!(mchain)
    mchain
end

function init_finitemixture(prior, tree, K, m)
    init = rand(prior, tree)
    init[:gen] = 0; init[:logπ] = 0
    clusterparams = [:λ, :μ, :q]
    for p in clusterparams
        delete!(init, p)
    end
    models = DuplicationLossWGD[]
    for i=1:K
        cluster = rand(prior, tree)
        for p in clusterparams
            init[Symbol("$p$i")] = cluster[p]
        end
        push!(models, DuplicationLossWGD(
            tree, cluster[:λ], cluster[:μ], cluster[:q], init[:η], m))
    end
    init[:w] = rand(Dirichlet(K,1.))
    init[:logπ] = logprior(tree, init, prior, K)
    proposals = get_defaultproposals(init)
    (state=init, proposals=proposals, models=models)
end


set_logpdf!(mchain::MixtureChain) = mchain.state[:logp] = curr_logpdf(mchain)
curr_logpdf(mchain::MixtureChain) = mapreduce(x->x.l[x.z], +, mchain.X)

function logprior(mchain::MixtureChain, args...)
    @unpack tree, state, priors, models = mchain
    logprior(tree, state, priors, length(models), args...)
end

function logprior(tree::SpeciesTree, state::State, priors, K, args...)
    s = deepcopy(state)
    for (k, v) in args
        if haskey(s, k)
            typeof(v)<:Tuple ? s[k][v[1]] = v[2] : s[k] = v
        else
            @warn "Trying to set unexisting variable ($k)"
        end
    end
    logπ = 0.
    for i=1:K
        θ = (Ψ=tree, ν=s[:ν], λ=s[Symbol("λ$i")],
            μ=s[Symbol("μ$i")], q=s[Symbol("q$i")], η=s[:η])
        logπ += Beluga.logprior(priors, θ) + log(s[:w, i])
        # XXX is this correct, should we include the weight?
    end
    logπ
end

# NOTE: Profile should be different for mixture case; would be good if it
# stored L matrices and likelihoods for all clusters, as well as the current
# cluster assignment

# NOTE: in the finite mixture case; we can parallelize the Gibbs sampler for
# latent assignments easily; as there is conditional independence of the zᵢ
# and z₋ᵢ given the weights

function mcmc!(chain, n; show_trace=true, show_every=10)
    for i=1:n
        move_latent_assignment!(chain)
        move_clusterparams!(chain)
        log_mcmc!(chain, stdout, show_trace, show_every)
    end
end

# gibbs sampler for latent assignments (parallel)
function move_latent_assignment!(chain)
    # distributed computation of new latent assignments
    # XXX is this possible, or does this screw up something with the weights?
    # I think it might only screw up the weights in case of the DP mixture?
    logweights = log.(chain[:w])
    map((x)->_move_latent_assignment!(x, logweights), chain.X)
    set_logpdf!(chain)
end

# We have the conditional independence p(zᵢ|θ, z₋ᵢ, x) = p(zᵢ|θ, xᵢ)
# and  p(zᵢ|θ, xᵢ) ∝ p(zᵢ)p(xᵢ|θzᵢ) = wᵢ p(xᵢ|θzᵢ)
function _move_latent_assignment!(x, logweights)
    # this does not a lot of work, as we assume logpdfs are stored
    logp  = x.l .+ logweights
    pvec  = exp.(logp .- maximum(logp))
    pvec /= sum(pvec)
    i = rand(Categorical(pvec))
    #clusters[i].n += 1
    x.z = i
end

function move_clusterparams!(chain)
    # for computing the MH ratio, we only need to compute the lhood for the
    # currently assigned families of a particular cluster
    # *upon acceptance* of an update we should recompute the logpdf for all
    # families, as we need this lhood in the next iteration of drawing latent
    # assignments.
    # Actually; we probably want to compute the logpdfs for all clusters just
    # once at the end of this function
    for (k, c) in enumerate(chain.models)
        move_rates!(chain, k)
        logpdf_allother!(c, chain.X, k, -1)
        set_L!(chain.X, k)
        # TODO add wgds
    end
end

# initially perhaps better to keep it separated from non-mixture case...
function move_rates!(chain, k)
    seen = Set{Int64}()  # HACK store indices that were already done
    λk = Symbol("λ$k")
    μk = Symbol("μ$k")
    for i in chain.tree.order
        idx = chain.tree[i,:θ]
        idx in seen ? continue : push!(seen, idx)
        prop = chain.proposals[λk, idx]
        λi, hr1 = prop(chain[λk, idx])
        μi, hr2 = prop(chain[μk, idx])
        p_ = logprior(chain, λk=>(idx, λi), μk=>(idx, μi))  # prior
        d = deepcopy(chain.models[k])
        d[:μ, i] = μi   # NOTE: implementation of setindex! uses node indices!
        d[:λ, i] = λi   # NOTE: implementation of setindex! uses node indices!
        l_ = logpdf!(d, chain.X, i, k)  # likelihood
        # NOTE: loglikelihood from global chain; logprior from cluster!
        mhr = l_ + p_ - chain[:logp] - chain[:logπ] + hr1 + hr2
        if log(rand()) < mhr
            set_L!(chain.X, k)    # update L matrix
            chain.models[k] = d
            chain[λk, idx] = λi
            chain[μk, idx] = μi
            chain[:logπ] = p_
            chain[:logp] = l_
            prop.accepted += 1
        else
            set_Ltmp!(chain.X, k)  # revert Ltmp matrix
        end
        consider_adaptation!(prop, chain[:gen])
    end
end
