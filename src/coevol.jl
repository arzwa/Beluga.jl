# MCMC like in coevol
# first without WGDs

# every branch (node) has a vector of parameters, and vector of moves
# while doing a postorder, moves are randomly chosen from the move vector
# this covers all likelihood involved computations

# the rest is the root prior, and rates prior

# the rates prior should be mixed across families

# we don't track the covariance matrix explicitly, since we can integrate it out

# it would be interesting to allow divergence times to be random of course

abstract type Model end

# Prior
# =====
struct MvBMPrior <: Model
    Σ₀::Matrix{Float64}  # InverseWishart prior covariance matrix
    X₀::Prior            # Prior on root state of X
    πη::Prior            # Prior on η
    πα::Prior            # Prior on α for mixture
end

function Base.rand(d::MvBMPrior, tree)
    @unpack Σ₀, X₀, πη, πα = d
    X0 = typeof(X₀)<:AbstractVector ? rand.(X₀) : rand(X₀)
    bm = Beluga.PhyloMvBM(tree.tree, Σ₀, X0)
    X = rand(bm)
    w = rand(πα)
    (tree=tree, X=X, η=rand(πη), w=w)
end

function Distributions.logpdf(d::MvBMPrior, θ::NamedTuple)
    @unpack Σ₀, X₀, πη = d
    @unpack X, η, tree = θ
    X0 = X[1,:]
    lp = logpdf(d, X, tree)
    lp += logpdf(πη, η)
    lp += typeof(X₀)<:AbstractVector ? sum(logpdf.(X₀, X0)) : logpdf(X₀, X0)
    lp
end

# marginal probability of the node states under the multivariate process
# marginalized over the covariance matrix as in Lartillot & Poujol.
function Distributions.logpdf(d::MvBMPrior, X, tree)
    @unpack Σ₀ = d
    @unpack Y, J, A, q, n = getpics(X, tree)
    logp_pics(Σ₀, A, J, q, n)
end

# page 733 o Lartillot & Poujol
function getpics(X, tree::SpeciesTree,)
    M = size(X)[2]
    Y = zeros(length(tree)-1, M)
    J = 1.
    A = zeros(M,M)
    for n in tree.order[1:end-1]
        ΔT = parentdist(tree, n)
        Y[n-1,:] = (X[parentnode(tree, n),:] - X[n,:]) / √ΔT
        A += Y[n-1,:]*Y[n-1,:]'
        J *= ΔT
    end
    (Y=Y, J=J^(-M/2), A=A, q=M+1, n=length(tree)-1)
end

# p(Y|Σ₀,q), q = df, n = # of branches
logp_pics(Σ₀, A, J, q, n) = J * det(Σ₀)^(q/2) / det(Σ₀ + A)^((q + n)/2)

function logprior(chain, args...)
    @unpack Ψ, θ, π, K = chain
    state = deepcopy(θ)
    for (k,v) in args
        haskey(state, k) ? state[k] = v : (@warn "Trying to set unexisting var")
    end
    logπ = [logpdf(π, (tree=Ψ, X=state[Symbol("X$k")], η=state[:η])) for k=1:K]
    logπ = sum(logπ) + logpdf(π.πα, state[:w])
    logπ
end


# the chain
# =========
struct MixtureChain{T<:AbstractProfile,M<:Model} <: Chain
    Ψ::SpeciesTree
    D::AbstractArray{T,1}
    θ::State
    m::Array{DuplicationLossWGD,1}
    π::M
    p::Proposals
    t::Vector{State}  # trace
    K::Int64
end

Base.getindex(w::MixtureChain, s::Symbol) = w.θ[s]
Base.getindex(w::MixtureChain, s::Symbol, i::Int64) = w.θ[s][i]
Base.setindex!(w::MixtureChain, x, s::Symbol) = w.θ[s] = x
Base.setindex!(w::MixtureChain, x, s::Symbol, i::Int64) = w.θ[s][i] = x
Base.display(chain::Chain) = display(sort(chain.θ))
Base.show(io::IO, w::Chain) = write(io, "$(typeof(w))($(w.θ))")
Base.getproperty(chain::Chain, s::Symbol) =
    s == :z ? Beluga.assignments(chain.D) : getfield(chain, s)

function MixtureChain(
        D::MPArray, prior::MvBMPrior, t::SpeciesTree, K::Int64, m::Int64)
    s = State(:gen=>0)
    for k=1:K
        θ = rand(prior, t)
        s[Symbol("X$k")] = θ.X
        s[:η] = θ.η
        s[:w] = θ.w
    end
    models = getmodels(s, t, K, m)
    proposals = getproposals(s)
    trace = State[]
    chain = MixtureChain{MixtureProfile,MvBMPrior}(
        t, D, s, models, prior, proposals, trace, K)
    l = logpdf!(chain.m, D, t.order)
    for k=1:K  # compute for each family the DP matrix and lhood for each k
        Beluga.logpdf_allother!(chain.m[k], D, k, t.order)
        Beluga.set_L!(chain.D, k)
    end
    set_logpdf!(chain)
    set_logprior!(chain)
    chain
end

set_logpdf!(chain::Chain) = chain.θ[:logp] = curr_logpdf(chain)
curr_logpdf(chain::Chain) = mapreduce(x->x.l[x.z], +, chain.D)
set_logprior!(chain::Chain) = chain.θ[:logπ] = logprior(chain)


function getmodels(state, tree, K, mmax)
    m = DuplicationLossWGD[]
    for k=1:K
        Xk = state[Symbol("X$k")]
        br = Beluga.branchrates(exp.(Xk), tree.tree)
        mk = DuplicationLoss(tree, br[:,1], br[:,2], state[:η], mmax)
        push!(m, mk)
    end
    m
end

function getproposals(state)
    proposals = Proposals()
    for (k,v) in state
        if k ∈ [:logπ, :logp]
            continue
        elseif typeof(v)<:AbstractMatrix
            proposals[k] = hcat([CoevolProposals() for i=1:size(v)[1]]...)
        elseif k == :η
            proposals[k] = AdaptiveUnitProposal(0.2)
        end
    end
    proposals
end


# MCMC
# ====
function move_X!(chain, k)
    t = chain.Ψ
    s = Symbol("X$k")
    proposals = chain.p[s]
    for node in t.order
        X = chain[s]
        m = chain.m[k]
        bs = get_pbranches(t, node)
        prop = rand(proposals[:,node])  # pick a random proposal
        X_, m_, hr = move!(deepcopy(X), deepcopy(m), node, prop)
        p_ = logprior(chain, s=>X_)  # TODO
        l_ = logpdf!(m_, chain.D, k, bs)
        mhr = l_ + p_ - chain[:logp] - chain[:logπ] + hr
        if log(rand()) < mhr
            set_L!(chain.D, k)
            chain.m[k] = m_
            chain[s] = X_
            chain[:logp] = l_
            chain[:logπ] = p_
            prop.accepted += 1
        else
            set_Ltmp!(chain.D, k)
        end
        consider_adaptation!(prop, chain[:gen])
    end
end

function move_clusterparams!(chain)
    # for computing the MH ratio, we only need to compute the lhood for the
    # currently assigned families of a particular cluster
    # *upon acceptance* of an update we should recompute the logpdf for all
    # families, as we need this lhood in the next iteration of drawing latent
    # assignments.
    # Actually; we probably want to compute the logpdfs for all clusters just
    # once at the end of this function
    for (k, c) in enumerate(chain.m)
        move_X!(chain, k)
        Beluga.logpdf_allother!(c, chain.D, k, c.tree.order)
        # ↑ in the rare case that no param has changed for cluster k this will
        # be wasteful. NOTE: in the constant-rates model this won't be that
        # rare...
        set_L!(chain.D, k)
        # TODO add wgds
    end
end

function _mcmc!(chain, n)
    for i=1:n
        move_latent_assignment!(chain)
        move_clusterparams!(chain)
        chain[:gen] += 1
        log_mcmc(chain)
        trace!(chain)
    end
end

trace!(chain; interval=1) = chain[:gen] % interval == 0 ?
    push!(chain.t, deepcopy(chain.θ)) : nothing

function log_mcmc(chain::MixtureChain; interval=10)
    chain[:gen] % interval != 0 ? (return) : nothing
    println(join(["gen $(chain[:gen])", chain[:logp], chain[:logπ]], ", "))
    for k in 1:chain.K
        println("k=$k/$(chain.K), w = $(round(chain[:w][k], digits=3))")
        l = [round(x, digits=3) for x in chain.m[k].λ]
        m = [round(x, digits=3) for x in chain.m[k].μ]
        n = min(10, length(l))
        println(" • λ: $(join(l[1:n], ", "))")
        println(" • μ: $(join(m[1:n], ", "))")
        println(" • η: $(round(chain.m[k].η, digits=3))")
    end
    println("_"^75)
end


function move!(X, model, node, prop)
    X[node,:], hr = prop(X[node,:])
    θ = Beluga.branchrates(exp.(X), model.tree.tree)
    update_model!(model, θ, node)
    X, model, hr
end

# gibbs sampler for latent assignments (parallel)
function move_latent_assignment!(chain)
    # distributed computation of new latent assignments
    # XXX is this possible, or does this screw up something with the weights?
    # I think it might only screw up the weights in case of the DP mixture?
    # Not sure, it might just be inefficient to alter weights and assignments
    # independently, but there shouldn't be anything wrong with that in theory.
    # I guess we can just put the weights to the fraction currently assigned?
    logweights = log.(chain[:w])
    map((x)->_move_latent_assignment!(x, logweights), chain.D)
    count = counts(chain.D, length(chain.m))
    chain[:w] .= rand(Dirichlet(count + chain.π.πα.alpha))
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
    x.z = i
end


# utilities
# =========
# update the model *branch rates* when a *node* state has changed
function update_model!(model, θ, node)
    bs = get_affected_branches(model.tree, node)
    for b in bs
        update_rates!(model, θ, b)
    end
end

# NOTE: when node states change, the partial recomputation should start from
# the children, as the rates on these branches are affected by a node state
# change!
get_affected_branches(tree, node) = !isleaf(tree, node) ?
    [childnodes(tree, node) ; node] : [node]

get_pbranches(tree, node) = !isleaf(tree, node) ?
    [childnodes(tree, node); tree.pbranches[node]] : tree.pbranches[node]

function update_rates!(model, r, i)
    model[:λ, i] = r[i, 1]
    model[:μ, i] = r[i, 2]
end
