# MCMC like in coevol
# first without WGDs

# every branch (node) has a vector of parameters, and vector of moves
# while doing a postorder, moves are randomly chosen from the move vector
# this covers all likelihood involved computations

# the rest is the root prior, and rates prior

# the rates prior should be mixed across families

# we don't track the covariance matrix explicitly, since we can integrate it out

# it would be interesting to allow divergence times to be random of course

struct MvBMPrior
    Σ₀::Matrix{Float64}  # InverseWishart prior covariance matrix
    X₀::Prior            # Prior on root state of X
    πη::Prior            # Prior on η
end

function Distributions.logpdf(d::MvBMPrior, θ::NamedTuple)
    @unpack Σ₀, X₀, πη = d
    @unpack X, η, tree = θ
    X0 = X[1,:]
    bm = Beluga.PhyloMvBM(tree, Σ₀, X0)
    lp = logpdf(bm, X)
    lp += logpdf(πη, η)
    lp += typeof(X₀)<:AbstractVector ? sum(logpdf.(X₀, X0)) : logpdf(X₀, X0)
    lp
end

function Base.rand(d::MvBMPrior, tree)
    @unpack Σ₀, X₀, πη = d
    X0 = typeof(X₀)<:AbstractVector ? rand.(X₀) : rand(X₀)
    bm = Beluga.PhyloMvBM(tree, Σ₀, X0)
    X = rand(bm)
    (tree=tree, X=X, η=rand(πη))
end


# MCMC
# η | X
# X | η
# that's it?
function move_X!(chain, k)
    t = chain.tree
    s = Symbol("X$k")
    proposals = chain.proposals[s]
    m = chain.models[k]
    X = chain[s]
    for node in t.order
        prop = rand(proposals[node])  # pick a random proposal
        m_, X_, hr = move!(deepcopy(X), deepcopy(m), node, prop)
        p_ = logprior(chain, s=>X_)
        l_ = logpdf!(m_, chain.D, k, bs)  # TODO: pass in correct branches
        mhr = l_ + p_ - chain[:logp] - chain[:logπ] + hr
        if log(rand()) < mhr
            set_L!(chain.X, k)
            ...
        else
            set_Ltmp!(chain.X, k)
        end
        consider_adaptation!(prop, chain[:gen])
    end
end

function move!(X, model, node, prop)
    X[node,:], hr = prop(X[node,:])
    θ = branchrates(exp.(X), model.tree)
    update_model!(model, θ, node)
    X, model, hr
end

# update the model *branch rates* when a *node* state has changed
function update_model!(model, θ, node)
    bs = get_affected_branches(model.tree, node)
    for b in bs
        update_rates!(model, θ, b)
    end
end

get_affected_branches(tree, node) =
    !isleaf(t, node) ? [node ; childnodes(tree, node)] : [node]

function update_rates!(model, r, i)
    model[:λ, i] = r[i, 1]
    model[:μ, i] = r[i, 2]
end




using Beluga, Parameters, Distributions
import Beluga.Prior
t, x = Beluga.example_data1()

# prior
ν = 0.01
p = MvBMPrior([ν 0.5ν ; 0.5ν ν], [Normal(), Normal()], Beta(5,2))
θ = rand(p, t.tree)
logpdf(p, θ)

# likelihood
r = Beluga.branchrates(exp.(θ.X), t.tree)
d = DuplicationLoss(t, r[:,1], r[:,2], θ.η, maximum(x))
logpdf(d, x)
