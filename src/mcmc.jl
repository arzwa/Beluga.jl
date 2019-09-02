
struct PhyloBDPChain{T<:Model}
    tree::SpeciesTree
    state::State
    prior::T
    samplers::Samplers
    gen::Int64
    df::DataFrame
end

#= example state:
    η  => float
    ν  => float
    λ  => [rate for each branch of T, 0 is mean]
    μ  => [rate for each branch of T, 0 is mean]
    q  => [retention rate for each WGD]
    K  => number of clusters
    z  => [cluster assignment for each family]
    rθ => [relative dup-loss rate for each cluster]
    rq => [relative retention rate for each cluster]

The priors are like in Whale for λ, μ, q, η and ν. K and z are governed by the
DP mixture with as base measure two Gamma distributions, one for the rθ and one
for the rq.

The main problem is to still have decent parallelism. The DP requires a gibbs
sweep over families, which will probably be prohibitive...

The number of genes at the root should perhaps also be modeled in an explicit
family specific way? Or do we still integrate it out? (the latter is easier and
faster so I would keep it)
=#

# sketch of the algorithm
function operator_switchmode!(X, chain)
    # changes z, K, rθ and rq (when relevant)
    # add κ clusters
    ...
    K = chain[:K] + κ
    n = count(chain[:z], 1:K)  # perhaps z should be in x
    for i=1:size(X)[1]   # not possible to parallelize...
        # can this be on a random subset in each generation? would it help?
        # this should also store the logpdf for each family for its chosen
        # cluster assignment

        # pick existing or new cluster assignment

        chain[:z, i] = z
        chain[:lhood, i] = l[z]
    end
    # remove empty clusters
    ...
    chain[:K] =  ...
    for k=1:chain[:K]
        # should be possible to compute logpdf in parallel, or should for k=1:K
        # be a distributed for loop?

        # update cluster params with a MH step, may be accelerated with a
        # scalable MH step or soe delayed acceptance thing.
    end
end

function operator_brates!(X, chain)
    # changes λ and μ
    for e in chain.tree.order
        # use partial recomputation and other speed-up tricks
    end
end

function operator_root!(X, chain)
    # changes η, only changes the conditional likelihood at the root
end

function operator_ν!(X, chain)
    # changes ν, does not change the likelihood, only the rates prior
end
