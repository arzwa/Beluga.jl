# MCMC with a DP mixture
abstract type Chain end
abstract type Model end
abstract type Proposal end

const State = Dict{Symbol,Union{Vector{<:Real},<:Real}}
const Proposals = Dict{Symbol,Union{Vector{<:Proposal},<:Proposal}}

mutable struct MixturePhyloBDPChain{T<:Model,V<:DLModel} <: Chain
    tree::SpeciesTree
    state::State
    bdps::Array{V,1}
    prior::T
    proposals::Array{Proposals,1}
    gen::Int64
    trace::Array
end

Base.getindex(c::Chain, s::Symbol) = c.state[s]
Base.getindex(c::Chain, s::Symbol, i::Int64) = c.state[s][i]
Base.setindex!(c::Chain, v, s::Symbol) = c.state[s] = v
Base.setindex!(c::Chain, v, s::Symbol, i::Int64) = c.state[s][i] = v

mutable struct AdaptiveRWProposal <: Proposal
    accepted::Int64
    tuneinterval::Int64
    kernel::Normal
end

AdaptiveRWProposal(p::AdaptiveRWProposal) = AdaptiveRWProposal(0,
    p.tuneinterval, p.kernel)

Base.rand(prop::Proposal) = rand(prop.kernel)
Base.rand(prop::Proposal, n::Int64) = rand(prop.kernel, n)
Base.getindex(spl::Proposals, s::Symbol, i::Int64) = spl[s][i]

# DP mixture model with constant rates and no WGDs.
struct ConstantDPModel{T<:Real} <: Model
    G0::MvLogNormal
    α::T
    ConstantDPModel(d::MvLogNormal, α::T) where T<:Real = new{T}(G0, α)
end

function init(d::ConstantDPModel, t::SpeciesTree, X::Matrix{Int64}, k=3)
    x = rand(d.G0, k)
    mmax = maximum(X)
    z = rand(1:k, size(X)[1])
    s = State(:λ => x[1,:], :μ => x[2,:], :z => z, :K => k, :n =>counts(z,1:k))
    bdps = [DLModel(t, mmax, x[1, i], x[2, i]) for i=1:k]
    prop = [Proposals(:θ => AdaptiveRWProposal(0, 20, Normal(0., 0.05)))
        for i=1:k]
    MixturePhyloBDPChain(t, s, bdps, d, prop, 0, [])
end

function mcmc!(chain::MixturePhyloBDPChain{T,V}, n=1000;
        show_trace=true, show_every=10,) where {T<:ConstantDPModel,V<:DLModel}
    for i=1:n
        operator_switchmode_constantrates_nowgd!(M, chain);
        push!(chain.trace, [chain[:K], chain[:λ], chain[:μ]])
        if i % show_every == 0
            println("Generation $i, K = $(chain[:K]), n = $(chain[:n])")
            for i=1:chain[:K]
                l = round(chain[:λ][i], digits=3)
                m = round(chain[:μ][i], digits=3)
                println("   ($i) λ = $l, μ = $m")
            end
        end
    end
end

function operator_switchmode_constantrates_nowgd!(X, chain, κ=5)
    # changes everything, i.e. z, K, λk, μk (except η)
    # state should contain :λ, :μ, :K, :z, :bdps
    # add κ clusters
    r = rand(chain.prior.G0, κ)
    α = chain.prior.α
    λ = [chain[:λ] ; r[1,:]]
    μ = [chain[:μ] ; r[2,:]]
    K = chain[:K] + κ
    z = chain[:z]
    n = [chain[:n] ; zeros(Int64, κ)]
    L = zeros(size(X)[1])

    # get the DLModel structs once for each cluster (ϵ & W computed once )
    bdps = chain.bdps
    prop = chain.proposals
    bdps = [bdps ; [DLModel(bdps[1], [λ[i], μ[i]]) for i=K-κ+1:K]]
    prop = [prop ; [Proposals(:θ =>AdaptiveRWProposal(prop[1][:θ])) for
        i=K-κ+1:K]]

    for i=1:size(X)[1]   # not possible to parallelize...
        n[z[i]] -= 1
        # XXX should be able to reuse the logpdf at least for the cluster to
        # which the family was assigned in previous iteration
        l = [logpdf(bdps[k], X[i,:]) for k=1:K]
        # pvector: [old clusters ; new clusters]
        pvec = [[n[k]*exp(l[k]) for k=1:K-κ] ; [(α/κ)*exp(l[k]) for k=K-κ+1:K]]
        pvec[isinf.(pvec)] .= 0 ; pvec[isnan.(pvec)] .= 0
        pvec ./= sum(pvec)
        if !isprobvec(pvec)
            @show z pvec l λ μ
        end
        zi = rand(Categorical(pvec))
        n[zi] += 1
        z[i] = zi
        L[i] = l[zi]
    end
    # update cluster params with a MH step, may be accelerated with a
    # scalable MH step or some delayed acceptance thing.
    empty = Int64[]
    for k=1:K
        if n[k] == 0
            push!(empty, k)
            continue
        end
        # should be possible to compute logpdf in parallel, or should for k=1:K
        # be a distributed for loop?
        r = rand(prop[k][:θ], 2)
        λ_ = exp(log(λ[k]) + r[1])
        μ_ = exp(log(μ[k]) + r[2])
        #λ_ < 0. || μ_ < 0. ? continue : nothing
        bdp = DLModel(bdps[k], [λ_, μ_])
        l  = sum(L[z .== k])
        p  = logpdf(chain.prior.G0, [λ[k], μ[k]])
        l_ = logpdf(bdp, X[z .== k, :])
        p_ = logpdf(chain.prior.G0, [λ_, μ_])
        a = l_ + p_ - l - p
        if log(rand()) < a
            bdps[k] = bdp
            λ[k] = λ_
            μ[k] = μ_
            prop[k][:θ].accepted += 1
        end
        if chain.gen % prop[k][:θ].tuneinterval == 0
            adapt!(prop[k][:θ], chain.gen)
        end
    end
    for k in reverse(empty)
        z, bdps, λ, μ, n, prop = decrease(z, k, bdps, λ, μ, n, prop)
    end
    chain[:K] = maximum(z)
    chain[:z] = z
    chain[:λ] = λ
    chain[:μ] = μ
    chain[:n] = n
    chain.bdps = bdps
    chain.proposals = prop
end

function decrease(z, k, args...)
    for i=1:length(z)
        z[i] > k ? z[i] -= 1 : nothing
    end
    a = []
    for i in 1:length(args)
        θ = args[i]
        θ = k > length(θ) ? θ[1:k-1] : [θ[1:k-1] ; θ[k+1:end]]
        push!(a, θ)
    end
    (z, a...)
end

function adapt!(spl::AdaptiveRWProposal, gen::Int64,
        target=0.25, bound=5., δmax=0.25)
    gen == 0 ? (return) : nothing
    δn = min(δmax, 1. /√(gen/spl.tuneinterval))
    α = spl.accepted / spl.tuneinterval
    lσ = α > target ? log(spl.kernel.σ) + δn : log(spl.kernel.σ) - δn
    lσ = abs(lσ) > bound ? sign(lσ) * bound : lσ
    spl.kernel = Normal(0., exp(lσ))
    spl.accepted = 0
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
