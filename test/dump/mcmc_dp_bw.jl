# MCMC with a DP mixture
abstract type Chain end
abstract type Model end

const State = Dict{Symbol,Union{Array{<:Real},<:Real}}

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

# DP mixture model with constant rates and no WGDs.
struct ConstantDPModel{T<:Real} <: Model
    G0::MvLogNormal
    α::T
    ConstantDPModel(d::MvLogNormal, α::T) where T<:Real = new{T}(G0, α)
end

function Base.rand(m::ConstantDPModel)
    x = rand(m.G0)
    n = length(x)÷2
    return x[1:n], x[n+1:end]
end

function Base.rand(m::ConstantDPModel, i::Int)
    x = rand(m.G0, i)
    n = size(x)[1]÷2
    return x[1:n,:], x[n+1:end,:]
end

function init(d::ConstantDPModel, t::SpeciesTree, X::Matrix{Int64}, k=3)
    x = rand(d.G0, k)
    mmax = maximum(X)
    z = rand(1:k, size(X)[1])
    n = length(t)
    λ = x[1:n,:] ; μ = x[n+1:end,:]
    s = State(:λ =>λ, :μ =>μ, :z =>z, :K=>k, :n=>counts(z,1:k), :logp=>-Inf)
    bdps = [DLModel(t, mmax, λ[:,i], μ[:,i], 0.8) for i=1:k]
    prop = [Proposals(:θ => [AdaptiveUnProposal() for j=1:n]) for i=1:k]
    MixturePhyloBDPChain(t, s, bdps, d, prop, 0, [])
end

function mcmc!(chain::MixturePhyloBDPChain{T,V}, n=1000;
        show_trace=true, show_every=10,) where {T<:ConstantDPModel,V<:DLModel}
    for i=1:n
        chain.gen += 1
        operator_switchmode_constantrates_nowgd!(M, chain);
        push!(chain.trace, [chain[:K], chain[:λ], chain[:μ]])
        if i % show_every == 0
            print("Generation $i, K = $(chain[:K]), n = $(chain[:n]), ")
            print("logpdf = $(round(chain[:logp]))\n")
            for i=1:chain[:K]
                l = join(round.(chain[:λ][1:3,i], digits=3), ",")
                m = join(round.(chain[:μ][1:3,i], digits=3), ",")
                println("   ($i) λ = $l,⋯ ; μ = $m,⋯")
            end
        end
    end
end

function operator_switchmode_constantrates_nowgd!(X, chain, κ=5)
    # changes everything, i.e. z, K, λk, μk (except η)
    # state should contain :λ, :μ, :K, :z, :bdps
    # add κ clusters
    l, m = rand(chain.prior, κ)
    α = chain.prior.α
    λ = [chain[:λ]  l]
    μ = [chain[:μ]  m]
    K = chain[:K] + κ
    z = chain[:z]
    n = [chain[:n] ; zeros(Int64, κ)]
    № = length(chain.tree)
    L = zeros(size(X)[1])

    # get the DLModel structs once for each cluster (ϵ & W computed once )
    bdps = chain.bdps
    prop = chain.proposals
    bdps = [bdps ; [DLModel(bdps[1], [λ[:,i] ; μ[:,i]]) for i=K-κ+1:K]]
    prop = [prop ; [Proposals(:θ =>[AdaptiveUnProposal() for j=1:№])
        for i=K-κ+1:K]]

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
    chain[:logp] = 0.
    for k=1:K
        if n[k] == 0
            push!(empty, k)
            continue
        end
        # should be possible to compute logpdf in parallel, or should for k=1:K
        # be a distributed for loop?
        for i in postorder(chain.tree)
            r = prop[k][:θ][i]
            λik, a = AdaptiveMCMC.scale(r, λ[i,k])
            μik, b = AdaptiveMCMC.scale(r, μ[i,k])
            λk_ = deepcopy(λ[:,k])
            μk_ = deepcopy(μ[:,k])
            λk_[i] = λik
            μk_[i] = μik
            bdp = DLModel(bdps[k], [λk_ ; μk_])
            l  = sum(L[z .== k])
            p  = logpdf(chain.prior.G0, [λ[:,k] ; μ[:,k]])
            l_ = logpdf(bdp, X[z .== k, :])
            p_ = logpdf(chain.prior.G0, [λk_ ; μk_])
            mhr = l_ + p_ - l - p + a + b
            if log(rand()) < mhr
                bdps[k] = bdp
                λ[i,k] = λik
                μ[i,k] = μik
                prop[k][:θ][i].accepted += 1
                chain[:logp] += l_ + p_
            else
                chain[:logp] += l + p
            end
            if chain.gen % prop[k][:θ][i].tuneinterval == 0
                AdaptiveMCMC.adapt!(prop[k][:θ][i], chain.gen, 0.36)
            end
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
        if length(size(θ)) == 1
            θ = k > length(θ) ? θ[1:k-1] : [θ[1:k-1] ; θ[k+1:end]]
        else
            θ = k > size(θ)[2] ? θ[:,1:k-1] : [θ[:,1:k-1]  θ[:,k+1:end]]
        end
        push!(a, θ)
    end
    (z, a...)
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
=

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
=#
