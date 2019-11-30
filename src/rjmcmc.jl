# NOTE:
# * update! will be faster than using a deepcopy of the model
# * initial implementation, hardcoded for only two correlated characters
# * should we implement a discrete Gamma mixture? But that's tricky...

abstract type Model end
abstract type RevJumpPrior <: Model end

const Proposals_ = Dict{Int64,Vector{ProposalKernel}}

@with_kw mutable struct RevJumpChain{T<:Real,M<:RevJumpPrior}
    data ::PArray{T}
    model::DLWGD{T}
    prior::M            = CoevolRevJumpPrior()
    props::Proposals_   = Proposals_()
    state::State        = State(:gen=>0, :logp=>NaN, :logπ=>NaN, :k=>0)
    trace::DataFrame    = DataFrame()
end

Base.vec(chain::RevJumpChain) = Vector(chain.trace[end,:])
Base.rand(df::DataFrame) = df[rand(1:size(df)[1]),:]

function init!(chain::RevJumpChain; rjump=(1., 10., 0.01))
    @unpack data, model, prior, state, props = chain
    setstate!(state, model)
    state[:logp] = logpdf!(model, data)
    state[:logπ] = logpdf(prior, model)
    trace!(chain)
    set!(data)
    setprops!(props, model, rjump)
end

function setstate!(state, model)
    K = 0
    for (i, n) in model.nodes
        K += iswgd(n) ? 1 : 0
        for (k, v) in n.x.θ
            k != :t ? state[id(n, k)] = v : nothing
        end
    end
    state[:k] = K
end

function setprops!(props, model, rjump)
    for (i, n) in model.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            props[i] = [AdaptiveUnitProposal(); WgdProposals()]
        elseif isroot(n)
            props[0] = [AdaptiveUnitProposal(),
                AdaptiveUvProposal(
                    kernel=Beta(rjump[1], rjump[2]),
                    bounds=(0.,1.), tuneinterval=10^10, stop=0,
                    move=AdaptiveMCMC.independent),
                DecreaseλProposal(rjump[3], 10^10)]
            # they should be correlated, a large q with a strong decrease in λ
            props[i] = CoevolUnProposals()
        else
            props[i] = CoevolUnProposals()
        end
    end
end

function trace!(chain)
    @unpack state, trace, model = chain
    chain.trace = vcat(trace, DataFrame(;sort(state)...,
        :wgds=>tracewgds(chain)), cols=:union)
end

function update!(state::State, node::ModelNode, args...)
    for s in args
        state[id(node, s)] = node[s]
    end
end


# Coevol-like Prior
# =================
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}

Base.rand(x::Real) = x
logpdf(x::Real, y) = 0.

@with_kw struct CoevolRevJumpPrior <: RevJumpPrior
    Σ₀::Matrix{Float64}  = [500. 0. ; 0. 500.]
    X₀::Prior            = MvNormal([1.,1.])
    πη::Prior            = Beta(3., 0.33)
    πq::Prior            = Beta()
    πK::Prior            = Geometric(0.5)
    @assert isposdef(Σ₀)
end

# one-pass prior computation based on the model
function logpdf(prior::CoevolRevJumpPrior, d::DLWGD)
    @unpack Σ₀, X₀, πη, πq, πK = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(N, M)
    A = zeros(M,M)
    for (i, n) in d.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            p += logpdf(πq, n[:q])  # what about the time? it is also random?
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, log.(n[:λ, :μ]))
        else
            pt = nonwgdparent(n.p)
            Δt = parentdist(n, pt)
            Y[i-1,:] = (log.(n[:λ, :μ]) - log.(pt[:λ, :μ])) / √Δt
            A += Y[i-1,:]*Y[i-1,:]'
            J *= Δt
        end
    end
    p += logp_pics(Σ₀, (Y=Y, J=J^(-M/2), A=A, q=M+1, n=N))
    p += logpdf(πK, k)
    p::Float64  # type stability not entirely optimal
end

# p(Y|Σ₀,q), q = df, n = # of branches
function logp_pics(Σ₀, θ)
    @unpack J, A, q, n = θ
    # in our case the Jacobian is a constant (tree and times are fixed)
    log(J) + (q/2)*log(det(Σ₀)) - ((q + n)/2)*log(det(Σ₀ + A))
end

function Base.rand(prior::CoevolRevJumpPrior, d::DLWGD)
    @unpack Σ₀, X₀, πη, πq, πK = prior
    model = deepcopy(d)
    Σ = rand(InverseWishart(3, Σ₀))
    for n in prewalk(model[1])
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            n[:q] = rand(πq)
        elseif isroot(n)
            n[:η] = rand(πη)
            r = exp.(rand(X₀))
            n[:λ] = r[1]
            n[:μ] = r[2]
        else
            θp = log.(nonwgdparent(n.p)[:λ, :μ])
            t = parentdist(n, nonwgdparent(n.p))
            θ = exp.(rand(MvNormal(θp, Σ*t)))
            n[:λ] = θ[1]
            n[:μ] = θ[2]
        end
    end
    set!(model)
    return (model=model, Σ=Σ)
end


# Independent rates pior
# ======================
# This works for both branch and node rates?
# I guess we can also exploit conjugacy in an independent rates prior for
# branch rates?
@with_kw struct IidRevJumpPrior <: RevJumpPrior
    Σ₀::Matrix{Float64} = [1 0. ; 0. 1]
    X₀::Prior = MvNormal([0. 0.], I)
    πη::Prior = Beta(3., 0.33)
    πq::Prior = Beta()
    πK::Prior = Geometric(0.5)
    @assert isposdef(Σ₀)
end

function logpdf(prior::IidRevJumpPrior, d::DLWGD)
    @unpack Σ₀, X₀, πη, πq, πK = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(N, M)
    A = zeros(M,M)
    X0 = log.(d[1][:λ, :μ])
    for (i, n) in d.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            p += logpdf(πq, n[:q])  # what about the time? it is also random?
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, X0)
        else
            Y[i-1,:] = log.(n[:λ, :μ]) - X0
            A += Y[i-1,:]*Y[i-1,:]'
        end
    end
    p += logp_pics(Σ₀, (Y=Y, J=1., A=A, q=M+1, n=N))
    p += logpdf(πK, k)
    p::Float64  # type stability not entirely optimal
end

struct UpperBoundedGeometric{T<:Real} <: DiscreteUnivariateDistribution
    d::DiscreteNonParametric{Int64,T,UnitRange{Int64},Array{T,1}}
    p::T
    b::Int64
    function UpperBoundedGeometric(p::T, bound::Int64) where T<:Real
        xs = pdf.(Geometric(p), 0:bound)
        ps = xs ./ sum(xs)
        new{T}(DiscreteNonParametric(0:bound, ps), p, bound)
    end
end

Base.rand(d::UpperBoundedGeometric) = rand(d.d)
Distributions.pdf(d::UpperBoundedGeometric, x::Int64) = pdf(d.d, x)
Distributions.logpdf(d::UpperBoundedGeometric, x::Int64) = logpdf(d.d, x)


# MCMC
# =====
function rjmcmc!(chain, n; trace=1, show=10, rjstart=0, rootequal=false)
    for i=1:n
        chain.state[:gen] += 1
        if i > rjstart
            rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        end
        move!(chain, rootequal=rootequal)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, last(chain.trace))
            flush(stdout)
        end
    end
end

function mcmc!(chain, n; trace=1, show=10)
    for i=1:n
        chain.state[:gen] += 1
        move!(chain)
        i % trace == 0 ? trace!(chain) : nothing
        if i % show == 0
            logmcmc(stdout, last(chain.trace))
            flush(stdout)
        end
    end
end

logmcmc(io::IO, df, n=15) = write(io, "", join([@sprintf("%d,%d",df[1:2]...);
    [@sprintf("%6.3f", x) for x in Vector(df[3:n])]], ","), " ⋯\n| ")

function move!(chain; rootequal=false)
    @unpack model, prior = chain
    for n in postwalk(model[1])
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            move_wgdtime!(chain, n)
            move_wgdrates!(chain, n)
        else
            if isroot(n)
                !(typeof(prior.πη)<:Number) ? move_root!(chain, n) : nothing
                rootequal ? move_rootequal!(chain, n) : move_node!(chain, n)
            else
                move_node!(chain, n)
            end
        end
    end
end

function randpos(model)
    l = length(model)
    v = zeros(l)
    for (i, n) in model.nodes
        v[i] = n[:t]
    end
    i = sample(1:l, Weights(v))
    t = rand(Uniform(0., model[i][:t]))
    model[i], t
end

nwgd(model) = length(getwgds(model))
randwgd(model) = model[rand(getwgds(model))]

function getwgds(model)
    wgds = Int64[]; i = maximum(keys(model.nodes))
    while isawgd(model[i])
        iswgd(model[i]) ? push!(wgds, i) : nothing
        i -= 1
    end
    wgds
end

function reindex!(d::Dict{Int64,T}, i::Int64) where T
    for j=i:2:maximum(keys(d))
        d[j-2] = deepcopy(d[j])
        delete!(d, j)
    end
end

function branchrates!(chain)
    @unpack model, trace = chain
    for (i,n) in model.nodes
        if isroot(n) || isawgd(n) ; continue ; end
        l = (trace[id(n, :λ)] .+ trace[id(nonwgdparent(n.p), :λ)]) / 2.
        m = (trace[id(n, :μ)] .+ trace[id(nonwgdparent(n.p), :μ)]) / 2.
        trace[Symbol("l$i")] = l
        trace[Symbol("m$i")] = m
    end
end

function posterior_Σ!(chain::RevJumpChain{Float64,IidRevJumpPrior}, model::DLWGD)
    @unpack Σ₀ = chain.prior
    chain.trace[:var] = NaN
    chain.trace[:cov] = NaN
    for row in eachrow(chain.trace)
        m = model(row)
        @unpack A, q, n = get_scattermat_iid(m)
        Σ = rand(InverseWishart(q + n, Σ₀ + A))
        row[:var] = Σ[1,1]
        row[:cov] = Σ[1,2]
    end
end

function get_scattermat_iid(d::DLWGD)
    N = ne(d); M = 2
    Y = zeros(N, M)
    A = zeros(M, M)
    X0 = log.(d[1][:λ, :μ])
    for (i, n) in d.nodes
        if isawgd(n) || isroot(n)
            continue
        else
            Y[i-1,:] = log.(n[:λ, :μ]) - X0
            A += Y[i-1,:]*Y[i-1,:]'
        end
    end
    (A=A, Y=Y, q=M+1, n=N)
end

#= for each branch trace the WGDs?
As we're mostly interested in questions like, "what is the marginal posterior
probability of 1 WGD on branch x?", It's perhaps best to trace WGDs as a dict-
like structure {branch: (q, t), ...}?
=#
function tracewgds(chain)
    @unpack model = chain
    d = Dict{Int64,Array{Tuple}}()
    for i in getwgds(model)
        n = model.nodes[i]
        c = nonwgdchild(n)
        x = (parentdist(c, n), n[:q])
        haskey(d, c.i) ? push!(d[c.i], x) : d[c.i] = [x]
    end
    d
end


# Custom Proposals
# ================
# Extension of AdaptiveMCMC lib, proposal moves for vectors [q, λ, μ]
WgdProposals(ϵ=[1.0, 1.0, 1.0, 1.0], ti=25) = [AdaptiveUvProposal(
    kernel=Uniform(-e, e), tuneinterval=ti, move=m)
        for (m, e) in zip([wgdrw, wgdrand, wgdiid, wgdqλ], ϵ)]

function wgdrw(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = x .+ rand(k)
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end

function wgdrand(k::AdaptiveUvProposal, x::Vector{Float64})
    i = rand(1:3)
    xp = copy(x)
    xp[i] = x[i] + rand(k)
    i == 1 ? xp[1] = reflect(xp[1], 0., 1.) : nothing
    return xp, 0.
end

function wgdiid(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = x .+ rand(k, 3)
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end

function wgdqλ(k::AdaptiveUvProposal, x::Vector{Float64})
    xp = copy(x)
    r = rand(k)
    xp[1] += r
    xp[2] -= r
    xp[1] = reflect(xp[1], 0., 1.)
    return xp, 0.
end
