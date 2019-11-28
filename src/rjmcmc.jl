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

function init!(chain::RevJumpChain; rjump=(1., 5., 0.5))
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
    X₀::Prior = Normal()
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
            p += sum(logpdf.(X₀, X0))
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
function rjmcmc!(chain, n; trace=1, show=10, rjstart=0)
    for i=1:n
        chain.state[:gen] += 1
        if i > rjstart
            rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        end
        move!(chain)
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

function move!(chain)
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
                 move_rootequal!(chain, n)
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
        t = (parentdist(c, n), n[:q])
        haskey(d, c.i) ? push!(d[c.i], t) : d[c.i] = [t]
    end
    d
end

function get_wgdtrace(chain)
    tr = Dict{Int64,Dict{Int64,DataFrame}}()
    for d in chain.trace[:wgds]
        for (k,v) in d
            n = length(v)
            if !haskey(tr, k)
                tr[k] = Dict{Int64,DataFrame}()
            end
            if !haskey(tr[k], n)
                tr[k][n] = DataFrame(
                    [Symbol("q$i")=>Float64[] for i=1:n]...,
                    [Symbol("t$i")=>Float64[] for i=1:n]...)
            end
            sort!(v)
            t = [x[1] for x in v]
            q = [x[2] for x in v]
            push!(tr[k][n], [q ; t])
        end
    end
    tr
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


# Bayes factors
# =============
# Prior probabilities for # of WGDs on each branch
# function kbranch_prior(n, k, T, prior, kmax=100)
#     # NOTE approximate: kmax is the truncation bound for the # of WGDs in total
#     # k is the maximal number of WGDs on the branch of interest for which the
#     # prior should be computed
#     te = parentdist(n, nonwgdparent(n.p))
#     p = Dict(j=>0. for j=0:k)
#     for j=0:k
#         q = (te/T)^j
#         for i=0:kmax
#             p[j] += binomial(i, j)*q*(1. - (te/T))^(i-j)*pdf(prior, i)
#         end
#     end
#     p
# end

# closed form
function kbranch_prior(k, t, T, p)
    a = t/T
    b = 1. - t/T
    q = 1. - p
    p * (a*q)^k * (1. - b*q)^(-k-1)
end

function branch_bayesfactors(chain, burnin::Int64=1000)
    @unpack trace, model, prior = chain
    trace_ = trace[burnin+1:end,:]
    df = branch_bayesfactors(trace_, model, prior.πK.p)
    show_bayesfactors(df)
    df
end

"""
    branch_bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)

Compute Bayes Factors for all branch WGD configurations. Returns a data frame
that is more or less self-explanatory.
"""
function branch_bayesfactors(trace::DataFrame, model::DLWGD, p::Float64)
    T = treelength(model)
    df = DataFrame()
    for (i,n) in sort(model.nodes)
        if isawgd(n); break; end
        if !(Symbol("k$i") in names(trace)); continue; end
        fmap = freqmap(trace[Symbol("k$i")])
        kmax = maximum(keys(fmap))
        te = parentdist(n, nonwgdparent(n.p))
        ps = [haskey(fmap, i) ? fmap[i] : 0. for i in 0:kmax]
        πs = kbranch_prior.(0:kmax, te, T, p)
        cl = Beluga.clade(model, n)
        df_ = DataFrame(:branch=>i,
            :clade=>repeat([cl], kmax+1),
            :k=>0:kmax, :eval=>"",
            :p1=>ps, :p0=>[NaN ; ps[1:end-1]],
            :π1=>πs, :π0=>[NaN ; πs[1:end-1]],
            :t=>te, :tfrac=>te/T)
        df = vcat(df, df_)
    end
    # compute BFs
    df[:K] = (df[:p1] ./ df[:p0]) ./ (df[:π1] ./ df[:π0])
    df[:log10K] = log10.(df[:K])
    for x in [0.5, 1., 2.]
        df[:eval][df[:log10K] .> x] .*= "*"
    end
    df
end

function show_bayesfactors(df::DataFrame)
    for d in groupby(df, :branch)
        @printf "🌲 %2d: " d[:branch][1]
        println("(", join(string.(d[:clade][1]), ","), ")")
        for (i,r) in enumerate(eachrow(d))
            isnan(r[:p0]) ? continue : nothing
            @printf "[%2d vs. %2d] " i-1 i-2
            @printf "K = (%.2f/%.2f) ÷ (%.2f/%.2f) " r[:p1] r[:p0] r[:π1] r[:π0]
            @printf "= %8.3f [log₁₀(K) = %8.3f] %s\n" r[:K] r[:log10K] r[:eval]
        end
        println("_"^78)
    end
end

# MCMCChains interface, diagnostics etc.
function MCMCChains.Chains(trace::DataFrame, burnin=1000)
    df = deletecols(trace, :wgds)
    if size(df)[1] < burnin
        @error "Trace not long enough to discard $burnin iterations as burn-in"
    end
    X = reshape(Matrix(df), (size(df)...,1))[burnin+1:end, 2:end, :]
    return Chains(X, [string(x) for x in names(df)][2:end])
end

# get an MCMCChains chain (gives diagnostics etc.)
MCMCChains.Chains(c::RevJumpChain, burnin=1000) = Chains(c.trace, burnin)
