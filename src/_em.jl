# # ancestral states _____________________________________________________________
# # are there optimizations possible when considering full matrix at once instead
# # of single vector?
# function ancestral_states(d::DLWGD{T}, L::Matrix, x::Vector) where T<:Real
#     mmax = size(d[1].x.W)[1]
#     Lroot = root_vector(L[:,1], d[1])
#     nroot = sample(1:length(Lroot), Weights(Lroot)) + 1
#     pvecs = Dict{Int64,Vector{Float64}}()
#     for (i,n) in d.nodes
#         if isroot(n) ; continue ; end
#         p = Float64[]
#         m = 0
#         while sum(p) < 1-1e-5 && m < mmax/2
#             lp = postp_ancestral_state(n, L, m, x)
#             push!(p, exp(lp))
#             m += 1
#         end
#         @show sum(p)
#         pvecs[i] = p ./ sum(p)
#     end
#     pvecs
# end
#
# function postp_ancestral_state(n::ModelNode, L::Matrix, m::Int64, x::Vector)
#     ep = gete(n, 1)
#     mmax = min(m, x[1])
#     inner = [exp(L[i+1,n.i])*binomial(m, i)*ep^(m-i)*(1. - ep)^i for i=0:mmax]
#     lp = truncated_logpdf(n, L, m, x)
#     log(sum(inner))+lp
# end
#
# function truncated_logpdf(n::ModelNode{T}, L::Matrix{T},
#         m::Int64, x::Vector) where T
#     mmax = size(L)[1]
#     y = reprofile!(copy(x), m, n)
#     ℓ = y[1]+1 > mmax ? [L ; minfs(T, y[1]+1-mmax, size(L)[2])] : L
#     # y[1]+1 > size(n.x.W)[1] ? extend!(n, y[1]+1) : nothing
#     csuros_miklos!(ℓ, n, y, true)  # treat `n` as a leaf
#     logpdf!(ℓ, n.p, y)
# end
#
# function reprofile!(x::Vector, m::Int64, n::ModelNode)
#     x[n.i] = m
#     n = n.p
#     while !(isnothing(n))
#         x[n.i] = sum([x[c.i] for c in n.c])
#         n = n.p
#     end
#     x
# end
#
# function extend!(d::DLWGD, m::Int64)
#     for n in postwalk(d[1])
#         n.x.W = zeros(m,m)
#         set!(n)
#     end
# end

using StatsBase, Parameters, QuadGK

function normlogp(x)
    p = exp.(x .- maximum(x))
    p = p ./ sum(sort(p))
end

function randreduce(d, profile, n=100)
    A = rand(d, profile, n)
    X = reduce(+, A, dims=1) ./ size(A)[1]
    X[1,:]
end

function Base.rand(d::DLWGD{T}, profile; hardmax=50) where T<:Real
    @unpack Lp, xp = profile
    mmax  = size(d[1].x.W)[1]
    Lroot = Beluga.root_vector(Lp[:,1], d[1])
    proot = normlogp(Lroot[2:end])
    X     = zeros(Int64, length(xp))
    X[1]  = sample(1:length(proot), Weights(proot))
    function walk(n)
        # sample from P(Xi|D,Xj) = [1/P(D'|Xj)]P(Xi|Xj) Σ_y P(D'|Yi=y) P(Yi=yi|Xi)
        # i.e. compute that vector for many Xi and sample from it, in preorder
        Beluga.isleaf(n) ? (return) : nothing
        for c in n.c
            p = pvec(c, X[n.i], profile, hardmax=hardmax)
            X[c.i] = sample(0:length(p)-1, Weights(p))
            walk(c)
        end
    end
    walk(d[1])
    X
end

# optimized for multiple sims for the same family, such that the pvector for a
# particular node is only computed once for every unique parent state.
function Base.rand(d::DLWGD{T}, profile, N::Int64; hardmax=50) where T<:Real
    @unpack Lp, xp = profile
    mmax  = size(d[1].x.W)[1]
    Lroot = Beluga.root_vector(Lp[:,1], d[1])
    proot = normlogp(Lroot[2:end])
    X     = zeros(Int64, N, length(xp))
    X[:,1]= sort(sample(1:length(proot), Weights(proot), N))
    function walk(n, x, m, i)  # n: node, x: node state, m: # reps with that state
        Beluga.isleaf(n) ? (return) : nothing
        for c in n.c
            p = pvec(c, x, profile, hardmax=hardmax)
            C = sort(sample(0:length(p)-1, Weights(p), m))
            X[i:i+m-1,c.i] = C
            j = i
            for (xc, count) in countmap(C)
                walk(c, xc, count, j)
                j += count
            end
        end
    end
    i = 1
    for (x, count) in countmap(X[:,1])
        walk(d[1], x, count, i)
        i += count
    end
    X
end

function pvec(n, xp, profile; hardmax=50, tol=1e-6)
    λ, μ = Beluga.getλμ(n.p, n)
    t = n[:t]
    e = Beluga.gete(n, 2)
    L = profile.Lp
    ymax = profile.x[n.i]
    pvec = Float64[]
    diff = 0
    x = 0
    imax = 0
    while diff < -log(tol) && x <= hardmax
        p = Beluga.tp(xp, x, t, λ, μ)
        inner = [exp(L[y+1,n.i])*binomial(x,y)*e^(x-y)*(1. -e)^y for y=0:min(x,ymax)]
        push!(pvec, log(p) + log(sum(inner)))
        imax > 0 ? diff = maximum(pvec[imax:end]) - minimum(pvec[imax:end]) : nothing
        imax = x > 0 && pvec[end-1] < pvec[end] ? x+1 : imax
        x += 1
    end
    normlogp(pvec)
end

function expected_summarystats(d::DLWGD, P::PArray, nsims=3)
    udt = zeros(3, Beluga.ne(d)+1)
    for p in P  # should be in parallel
        for i=1:nsims
            udt += expected_summarystats(d, rand(d, p))
        end
    end
    udt
end

function expected_summarystats(d::DLWGD, x)
    UDT = zeros(3, length(x))
    for (i,n) in d.nodes
        i == 1 ? continue : nothing
        a, b = x[n.p.i], x[n.i]
        λ, μ = Beluga.getλμ(n.p, n)
        t = n[:t]
        UDT[:, n.i] .= expected_summarystats(a, b, λ, μ, t)
    end
    UDT
end

function expected_summarystats(a, b, λ, μ, t, kmax=max(2*max(a,b), 10))
    pab = Beluga.tp(a, b, t, λ, μ)
    U, D, T = 0., 0., 0.
    for k=1:kmax
        fU = (τ)->Beluga.tp(a, k, τ, λ, μ) * k*λ * Beluga.tp(k+1, b, t-τ, λ, μ)
        fD = (τ)->Beluga.tp(a, k, τ, λ, μ) * k*μ * Beluga.tp(k-1, b, t-τ, λ, μ)
        fT = (τ)->Beluga.tp(a, k, τ, λ, μ) * Beluga.tp(k, b, t-τ, λ, μ)
        U += quadgk(fU, 0., t, rtol=1e-6)[1]
        D += quadgk(fD, 0., t, rtol=1e-6)[1]
        T += k*quadgk(fT, 0., t, rtol=1e-6)[1]
    end
    [U/pab, D/pab, T/pab]
end

function _em!(d::DLWGD, pa::PArray, nsims=3, α=1.)
    logpdf!(d, pa)
    udt = expected_summarystats(d, pa, nsims)
    λ = (udt[1,:] .+ α) ./ (udt[3,:] .+ α)
    μ = (udt[2,:] .+ α) ./ (udt[3,:] .+ α)
    λ[1] = μ[1] = -1.0
    Beluga.setrates!(d, permutedims([λ μ]))
end


# ______________________________________________________________________________
using Plots

begin
    nw = open("test/data/dicots/plants2.nw", "r") do f ; readline(f); end
    df = CSV.read("test/data/dicots/dicots-f01-25.csv", delim=",")
    d, p = DLWGD(nw, df, 2*rand(), 2*rand(), 0.9)
    logpdf!(d, p)
end

function em!(d, p, n; nsims=3, α=1.)
    r = Beluga.getrates(d)
    l = [r[1,2:end]]
    for i=1:n
        _em!(d, p, nsims, α)
        r = Beluga.getrates(d)
        push!(l, r[1,2:end])
        R = log.(hcat(l...))
        pl = plot(R[1,:], linewidth=2)
        plot!(R[2,:], linewidth=2)
        plot!(R[5,:], linewidth=2)
        display(pl)
    end
    d
end

em!(d,p,1000, nsims=3, α=0.5)

# # mcmc logλ, 2500 gens
# 0.9475193397570026
# -0.41656689681084513
# 0.6864651685293978
# -0.16576272500282682
# 1.72726054015759
# 0.7014350235617175
# 1.2381177159058383
# 0.45956268912452825
# 0.5622803638002509
# 0.38442062985051084
# -0.1730132524703185
# 0.24473869003139492
# 2.0254912349668857
# 0.14549459892096392
# 0.625068165777596
# 1.0367856748587374
