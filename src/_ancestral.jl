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

using StatsBase

function normlogp(x)
    p = exp.(x .- maximum(x))
    p = p ./ sum(sort(p))
end

# must be fast! should be optimized for multiple sims (for one family)
function Base.rand(d::DLWGD{T}, profile; hardmax=50) where T<:Real
    mmax  = size(d[1].x.W)[1]
    Lroot = Beluga.root_vector(L[:,1], d[1])
    proot = normlogp(Lroot[2:end])
    X     = zeros(Int64, length(x))
    X[1]  = sample(1:length(proot), Weights(proot))
    function walk(n)
        # sample from P(Xi|D,Xj) = [1/P(D'|Xj)]P(Xi|Xj) Σ_y P(D'|Yi=y) P(Yi=yi|Xi)
        # i.e. compute that vector for many Xi and sample from it, in preorder
        for c in n.c
            # sample
            # P(Xi|Xj) Σ_y P(D'|Yi=y) P(Yi=yi|Xi)
            Xj = X[n.i]


            walk(c)
        end
    end
end

function pvec(n, xp, profile; hardmax=50, tol=1e-6)
    λ, μ = Beluga.getλμ(n, n.p)
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
