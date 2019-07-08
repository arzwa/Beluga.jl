using Distributions
using BirthDeathProcesses
using PhyloTrees

abstract type GeneFamilyCTMC <: DiscreteMultivariateDistribution end

# extensions
Distributions.insupport(d::GeneFamilyCTMC, x::AbstractVector) = all(x .> 0)
PhyloTrees.isleaf(d::GeneFamilyCTMC, n::Int64) = isleaf(d.tree, n)
PhyloTrees.childnodes(d::GeneFamilyCTMC, n::Int64) = childnodes(d.tree, n)
PhyloTrees.parentdist(d::GeneFamilyCTMC, n::Int64) = parentdist(d.tree, n)
Base.length(tree::Arboreal) = length(tree.tree.nodes)

# map from leaves to consecutive indices
leafmap(tree::Arboreal) = Dict(k=>i for (i,(k,v)) in enumerate(tree.leaves))

# Dl model (branch-wise rates by default)
struct DLModel{T<:Real,Ψ<:Arboreal} <: GeneFamilyCTMC
    tree::Ψ
    porder::Array{Int64,1}
    leafmap::Dict{Int64,Int64}
    b::Array{LinearBDP,1}
    ϵ::Array{T}
    η::T

    function DLModel(tree, λ::Array{T}, μ::Array{T}, η::T=0.9) where T<:Real
        o = postorder(tree)
        l = leafmap(tree)
        b = [LinearBDP(λ[i], μ[i]) for i in o]
        d = new{T,typeof(tree)}(tree, o, l, b, similar(o), η)
        get_ϵ!(d)
        return d
    end
end

DLModel(tree, λ, μ, η=0.9) = DLModel(tree, promote(λ, μ, η)...)
DLModel(tree, λ::T, μ::T, η::T=0.9) where T<:Real = DLModel(tree, repeat([λ],
    length(tree)), repeat([μ], length(tree)))

# helpers for DLModel
Base.length(d::DLModel) = length(d.porder)
Base.show(io::IO, d::DLModel) = show(io, d, (:leafmap, :porder))
Base.getindex(d::DLModel, i::Int64, s::Symbol) = getfield(d.b[i], s)
Base.getindex(d::DLModel, i::Int64) = d.b[i]
Base.getindex(d::DLModel, s::Symbol) = [getfield(d.b[i], s) for i in d.porder]
Base.setindex!(d::DLModel{T}, v::T, i::Int64, s::Symbol) where T<:Real =
    getfield(d, s)[i] = v
leafmap(d::DLModel, e::Int64) = d.leafmap[e]

# extinction probabilities
function get_ϵ!(d::DLModel)
    for e in d.porder
        d[e,:ϵ] = isleaf(d, e) ? 0.0 :
            prod([ep(d.b[c], parentdist(d,c),d.ϵ[c]) for c in childnodes(d, e)])
    end
end

# NOTE implement for matrix as well s.t. we only traverse the tree once for
# large data sets
function get_M(d::DLModel, x::AbstractVector)
    M = zeros(Int64, length(d.porder))
    for n in d.porder
        M[n] = isleaf(d, n) ? x[d[n]] : sum([M[c] for c in childnodes(d, n)])
    end
    return M
end

# dpmatrix, intuitive
function pgm(d::DLModel, x::Vector{Int64}, max=50)
    P = zeros(length(d), max+1)
    for e in d.porder
        if isleaf(d, e)
            P[e, leafmap(d[e])+1] = 1.0
        else
            children = childnodes(d, e)
            for i = 0:max
                p = 1.
                for c in children
                    p_ = 0.
                    for j in 0:length(P[c, :])-1
                        p_ += tp(d.b[c], i, j, parentdist(d, c)) * P[c, j+1]
                    end
                    p *= p_
                end
                P[e, i+1] = p
            end
        end
    end
    return P
end

# Csuros & Miklos algorithm
# maybe put a hard maximum on the number of genes as well, to prevent overflows
# etc. for large species trees...
# FIXME not correct
function cm_mbe(d::DLModel, x::T, M::T, hardmax=Inf) where
        {T<:AbstractVector{Int64}}
    mx = min(maximum(M), hardmax)
    L = zeros(length(d.tree), mx+1)
    for e in d.porder
        if isleaf(d, e)
            L[e, x[d[e]]+1] = 1.0
        else
            f, g = childnodes(d, e)
            for i=0:M[e]
                L[e, i+1] = _conditional_lhood(L, e, f, g, d, i, M[f], M[g])
            end
        end
    end
    return L
end

#=function _conditional_lhood(L, e, f, g, d, n, Mf, Mg)
    p = 0.
    tf = parentdist(d, f)
    for t=0:Mf
        s = n-t
        s > Mg ? continue : nothing
        Bf0t = sum([L[f, m+1] * tp(d.b[f], t, m, tf) for m=0:Mf])
        ... =#


function csuros_miklos(d::DLModel, x::T, M::T, hardmax=Inf) where
        {T<:AbstractVector{Int64}}
    mx = Int(min(maximum(M), hardmax))
    L = zeros(length(d.tree), mx+1)
    for e in d.porder
        if isleaf(d, e)
            L[e, leafmap(d[e])+1] = 1.0
        else
            children = childnodes(d, e)
            _M = [M[c] for c in children]
            _m = cumsum([0 ; _M])
            _ϵ = cumprod([1.; [d.ϵ[c] for c in children]])
            B = zeros(length(children), _m[end]+1, mx+1)
            A = zeros(length(children), _m[end]+1)
            for i = 1:length(children)
                c = children[i]
                δ = parentdist(d, c)
                for t = 0:_m[i]  # this is 0 ... M[i-1]
                    for s = 0:_M[i]  # this is 0 ... Mi
                        if t == 0
                            B[i, 1, s+1] = sum([L[c, m+1] *
                                tp(d.b[c], s, m, δ) for m=0:_M[i]])
                        elseif s == _M[i]
                            B[i, t+1, s+1] = B[i,t,s+1] * d.ϵ[c]
                        else
                            B[i, t+1, s+1] = B[i,t,s+2] + d.ϵ[c]*B[i, t, s+1]
                        end
                    end
                end
                if i == 1
                    for n = 0:_m[i+1]  # this is 0 ... M[i]
                        A[1, n+1] = (1. - _ϵ[2])^(-n)*B[i, 1, n+1]
                    end
                else
                    for t = 0:_m[i]  # this is 0 ... M[i-1]
                        for s = 0:_M[i]  # this is 0 ... Mi
                            A[i, t+1] += pdf(Binomial(t, _ϵ[i]), s) *
                                A[i-1, t+1]*B[i, t+1, s+1]
                        end
                    end
                    for n = 0:_m[i+1]  # this is 0 ... M[i]
                        A[i, n+1] *= (1. - _ϵ[i+1])^(-n)
                    end
                end
            end
            for n=0:M[e]
                L[e, n+1] = A[end, n+1]
            end
        end
    end
    L
end

csuros_miklos(d, x, M)

function _wstar(d::DLModel, e::Int64, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    p = getϕ(d[e], parentdist(d, e))
    q = getψ(d[e], parentdist(d, e))
    w = zeros(mmax+1, mmax+1)
    w[1,1] = 1.
    for m=1:mmax, n=1:m
        w[m+1, n+1] = n == m ? (1. - p)*(1. - q)*w[m, n] :
            q*w[m, n+1] + (1. - p)*(1. - q)*w[m, n]
    end
    return w
end

function csuros_miklos2(d::DLModel, x, M, hardmax=Inf)
    # Based on Csuros & Miklos (2009) and Cecile Ane (2013)
    mx = Int(min(maximum(M), hardmax))
    L = zeros(length(d.tree), mx+1)

end



function _logpdf(d::DLModel, x::T, M::T) where {T<:AbstractVector{Int64}}
    L = csuros_miklos(d, x, M)
    root = d.porder[end]
    e = d.ϵ[root]
    η = d.η
    p = 0.
    for i in 2:length(L[root, :])
        ξ = (1. - e)^(i-1)*η*(1. - η)^(i-2)/(1. - (1. - η)*e)^i
        @show ξ
        p += L[root,i] * ξ
    end
    # conditioning?
    p
end

function oib(d)
    e = d.porder[end]
    f, g  = childnodes(d.tree, e)
    ε_root = geometric_extinctionp(d.ϵ[e], d.η)
    ε_left = geometric_extinctionp(d.ϵ[f], d.η)
    ε_rght = geometric_extinctionp(d.ϵ[g], d.η)
    nf = log(1. - ε_left - ε_rght + ε_root)
    ε_root, ε_left, ε_rght, nf
end

geometric_extinctionp(ε, η) = η * ε / (1. - (1. - η)*ε)
