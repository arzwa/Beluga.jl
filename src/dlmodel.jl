using Distributions
import Distributions: _logpdf, _rand!, insupport
#using BirthDeathProcesses
using PhyloTrees
import PhyloTrees: isleaf, childnodes, parentdist

abstract type GeneFamilyCTMC <: DiscreteMultivariateDistribution end

insupport(d::GeneFamilyCTMC, x::AbstractVector) = all(x .> 0)
isleaf(d::GeneFamilyCTMC, n::Int64) = isleaf(d.tree, n)
childnodes(d::GeneFamilyCTMC, n::Int64) = childnodes(d.tree, n)
parentdist(d::GeneFamilyCTMC, n::Int64) = parentdist(d.tree, n)

function leafmap(tree::Arboreal)
    d = Dict{Int64,Int64}()
    for (i,(k, v)) in enumerate(tree.leaves)
        d[k] = i
    end
    return d
end

# Dl model (constant/branchwise rates)
# if BirthDeathProcess.jl is nice, it would be good to have instead of λ
# and μ arrays, an array of BDP distributions
struct DLModel{T<:Real,U<:Arboreal} <: GeneFamilyCTMC
    tree::U
    λ::Union{T,Array{T}}
    μ::Union{T,Array{T}}
    ϵ::Array{T}
    porder::Array{Int64}
    leafmap::Dict{Int64,Int64}

    function DLModel(tree, λ::T, μ::T) where T<:Real
        o = postorder(tree)
        l = leafmap(tree)
        d = new{T,typeof(tree)}(tree, λ, μ, similar(o), o, l)
        get_ϵ!(d)
        return d
    end
end

DLModel(tree, λ, μ) = DLModel(tree, promote(λ, μ)..., postorder(tree))

# helpers for DLModel
Base.getindex(d::DLModel, i::Int64, s::Symbol) =
    typeof(getfield(d, s)) <: AbstractArray ? getfield(d, s)[i] : getfield(d, s)
Base.getindex(d::DLModel, i::Int64) = d.leafmap[i]

Base.setindex!(d::DLModel{T}, v::T, i::Int64, s::Symbol) where T<:Real =
    getfield(d, s)[i] = v

Base.length(d::DLModel) = length(d.porder)
constant(d::DLModel) = typeof(d.λ)<:Real

# Distributions interface
# _logpdf following Csuros & Miklos
function _logpdf(d::DLModel, x::T, M::T) where T::AbstractVector{Int64}
    L = initmatrix(d, maximum(M)+1)
    for e in d.porder
        if isleaf(d, e)
            L[e, x[d[e]]+1] = 1.0
        else
            children = childnodes(d, e)
            _M = [M[c] for c in children]
            _m = cumsum([0 ; _M])
            _ϵ = cumprod([1.; [d[c, :ϵ] for c in children]])
            B = zeros(length(children), _m[end], maximum(_M))
            for i = 1:length(children)
                c = children[i]
                δ = parentdist(d, c)
                for t = 0:_M[i]
                    for s = 0:M[i]
                        if t == 0
                            B[i, t+1, s+1] = sum([L[c, m] *
                                p(m, s, d[c, :λ], d[c, :μ], δ)] for m=0:M[i]])
                        end
                        if s == M[i]
                            B[i, t+1, s+1] = B[i, t, M[i]+1] * d[c, :ϵ]
                        else
                            B[i, t+1, s+1] = B[i,t,s+2] + d[c, :ϵ]*B[i, t, s+1]
                        end
                    end
                end
            end
        end
    end
end


# DLModel implementation
dpmatrix(d::DLModel) = Matrix(length(d))

# extinction probabilities
function get_ϵ!(d::DLModel)
    for e in d.porder
        d[e, :ϵ] = if isleaf(d, e)
            p = 0.0
        else
            prod([ϵ_slice(d[c, :λ], d[c, :μ], parentdist(d, c), d[c, :ϵ])
                for c in childnodes(d, e)])
        end
    end
end

ϵ_slice(λ, μ, t, ε) = isapprox(λ, μ, atol=1e-5) ?
    1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
        (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ

# NOTE implement for matrix s.t. we only traverse the tree once for large data
function get_M(d::DLModel, x::AbstractVector)
    M = initvector(d)
    for n in d.porder
        M[n] = isleaf(d, n) ? x[d[n]] : sum([M[c] for c in childnodes(d, n)])
    end
    return M
end
