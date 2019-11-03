
# ==============================================================================
# separating the tree
using PhyloTree, Parameters
import PhyloTree: isroot, isleaf

abstract type ModelNode{T} end

mutable struct Speciation{T} <: ModelNode{T}
    t::T
    θ::Vector{T}
    ϵ::Vector{T}
    W::Matrix{T}
end

mutable struct WGD{T} <: ModelNode{T}
    t::T
    q::T
    ϵ::Vector{T}
    W::Matrix{T}
end

mutable struct Root{T} <: ModelNode{T}
    η::T
    θ::Vector{T}
    ϵ::Vector{T}
end

const DLWGD{T} = DuplicationLossWGDModel{T}

struct DuplicationLossWGDModel{T<:Real}
    tree  ::Dict{Int64,TreeNode{Float64}}
    nodes ::Dict{Int64,ModelNode{T}}
    leaves::Dict{Int64,Symbol}
end

function DuplicationLossWGDModel(t::TreeNode, l, θ, η::T, m) where T<:Real
    tree   = Dict{Int64,TreeNode{Float64}}()
    nodes  = Dict{Int64,ModelNode{T}}()
    leaves = Dict{Int64,Symbol}()
    for (i, n) in enumerate(prewalk(t))
        n.i = i
        tree[n.i] = n
        nodes[n.i] = isroot(n) ?
            Root(η, θ, ones(2)) : Speciation(n.x, θ, ones(2), zeros(m,m))
        isleaf(n) ? leaves[n.i] = Symbol(l[n]) : nothing
    end
    DuplicationLossWGDModel(tree, nodes, leaves)
end

function setϵ!(d::DLWGD{T}, i::Int64) where T<:Real
    if isleaf(d.tree[i])
        d.nodes[i].ϵ[2] = 0.
    else
        θ = d.nodes[i].θ
        for c in d.tree[i].c
            λ, μ = (d.nodes[c.i].θ .+ θ) / 2
            d.nodes[c.i].ϵ[1] = ep(λ, μ, gett(c), gete(c, 2))
            x.ϵ[2] *= c.x.ϵ[1]
        end
    end
end
