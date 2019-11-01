
#= alternative, but I don't like the dict ======================================
it seems promising
challenge remains to get the two-node WGD representation to a single node...
once that works do we have everything?
TODO: wgds, root integration, conditioning, io from data frame, tree struct
=#
import Beluga: ep, getϕ, getψ, approx1, minfs
import PhyloTree: readnw, isroot, isleaf, TreeNode, postwalk
using Parameters

struct BelugaNode{T}
    t::T
    θ::Dict{Symbol,T}
    ϵ::Vector{T}
    W::Matrix{T}
end

const ModelNode{T} = ModelNode{T} where T

Base.getindex(n::ModelNode{T}, s::Symbol) where T<:Real = n.x.θ[s]
getϵ(n::ModelNode{T}, i::Int64) where T<:Real = n.x.ϵ[i]
getw(n::ModelNode{T}, i::Int64, j::Int64) where T<:Real = n.x.W[i,j]
gett(n::ModelNode{T}) where T<:Real = n.x.t

inittree(nw::String, m=10, λ=1., μ=1., η=0.9) =
    inittree(readnw(nw).t, η, λ, μ, m)

rootnode(η::T, λ::T, μ::T, m) where T<:Real =
    TreeNode(1, BelugaNode{T}(0., Dict(:η=>η,:λ=>λ,:μ=>μ), ones(2), zeros(m,m)))

speciationnode(i::Int64, p, λ::T, μ::T, t::T, m::Int64) where T<:Real =
    TreeNode(i, BelugaNode{T}(t, Dict(:λ=>λ, :μ=>μ), ones(2), zeros(m,m)), p)

wgdnode(i::Int64, p::ModelNode{T}, q::T) where T<:Real =
    TreeNode(i, BelugaNode{T}(t, Dict(:q=>q), ones(2), zeros(m,m)), p)

function inittree(t::TreeNode, η::T, λ::T, μ::T, m) where T<:Real
    i = 0
    function walk(x, y)
        i += 1
        if isroot(x)
            x_ = rootnode(η, λ, μ, m)
        else
            x_ = speciationnode(i, y, λ, μ, x.x, m)
            push!(y.c, x_)
        end
        isleaf(x) ? (return) : [walk(n, x_) for n in x.c]
        return x_
    end
    return walk(t, nothing)
end

function initmodel!(t::TreeNode) where T<:Real
    for n in postwalk(t)
        setϵ!(n)
        setW!(n)
    end
end

function setϵ!(n::ModelNode{T}) where T<:Real
    @unpack x = n
    if isleaf(n)
        x.ϵ[2] = 0.
    else
        for c in n.c
            λ, μ = getλμ(n, c)
            c.x.ϵ[1] = ep(λ, μ, gett(c), getϵ(c, 2))
            x.ϵ[2] *= c.x.ϵ[1]
        end
    end
end

function setW!(n::ModelNode{T}) where T<:Real
    if isroot(n)
        return
    end
    λ, μ = getλμ(n, n.p)
    ϵ = getϵ(n, 2)
    t = gett(n)
    wstar!(n.x.W, t, λ, μ, ϵ)
end

function wstar!(w, t, λ, μ, ϵ)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = approx1((ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n)
    ψp = approx1(ψ*(1. - ϵ) / _n)
    w[1,1] = 1.
    for m=1:size(w)[1]-1, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
end

function wstar!(w, t, λ, μ, ϵ)
    # compute w* (Csuros Miklos 2009)
    ϵ = log(ϵ)
    ϕ = log(getϕ(t, λ, μ))  # p
    ψ = log(getψ(t, λ, μ))  # q
    _n = log1mexp(ψ+ϵ)
    ϕp = logaddexp(ϕ+log1mexp(ϵ), log1mexp(ψ)+ϵ) - _n
    ψp = ψ+log1mexp(ϵ) - _n
    w[1,1] = 0.
    for m=1:size(w)[1]-1, n=1:m
        w[n+1, m+1] = logaddexp(ψp + w[n+1, m],
            log1mexp(ϕp) + log1mexp(ψp) + w[n, m])
    end
end

getλμ(n1, n2) where T<:Real = [(n1[:λ] .+ n2[:λ])/ 2., (n1[:μ] .+ n2[:μ])/ 2.]

# not decently tested
function csuros_miklos(x, node::ModelNode{T}) where T<:Real
    L = minfs(T, length(x), maximum(x)+1)
    for n in postwalk(node)
        csuros_miklos!(L, x, n)
    end
    L
end

function csuros_miklos!(L::Matrix{T}, x::AbstractVector{Int64},
        node::ModelNode{T}) where T<:Real
    @unpack W, ϵ = node.x
    mx = maximum(x)
    if isleaf(node)
        L[node.i, x[node.i]+1] = 0.
    else
        Mc = [x[c.i] for c in node.c]
        _M = cumsum([0 ; Mc])
        _ϵ = cumprod([1.; [getϵ(c, 1) for c in node.c]])
        B = minfs(eltype(_ϵ), length(Mc), _M[end]+1, mx+1)
        A = minfs(eltype(_ϵ), length(Mc), _M[end]+1)
        for i = 1:length(Mc)
            c = node[i]
            Mi = Mc[i]

            Wc = c.x.W[1:mx+1, 1:mx+1]
            B[i, 1, :] = log.(Wc * exp.(L[c.i, 1:mx+1]))

            for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                B[i,t+1,s+1] = s == Mi ? B[i,t,s+1] + log(getϵ(c, 1)) :
                    logaddexp(B[i,t,s+2], log(getϵ(c, 1))+B[i,t,s+1])
            end
            if i == 1
                for n=0:_M[i+1]  # this is 0 ... M[i]
                    A[i,n+1] = B[i,1,n+1] - n*log(1. - _ϵ[2])
                end
            else
                # XXX is this loop as efficient as it could?
                for n=0:_M[i+1], t=0:_M[i]
                    s = n-t
                    if s < 0 || s > Mi
                        continue
                    else
                        p = _ϵ[i]
                        if !(zero(p) <= p <= one(p))
                            @error "Invalid extinction probability ($p)"
                            p = one(p)
                        end
                        lp = logpdf(Binomial(n, p), s) +
                            A[i-1,t+1] + B[i,t+1,s+1]
                        A[i,n+1] = logaddexp(A[i,n+1], lp)
                    end
                end
                for n=0:_M[i+1]  # this is 0 ... M[i]
                    A[i,n+1] -= n*log(1. - _ϵ[i+1])
                end
            end
        end
        for n=0:x[node.i]
            L[node.i, n+1] = A[end, n+1]
        end
    end
end


# ==============================================================================
# two opposing ways of doing things:
#   (1) everything in the tree (nodes contain parameters)
#   (2) tree is only topology, rest are vectors

#= (1)

=#
# separating the tree
import PhyloTree: read_nw, isroot, isleaf

abstract type AbstractEvent{T} end

mutable struct Speciation{T} <: AbstractEvent{T}
    t::T
    θ::Vector{T}
    ϵ::Vector{T}
    W::Matrix{T}
end

mutable struct WGD{T} <: AbstractEvent{T}
    t::T
    q::T
    ϵ::Vector{T}
    W::Matrix{T}
end

mutable struct Root{T} <: AbstractEvent{T}
    η::T
    θ::Vector{T}
    ϵ::Vector{T}
end

function inittree(t::TreeNode, m, λ, μ)
    i = 0
    function walk(x, y)
        i += 1
        if isroot(x)
            x_ = TreeNode(i, Root(0.9, [λ, μ], ones(2)),
                nothing, TreeNode{AbstractEvent}[])
        else
            x_ = TreeNode(i, Speciation(1., [λ, μ], ones(2), zeros(m,m)),
                y, TreeNode{AbstractEvent}[])
            push!(y.c, x_)
        end
        isleaf(x) ? (return) : [walk(n, x_) for n in x.c]
        return x_
    end
    return walk(t, nothing)
end
# direct approach
# problems are postorders
import PhyloTrees: isleaf, isroot
abstract type AbstractNode{T} end
abstract type AbstractSpeciationNode{T} <: AbstractNode{T} end

mutable struct SpeciationNode{T<:Real} <: AbstractSpeciationNode{T}
    i::Int64
    t::T
    θ::Vector{T}
    ϵ::Vector{T}
    W::Matrix{T}
    p::AbstractNode
    c::Vector{AbstractNode{T}}
end

mutable struct RootNode{T<:Real} <: AbstractSpeciationNode{T}
    i::Int64
    η::T
    θ::Vector{T}
    ϵ::Vector{T}
    c::Vector{AbstractNode{T}}
end

# think about whether we really need the 2 WGD nodes...
mutable struct WGDNode{T<:Real} <:AbstractNode{T}
    i::Int64
    t::T
    q::T
    ϵ::Vector{T}
    W::Matrix{T}
    p::AbstractNode
    c::Vector{AbstractNode{T}}
end

Base.show(io::IO, t::AbstractNode) = write(io, "$(typeof(t))($(t.i))")
isleaf(n::AbstractNode) = length(n.c) == 0

function inittree(t::LabeledTree, m=10, λ=1., μ=1.)
    i = 0
    function walk(x, y)
        i += 1
        if isroot(t, x)
            x_ = RootNode(1, 0.9, [λ, μ], zeros(2), AbstractNode[])
        else
            x_ = SpeciationNode(i, parentdist(t, x), [λ, μ], zeros(2), zeros(m,m), y, AbstractNode[])
            push!(y.c, x_)
        end
        isleaf(t, x) ? (return) : [walk(n, x_) for n in childnodes(t, x)]
        return x_
    end
    return walk(1, nothing)
end




#= this would compute the matrix by going over the full tree structure
l, L = logpdf(x, tree, true)  # returns matrix
l = logpdf(x, tree)           # returns only logpdf

# such an implementation would allow something like:
logpdf!(L::Matrix, n::AbstractNode)  # compute L from node upwards to the root
    while !isroot(n)
        compute(n)
        n = parentnode(n)
    end
=#


#= (2)
Starting to think this might make more sense; at least for MCMC purposes etc,
tree is just a topology, times are variable like everything else, WGDs are not
in the topology. There is no peculaiar index; this is external to the model
(i.e. the vector passed can be arbitrarily constrained, but must be of length
# of nodes)

mutable struct DuplicationLossWGD{T<:Real,V<:Real}
    tree::Tree{T}
    λ::Vector{V}
    μ::Vector{V}
    q::Vector{V}
    η::V
    W::Array{V,3}
    ϵ::Matrix{V}
    m::Int64
end
=#
