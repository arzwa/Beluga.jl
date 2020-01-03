# Node/ModelNode
# NOTE: We have branch models and node models. In branch models, rates are
# defined for branches directly, whereas in node models rates are defined for
# nodes, and branch-rates are computed as the arithmetic or geometric mean of the
# two flanking nodes. The BDPs along each branch are always defined based on the
# branch rates.
# NOTE: I have experimented with a type tree instead of the 'kind' field, so
# where I have a different type for each kind of node, each a subtype of some
# abstract DLWGD node. However I failed to obtain good performance.

# node rates (for BM prior etc.)
@with_kw mutable struct Node{T}
    θ::Dict{Symbol,T}          # parameter vector (λ, μ, q, η, [κ])
    ϵ::Vector{T} = ones(T, 2)  # extinction probabilities
    W::Matrix{T}               # transition probability matrix
    kind::Symbol               # fake node type, for 'manual' dispatch
    # NOTE: potential efficiency increase, use vector instead of Dict for θ
end

# branch instead of node rates
@with_kw mutable struct Branch{T}
    θ::Dict{Symbol,T}          # parameter vector (λ, μ, q, η, [κ])
    ϵ::Vector{T} = ones(T, 2)  # extinction probabilities
    W::Matrix{T}               # transition probability matrix
    kind::Symbol               # fake node type, for 'manual' dispatch
end

const ModelNode{T} = Union{TreeNode{Node{T}},TreeNode{Branch{T}}} where T
nodetype(n::TreeNode{Node{T}}) where T = Node
nodetype(n::TreeNode{Branch{T}}) where T = Branch

Base.setindex!(n::ModelNode{T}, v::T, s::Symbol) where T = n.x.θ[s] = v
Base.getindex(n::ModelNode, s::Symbol) = n.x.θ[s]
Base.getindex(n::ModelNode, args::Symbol...) = [n.x.θ[s] for s in args]
Base.eltype(n::TreeNode{T}) where T = T

# Different node kinds
rootnode(η::T, λ::T, μ::T, m::Int64, nt::Type) where T<:Real =
    TreeNode(1, nt{T}(
        θ=Dict(:t=>0.,:λ=>λ,:μ=>μ,:η=>η),
        W=zeros(m,m),
        kind=:root))

spnode(i::Int64, p::V, λ::T, μ::T, t::T) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>t,:λ=>λ,:μ=>μ),
        W=zeros(size(p.x.W)),
        kind=:sp), p)

wgdnode(i::Int64, p::V, q::T, t::T) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>t,:q=>q),
        W=zeros(size(p.x.W)),
        kind=:wgd), p)

wgtnode(i::Int64, p::V, q::T, t::T) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>t,:q=>q),
        W=zeros(size(p.x.W)),
        kind=:wgt), p)

wgdafternode(i::Int64, p::V) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>0.),
        W=zeros(size(p.x.W)),
        kind=:wgdafter), p)

iswgd(n::ModelNode) = n.x.kind == :wgd
iswgt(n::ModelNode) = n.x.kind == :wgt
iswgm(n::ModelNode) = iswgd(n) || iswgt(n)
iswgdafter(n::ModelNode) = n.x.kind == :wgdafter && n.p.x.kind == :wgd
iswgtafter(n::ModelNode) = n.x.kind == :wgdafter && n.p.x.kind == :wgt
isawgd(n::ModelNode) = iswgd(n) || iswgdafter(n) || iswgt(n) || iswgtafter(n)
issp(n::ModelNode) = n.x.kind == :sp
gete(n::ModelNode, i::Int64) = n.x.ϵ[i]
getw(n::ModelNode, i::Int64, j::Int64) = n.x.W[i,j]

function nonwgdparent(n::ModelNode)
    # NOTE assumes n is a wgdnode, returns n if not a wgd node
    while isawgd(n) ; n = n.p ; end; n
end

function nonwgdchild(n::ModelNode)
    # NOTE assumes n is a wgdnode, returns n if not a wgd node
    while isawgd(n); n = first(n.c); end; n
end

function closestnode(n::ModelNode{T}, t::T) where T
    t_ = t - n[:t]
    while t_ > 0
        t = t_
        n = n.p
        t_ -= n[:t]
    end
    n, t
end

update!(n::TreeNode{Branch{T}}) where T = setabove!(n)

function update!(n::TreeNode{Node{T}}) where T
    setbelow!(n)
    setabove!(n)
end

function update!(n::ModelNode{T}, θ::Symbol, v::T) where T
    n[θ] = v
    update!(n)
end

function update!(n::ModelNode, θ::NamedTuple)
    for (k, v) in pairs(θ)
        n[k] = v
    end
    update!(n)
end

function set!(n::ModelNode)
    setϵ!(n)
    setW!(n)
end

function setabove!(n::ModelNode)
    while !isnothing(n)
        set!(n)
        n = n.p
    end
end

function setbelow!(n::ModelNode)
    for c in n.c
        set!(c)
    end
end

function setϵ!(n::ModelNode{T}) where T<:Real
    if isleaf(n)
        n.x.ϵ[2] = 0.
    elseif iswgd(n)
        q = n[:q]
        c = first(n.c)
        λ, μ = getλμ(n, c)
        c.x.ϵ[1] = ϵc = gete(c, 2)
        n.x.ϵ[2] = c.x.ϵ[1] = q*ϵc^2 + (1. - q)*ϵc
    elseif iswgt(n)
        # q1, q2 = n[:q1, :q2]
        q = n[:q]
        c = first(n.c)
        λ, μ = getλμ(n, c)
        c.x.ϵ[1] = ϵc = gete(c, 2)
        n.x.ϵ[2] = c.x.ϵ[1] = q*ϵc^3 + 2q*(1. - q)*ϵc^2 + (1. - q)^2*ϵc
    else
        n.x.ϵ[2] = one(T)
        for c in n.c
            c.x.ϵ[1] = one(T)
            λ, μ = getλμ(n, c)
            c.x.ϵ[1] = approx1(ep(λ, μ, c[:t], gete(c, 2)))
            n.x.ϵ[2] *= c.x.ϵ[1]
        end
        n.x.ϵ[2] = approx1(n.x.ϵ[2])
    end
end

function setW!(n::ModelNode{T}) where T<:Real
    if isroot(n)
        return
    end
    λ, μ = getλμ(n.p, n)
    ϵ = gete(n, 2)
    if iswgdafter(n)
        q = n.p[:q]
        wstar_wgd!(n.x.W, n[:t], λ, μ, q, ϵ)
    elseif iswgtafter(n)
        q = n.p[:q]
        wstar_wgt!(n.x.W, n[:t], λ, μ, q, ϵ)
    else
        wstar!(n.x.W, n[:t], λ, μ, ϵ)
    end
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

function wstar_wgd!(w, t, λ, μ, q, ϵ)
    # compute w* (Csuros Miklos 2009)
    w[1,1] = 1.
    w[2,2] = ((1. - q) + 2q*ϵ)*(1. - ϵ)
    w[2,3] = q*(1. - ϵ)^2
    mmax = size(w)[1]-1
    for i=1:mmax, j=2:mmax
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1]
    end
end

function wstar_wgt!(w, t, λ, μ, q, ϵ)
    # compute w* (Csuros Miklos 2009)
    q1 = q
    q2 = 2q*(1. - q)
    q3 = (1. - q)^2
    w[1,1] = 1.
    w[2,2] = q1*(1. - ϵ) + 2*q2*ϵ*(1. - ϵ) + 3*q3*(ϵ^2)*(1. - ϵ)
    w[2,3] = q2*(1. - ϵ)^2 + 3q3*ϵ*(1. - ϵ)^2
    w[2,4] = q3*(1. - ϵ)^3
    mmax = size(w)[1]-1
    for i=1:mmax, j=3:mmax
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1] + w[2,4]*w[i, j-2]
    end
end

function getλμ(a::TreeNode{Node{T}}, b::TreeNode{Node{T}}) where T
    a = isawgd(a) ? nonwgdparent(a) : a
    b = isawgd(b) ? nonwgdchild(b)  : b
    # 0.5*(a[:λ, :μ] .+ b[:λ, :μ]) below is faster
    [(a[:λ]+b[:λ])/2., (a[:μ]+b[:μ])/2.]
end

function getλμ(a::TreeNode{Branch{T}}, b::TreeNode{Branch{T}}) where T
    b = isawgd(b) ? nonwgdchild(b) : b
    [b[:λ], b[:μ]]
end
