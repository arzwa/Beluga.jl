using AbstractTrees, Parameters

abstract type AbstractNode{T} end

Base.length(n::AbstractNode) = length(n.children)
Base.iterate(n::AbstractNode, args...) = iterate(n.children, args...)

children(node::N) where N<:AbstractNode = node.children

# this approach with Event is not easy (possible?) to get type stable...
abstract type Event{T} end

mutable struct Leaf{T} <: Event{T}
    s::Symbol
    λ::T
    μ::T
    t::T
end

mutable struct Speciation{T} <: Event{T}
    λ::T
    μ::T
    t::T
end

mutable struct WGD{T} <: Event{T}
    q::T
    t::T
end

mutable struct Root{T} <: Event{T}
    λ::T
    μ::T
    η::T
end

struct WGDAfter{T} <: Event{T} ; end

const Event1{T} = Union{Speciation{T},Root{T},WGDAfter{T}}

@with_kw mutable struct PhyloBDPNode{T,E<:Event{T}} <: AbstractNode{T}
    event   ::E
    expr    ::Vector{T} = zeros(Float64, 2)
    trpr    ::Matrix{T} = zeros(Float64, 0, 0)
    parent  ::Union{PhyloBDPNode{T,Event{T}},Nothing} = nothing
    children::Vector{PhyloBDPNode{T,Event{T}}} = PhyloBDPNode{Float64,Event{Float64}}[]
end

setexpr!(n::PhyloBDPNode) = setexpr!(n, n.event)
setexpr!(n, e::Leaf{T}) where T = n.expr[2] = zero(T)

function setexpr!(n, e::Event1{T}) where T
    n.expr[2] = one(T)
    for c in n
        c.expr[1] = one(T)
        @unpack λ, μ, t = c.event
        c.expr[1] = approx1(ep(λ, μ, t, c.expr[2]))
        n.expr[2] *= c.expr[1]
    end
    n.expr[2] = approx1(n.expr[2])
end

function setexpr!(n, e::WGD{T}) where T
    c = first(n)
    @unpack q = e
    @unpack λ, μ, t = c.event
    c.expr[1] = eps = c.expr[2]
    n.expr[2] = c.expr[1] = q*ϵps^2 + (1. - q)*eps
end

ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    approx1((μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ)
approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x


# test
l = Leaf(:ath, 0.2, 0.3, 0.8)
k = Leaf(:cpa, 0.5, 0.6, 0.8)
n = Root(0.4, 0.1, 0.9)
x = PhyloBDPNode(event=l)
y = PhyloBDPNode(event=k)
z = PhyloBDPNode(event=n, children=[x,y])
