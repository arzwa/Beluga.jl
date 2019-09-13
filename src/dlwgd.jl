# full reconsideration?
# NOTE: assuming only linear BDPs
# NOTE: assuming geometric at root could be generailzed to uniform,
# Poisson, Negbinomial and geometric

using Beluga, Distributions, Parameters, PhyloTrees
import Beluga: PhyloBDP

mutable struct CsurosMiklos{T<:Real}
    W::Array{T,3}
    ϵ::Array{T,2}
    m::Int64
end

struct DuplicationLossWGD{T<:Real,Ψ<:Arboreal} <: PhyloBDP
    tree::Ψ
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}
    η::T  # geometric prior probability at the root
    value::CsurosMiklos{T}
end

function DuplicationLossWGD(tree::Ψ, λ::Vector{T}, μ::Vector{T}, q::Vector{T},
        η::T, mmax::Int) where {Ψ<:Arboreal,T<:Real}
    ϵ = ones(T, length(tree.order), 2)
    # last dimension of W one too large, unnecessary memory...
    W = zeros(eltype(λ), mmax+1, mmax+1, maximum(tree.order))
    c = CsurosMiklos(W, ϵ, mmax)
    d = DuplicationLossWGD{T,Ψ}(tree, λ, μ, q, η, c)
    get_ϵ!(d)
    get_W!(d, mmax)
    return d
end

Base.show(io::IO, d::DuplicationLossWGD) = show(io, d, (:λ, :μ, :q, :η))

function Distributions.logpdf(d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    L = zeros(T, length(d.tree), maximum(x)+1)
    logpdf!(L, d, x, d.tree.order)
end

function Distributions.logpdf!(L::Matrix{T},
        d::DuplicationLossWGD{T,Ψ},
        x::Vector{Int64},
        branches::Vector{Int64}) where {Ψ<:Arboreal,T<:Real}
    csuros_miklos!(L, x, d.value, d.tree, branches)
    root = branches[end]
    l = integrate_root(L[root,:], d.η, d.value.ϵ[root,2])
    l - log(condition_oib(d.tree, d.η, d.value.ϵ)), L
end

function get_ϵ!(model::DuplicationLossWGD, node=-1)
    # Extinction probabilities, ϵ[node, 2] is the extinction P at the end of the
    # branch leading to `node`, ϵ[node, 1] is the extinction P at the beginning
    # of the branch leading to `node`.
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    order = node == -1 ? tree.order : get_parentbranches(tree, node)
    ϵ[1,1] = NaN  # root has only one extinction P, NaN for safety
    for e in order
        if isleaf(tree, e)
            ϵ[e, 2] = zero(eltype(ϵ))
        elseif Beluga.iswgd(tree, e)
            qe = q[tree.bindex[e, :q]]
            f = childnodes(tree, e)[1]
            ϵ[f, 1] = ep(λ[f], μ[f], parentdist(tree, f), ϵ[f, 2])
            ϵ[e, 2] = qe*ϵ[f, 1]^2 + (1. - qe)*ϵ[f, 1]
        else
            for c in childnodes(tree, e)
                ϵ[c, 1] = ep(λ[c], μ[c], parentdist(tree, c), ϵ[c, 2])
                ϵ[e, 2] *= ϵ[c, 1]
            end
        end
    end
end

# W matrix for Csuros & Miklos algorithm
function get_W!(model::DuplicationLossWGD, mmax::Int64, node=-1)
    @unpack tree, λ, μ, q, η, value = model
    @unpack W, ϵ, m = value
    order = node == -1 ? tree.order : get_parentbranches(tree, node)
    for i in order[1:end-1]  # excluding root node
        μi = μ[tree.bindex[i, :θ]]
        λi = λ[tree.bindex[i, :θ]]
        ϵi = ϵ[i, 2]
        t = parentdist(tree, i)
        if Beluga.iswgd(tree, i)
            qi = q[tree.bindex[i, :q]]
            W[:, :, i] = _wstar_wgd(t, λi, μi, qi, ϵi, m)
        else
            W[:, :, i] = _wstar(t, λi, μi, ϵi, m)
        end
    end
end

function _wstar(t, λ, μ, ϵ, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = (ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n
    ψp = ψ*(1. - ϵ) / _n
    w = zeros(typeof(λ), mmax+1, mmax+1)
    w[1,1] = 1.
    for m=1:mmax, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
    return w
end

function _wstar_wgd(t, λ, μ, q, ϵ, mmax::Int64)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = (ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n
    ψp = ψ*(1. - ϵ) / _n
    w = zeros(typeof(λ), mmax+1, mmax+1)
    w[1,1] = 1.
    w[2,2] = (1. - q)*(1. - ϵ) + 2q*ϵ*(1. - ϵ)
    w[2,3] = q*(1. - ϵ)^2
    for m=1:mmax, n=2:m
        w[n+1, m+1] != 0. ? continue : nothing
        w[n+1, m+1] = w[2,2]*w[n, m] + w[2,3]*w[n-1, m]
    end
    return w
end

function integrate_root(L::Vector{T}, η::T, ϵ::T) where T<:Real
    p = 0.
    for i in 2:length(L)
        f = (1. - ϵ)^(i-1)*η*(1. -η)^(i-2)/(1. -(1. -η)*ϵ)^i
        p += exp(L[i]) * f
    end
    log(p)
end

function condition_oib(tree::Arboreal, η::T, ϵ::Matrix{T}) where T<:Real
    e = findroot(tree)
    f, g  = childnodes(tree, e)
    #root = geometric_extinctionp(ϵ[e, 2], η)
    left = geometric_extinctionp(ϵ[f, 1], η)
    rght = geometric_extinctionp(ϵ[g, 1], η)
    #1. - left - rght + root
    p = (1. -left)*(1. -rght)
    #p = isapprox(p, zero(p), atol=1e-12) ? zero(p) : p  # XXX had some issues
    return p
end

geometric_extinctionp(ϵ::T, η::T) where T<:Real = η*ϵ/(1. - (1. - η)*ϵ)
geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)

# BDP utilities
getϕ(t, λ, μ) = λ ≈ μ ?
    λ*t/(1 + λ*t) : μ*(exp(t*(λ-μ)) - 1)/(λ*exp(t*(λ-μ)) - μ)
getψ(t, λ, μ) = λ ≈ μ ?
    λ*t/(1 + λ*t) : (λ/μ)*ϕ(t, λ, μ)
getξ(i, j, k, t, λ, μ) = binomial(i, k)*binomial(i+j-k-1,i-1)*
    getϕ(t, λ, μ)^(i-k)*getψ(t, λ, μ)^(j-k)*(1-getϕ(t, λ, μ)-getψ(t, λ, μ))^k
tp(a, b, t, λ, μ) = (a == b == 0) ? 1.0 :
    sum([ξ(a, b, k, t, λ, μ) for k=0:min(a,b)])
ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    (μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ

# example
t, M = Beluga.example_data2()
d = DuplicationLossWGD(t, rand(8)./100, rand(8)./100, [0.2], 0.8, 11)
