# Utilities
minfs(::Type{T}, dims::Tuple{}) where T<:Real = Array{T}(fill(-Inf, dims))
minfs(::Type{T}, dims::Union{Integer, AbstractUnitRange}...) where T<:Real =
    Array{T}(fill(-Inf, dims))


# BDP utilities
getϕ(t, λ, μ) = λ ≈ μ ? λ*t/(1. + λ*t) : μ*(exp(t*(λ-μ))-1.)/(λ*exp(t*(λ-μ))-μ)
getψ(t, λ, μ) = λ ≈ μ ? λ*t/(1. + λ*t) : (λ/μ)*getϕ(t, λ, μ)
getξ(i, j, k, t, λ, μ) = binomial(i, k)*binomial(i+j-k-1,i-1)*
    getϕ(t, λ, μ)^(i-k)*getψ(t, λ, μ)^(j-k)*(1-getϕ(t, λ, μ)-getψ(t, λ, μ))^k
tp(a, b, t, λ, μ) = (a == b == 0) ? 1.0 :
    sum([getξ(a, b, k, t, λ, μ) for k=0:min(a,b)])


# NOTE: when μ >> λ, numerical issues sometimes result in p > 1.
ep(λ, μ, t, ε) = λ ≈ μ ? 1. + (1. - ε)/(μ * (ε - 1.) * t - 1.) :
    approx1((μ + (λ - μ)/(1. + exp((λ - μ)*t)*λ*(ε - 1.)/(μ - λ*ε)))/λ)
approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x

geometric_extinctionp(ϵ::Real, η::Real)=geometric_extinctionp(promote(ϵ, η)...)
geometric_extinctionp(ϵ::T, η::T) where T<:Real = η + ϵ -log1mexp(log1mexp(η)+ϵ)


# Tree utilities
# get distance from n to m, where it is guaranteed that m is above n
function parentdist(n::ModelNode{T}, m::ModelNode{T}) where T<:Real
    d = zero(T)
    while n != m
        d += n[:t]
        n = n.p
    end
    return d
end

treelength(d::DLWGD) = sum([n[:t] for (i,n) in d.nodes])

id(node::ModelNode, args::Symbol...) = [id(node, s) for s in args]
id(node::ModelNode, s::Symbol) = Symbol("$s$(node.i)")

clade(m, n) = [m.leaves[x.i] for x in postwalk(n) if haskey(m.leaves, n.i)]


# Trace utilities
function freqmap(x)
    c = countmap(x)
    !haskey(c, 0) ? c[0] = 0 : nothing
    haskey(c, missing) ? c[0] += c[missing] : nothing
    delete!(c, missing)
    N = sum(values(c))
    d = Dict{Int64,Float64}()
    for (k,v) in c
        d[k] = v/N
    end
    d
end

function clade(d::DLWGD, n::TreeNode)
    leaves = Symbol[]
    for node in postwalk(n)
        if isleaf(node)
            push!(leaves, d.leaves[node.i])
        end
    end
    leaves
end

clades(m::DLWGD) = Dict(join(sort(clade(m,v)), ",")=>n for (n,v) in m.nodes)

lca_node(d::DLWGD, s1::Symbol) = lca_node(d, s1, s1)

function lca_node(d::DLWGD, s1::Symbol, s2::Symbol)
    n = first([d.nodes[x] for (x,n) in d.leaves if n == s1])
    while !(s2 in clade(d, n))
        n = n.p
    end
    n
end
