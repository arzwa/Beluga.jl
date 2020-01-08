"""
    ConstantDistribution(x)

A 'constant' distribution (Dirac mass), sometimes useful.
"""
struct ConstantDistribution{T}
    x::T
end

Base.rand(x::ConstantDistribution) = x
pdf(x::ConstantDistribution, y) = y ≈ x.x ? 1. : 0.
logpdf(x::ConstantDistribution, y) = log(pdf(x, y))

"""
    UpperBoundedGeometric{T<:Real}

An upper bounded geometric distribution, basically a constructor for a
`DiscreteNonParametric` distribution with the relevant probabilities.
"""
struct UpperBoundedGeometric{T<:Real} <: DiscreteUnivariateDistribution
    d::DiscreteNonParametric{Int64,T,UnitRange{Int64},Array{T,1}}
    p::T
    b::Int64
    function UpperBoundedGeometric(p::T, bound::Int64) where T<:Real
        xs = pdf.(Geometric(p), 0:bound)
        ps = xs ./ sum(xs)
        new{T}(DiscreteNonParametric(0:bound, ps), p, bound)
    end
end

Base.rand(d::UpperBoundedGeometric) = rand(d.d)
pdf(d::UpperBoundedGeometric, x::Int64) = pdf(d.d, x)
logpdf(d::UpperBoundedGeometric, x::Int64) = logpdf(d.d, x)

abstract type RevJumpPrior end

"""
    BMRevJumpPrior

Bivariate autocorrelated rates (Brownian motion) prior inspired by coevol
(Lartillot & Poujol 2010) with an Inverse Wishart prior on the unknown
covariance 2×2 matrix. Crucially, this is defined for the `Node` based model,
i.e. states at model nodes are assumed to be states at *nodes* of the phylogeny.
"""
@with_kw struct BMRevJumpPrior{T,U,V,W} <: RevJumpPrior
    Ψ ::Matrix{Float64}  = [500. 0. ; 0. 500.]
    X₀::T                = MvNormal([1.,1.])
    πη::U                = Beta(3., 1)
    πq::V                = Beta()
    πK::W                = Geometric(0.5)
    Tl::Float64
    @assert isposdef(Ψ)
end

# one-pass prior computation based on the model
function logpdf(prior::BMRevJumpPrior, d::DLWGD)
    @unpack Ψ, X₀, πη, πq, πK, Tl = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(N, M)
    A = zeros(M,M)
    for (i, n) in d.nodes
        if iswgdafter(n)
            continue
        elseif iswgd(n)
            p += logpdf(πq, n[:q]) - log(Tl)
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, log.(n[:λ, :μ]))
        else
            pt = nonwgdparent(n.p)
            Δt = parentdist(n, pt)
            Y[i-1,:] = (log.(n[:λ, :μ]) - log.(pt[:λ, :μ])) / √Δt
            A += Y[i-1,:]*Y[i-1,:]'
            J *= Δt
        end
    end
    p += logp_pics(Ψ, (Y=Y, J=J^(-M/2), A=A, q=M+1, n=N))
    p + logpdf(πK, k)
end

# p(Y|Ψ,q), q = df, n = # of branches
function logp_pics(Ψ, θ)
    @unpack J, A, q, n = θ
    # in our case the Jacobian is a constant (tree and times are fixed)
    log(J) + (q/2)*log(det(Ψ)) - ((q + n)/2)*log(det(Ψ + A))
end

function Base.rand(prior::BMRevJumpPrior, d::DLWGD, k::Int64=-1)
    @unpack Ψ, X₀, πη, πq, πK = prior
    model = deepcopy(d)
    Σ = rand(InverseWishart(3, Ψ))
    k = k < 0 ? rand(prior.πK) : k
    wgds = Dict()
    for i=1:k
        n, t = randpos(model)
        q = rand(πq)
        wgdnode = addwgd!(model, n, t, q)
        child = nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (child.i, q)
    end
    for n in prewalk(model[1])
        if isawgd(n)
            continue
        elseif isroot(n)
            n[:η] = rand(πη)
            r = exp.(rand(X₀))
            n[:λ] = r[1]
            n[:μ] = r[2]
        else
            θp = log.(nonwgdparent(n.p)[:λ, :μ])
            t = parentdist(n, nonwgdparent(n.p))
            θ = exp.(rand(MvNormal(θp, Σ*t)))
            n[:λ] = θ[1]
            n[:μ] = θ[2]
        end
    end
    set!(model)
    (model=model, Σ=Σ, η=model[1,:η], rates=getrates(model), wgds=wgds)
end

"""
    IRRevJumpPrior

Bivariate uncorrelated rates prior with an Inverse Wishart prior on the unknown
covariance 2×2 matrix. Crucially, this is defined for the `Branch` based model,
i.e. states at model nodes are assumed to be states at *branches* of the
phylogeny.
"""
@with_kw struct IRRevJumpPrior{T,U,V,W,X} <: RevJumpPrior
    Ψ ::Matrix{Float64} = [1 0. ; 0. 1]
    X₀::T               = MvNormal([1., 1.])
    πη::U               = Beta(3., 1)
    πq::V               = Beta()
    πK::W               = Geometric(0.5)
    πE::X               = nothing
    Tl::Float64
    @assert isposdef(Ψ)
end

function logpdf(prior::IRRevJumpPrior, d::DLWGD{T}) where T<:Real
    @unpack Ψ, X₀, πη, πq, πK, Tl, πE = prior
    p = 0.; M = 2; J = 1.; k = 0
    N = ne(d)
    Y = zeros(T, N, M)
    A = zeros(T, M, M)
    X0 = log.(d[1][:λ, :μ])
    for (i, n) in d.nodes
        if iswgmafter(n)
            continue
        elseif iswgm(n)
            p += logpdf(πq, n[:q]) - log(Tl)
            k += 1
        elseif isroot(n)
            p += logpdf(πη, n[:η])
            p += logpdf(X₀, X0)
        else
            rates = n[:λ, :μ]
            Y[i-1,:] = log.(rates) - X0
            A += Y[i-1,:]*Y[i-1,:]'
            if πE != nothing
                t  = parentdist(n, nonwgdparent(n.p))
                p += logpdf(πE, expectedX(rates[1], rates[2], t))
            end
        end
    end
    p += logp_pics(Ψ, (Y=Y, J=1., A=A, q=M+1, n=N))
    p + logpdf(πK, k)
end

function Base.rand(prior::IRRevJumpPrior, d::DLWGD, k::Int64=-1)
    @unpack Ψ, X₀, πη, πq, πK = prior
    model = deepcopy(d)
    update!(model[1], :η, rand(πη))
    Σ = rand(InverseWishart(3, Ψ))
    r = rand(X₀)
    k = k < 0 ? rand(prior.πK) : k
    wgds = Dict()
    for i=1:k
        n, t = randpos(model)
        q = rand(πq)
        wgdnode = addwgd!(model, n, t, q)
        child = nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (child.i, n.i, q, t)
    end
    X = rand(X₀)
    rates = exp.(rand(MvNormal(X, Σ), Beluga.ne(model)+1))
    setrates!(model, rates)
    (model=model, Σ=Σ, η=model[1,:η], rates=rates, wgds=wgds)
end

scattermat(m::DLWGD, pr::IRRevJumpPrior) = scattermat_iid(m)

expectedX(λ, μ, t, X0=1) = X0*exp(t*(λ - μ))

function gradient(pr::IRRevJumpPrior, m::DLWGD{T}) where T<:Real
    v = asvector(m)
    f = (u) -> logpdf(pr, m(u))
    g = ForwardDiff.gradient(f, v)
    return g::Vector{Float64}
end

function Base.write(io::IO, prior::T) where T<:RevJumpPrior
    for f in fieldnames(T)
        write(io, f, ": ", string(getfield(prior, f)), "\n")
    end
end
