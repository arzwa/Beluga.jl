# instead of trying to fit the previous rates prior in the mixture, maybe just
# abandon the previous ratespriors? the previous models were a special case with
# K = 1 anyway right?

# Most sensible model: η across clusters, ν for each cluster

# q: there should be a global q, and relative retention rates, allowing to
# detect different retention patterns across families; but I guess we should
# worry about that later

# abstract RatesPrior
abstract type RatesPrior end
const Prior = Union{<:Distribution,Array{<:Distribution,1},<:Real}

# HACK for constant priors (not proper ditributions)
Base.rand(x::Real) = x
logpdf(x::Real, y) = 0.

# Mixtures
# ========


struct MixturePrior
    mixed::Array{Symbol,1}
    unmixed::Array{Symbol,1}
    λ::Prior
    μ::Prior
    q::Prior
    ν::Prior
    η::Prior
    α::Prior
    K::Int64
end

function Base.rand(d::MixturePrior)#, tree::SpeciesTree)
    @unpack mixed, unmixed, K = d
    state = State()
    nr = nrates(tree)
    nw = nwgd(tree)
    for s in unmixed
        if s in [:λ, :μ]
            state[s] = rand(getfield(d, s), nr)
        elseif s == :q
            state[s] = rand(getfield(d, s), nw)
        else
            state[s] = rand(getfield(d, s))
        end
    end
    for s in mixed
        for i=1:K
            s_ = Symbol("$s$i")
            if s in [:λ, :μ]
                state[s_] = rand(getfield(d, s), nr)
            elseif s == :q
                state[s_] = rand(getfield(d, s), nw)
            else
                state[s_] = rand(getfield(d, s))
            end
        end
    end
    state[:w] = rand(Dirichlet(K, state[:α]))
    state
end
