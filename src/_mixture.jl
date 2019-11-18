# Mixture models
# ==============

# simple (discrete Γ) mixture
@with_kw struct DLWGDSimpleMixture{T<:Real} <: PhyloBDPModel{T}
    components::Vector{DLWGD{T}}
    K::Int64 = 4
end

const DLWGDMix{T} = DuplicationLossWGDMixture{T}

DuplicationLossWGDMixture(m::DLWGD, K::Int64) =
    DuplicationLossWGDMixture(components=[deepcopy(m) for i=1:K], K=K)


# TODO: adapt all methods to work on mixture by simply iterating over components
# metaprogramming? broadcasting?
logpdf(d::DLWGDMix, x::Vector{Int64}) = sum(logpdf.(d.components, [x]))
setrates!(d::DLWGDMix{T}, x::Matrix{T}) where T = setrates!.(d.components, [x])
# update functions for MCMC that iterate over the same ModelNode across models
