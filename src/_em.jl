@with_kw mutable struct ExpMax{T<:Real}
    state ::Dict{Symbol,T}
    latent::Matrix
    model ::DLWGD{T}
end

function init!(em::ExpMax, p::PArray)
    @unpack model = em
end
