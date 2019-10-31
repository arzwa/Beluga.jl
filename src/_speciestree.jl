
mutable struct SpeciesTree{T<:Real} <: AbstractTree
    tree::Tree{T}
    leaves::Dict{Int64,Symbol}
    wgds::Dict{Int64,Tuple}
end

mutable struct WGD{T<:Real}
    t::T
    q::T
end


#=
Starting to think this might make more sense; at least for MCMC purposes etc,
tree is just a topology, times are variable like everything else, WGDs are not
in the topology. There is no peculaiar index; this is external to the model
(i.e. the vector passed can be arbitrarily constrained, but must be of length
# of nodes)
=#
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
