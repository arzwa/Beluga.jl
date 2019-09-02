
struct PhyloBDPChain{T<:Model}
    T::SpeciesTree
    state::State
    prior::T
    samplers::Samplers
    gen::Int64
    df::DataFrame
end

#= example state:
    η  => float
    ν  => float
    λ  => [rate for each branch of T, 0 is mean]
    μ  => [rate for each branch of T, 0 is mean]
    q  => [retention rate for each WGD]
    K  => number of clusters
    z  => [cluster assignment for each family]
    rθ => [relative dup-loss rate for each cluster]
    rq => [relative retention rate for each cluster]

The priors are like in Whale for λ, μ, q, η and ν. K and z are governed by the
DP mixture with as base measure two Gamma distributions, one for the rθ and one
for the rq. =#
