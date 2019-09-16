module Beluga

    using Distributions
    using BirthDeathProcesses
    using PhyloTrees
    using DataFrames
    using ForwardDiff
    using Parameters
    using Optim
    using StatsBase
    using AdaptiveMCMC
    using DistributedArrays

    include("speciestree.jl")
    include("dlwgd.jl")
    include("profile.jl")
    #include("dlmodel.jl")
    #include("csurosmiklos.jl")
    #include("mle.jl")
    #include("mcmc.jl")
    #include("gbm.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray
end
