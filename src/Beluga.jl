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
    include("gbm.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray, set_L!, set_Ltmp!, GBM
end
