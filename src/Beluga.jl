module Beluga

    using AdaptiveMCMC
    using DataFrames
    using Distributed
    using DistributedArrays
    using Distributions
    using Parameters
    using PhyloTrees
    using StatsBase
    # using BirthDeathProcesses
    # using ForwardDiff
    # using Optim

    include("speciestree.jl")
    include("dlwgd.jl")
    include("profile.jl")
    include("gbm.jl")
    include("mcmc.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray, set_L!, set_Ltmp!, GBM, mcmc!,
        DLChain, GBMRatesPrior
end
