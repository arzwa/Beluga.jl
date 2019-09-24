module Beluga

    using AdaptiveMCMC
    using DataFrames
    using Distributed
    using DistributedArrays
    using Distributions
    using Parameters
    using PhyloTrees
    using StatsBase
    using ForwardDiff
    # using BirthDeathProcesses
    # using Optim

    include("speciestree.jl")
    include("core.jl")
    include("profile.jl")
    include("gbm.jl")
    include("priors.jl")
    include("mcmc.jl")
    include("gradient.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray, set_L!, set_Ltmp!, GBM, mcmc!,
        DLChain, GBMRatesPrior, LogUniform, ConstantRatesPrior, ExpRatesPrior,
        IIDRatesPrior, nrates, nwgd, gradient
end
