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
    using Optim
    using CSV
    using MCMCChains
    import StatsFuns: logaddexp, log1mexp, log1pexp
    # using BirthDeathProcesses

    include("speciestree.jl")
    include("core.jl")
    include("profile.jl")
    include("gbm.jl")
    include("priors.jl")
    include("mcmc.jl")
    include("gradient.jl")
    include("mle.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray, set_L!, set_Ltmp!, GBM, mcmc!,
        DLChain, GBMRatesPrior, LogUniform, ConstantRatesPrior, NhRatesPrior,
        IIDRatesPrior, nrates, nwgd, gradient, mle
end
