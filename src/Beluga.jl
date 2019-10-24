module Beluga

    # using AdaptiveMCMC
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
    # using MCMCChains
    import StatsFuns: logaddexp, log1mexp, log1pexp
    import Distributions: logpdf!, logpdf

    include("speciestree.jl")
    include("core.jl")
    include("profile.jl")
    # include("gbm.jl")
    # include("priors.jl")
    # include("mcmc.jl")
    include("gradient.jl")
    include("mle.jl")
    include("sim.jl")
    include("mprofile.jl")
    # include("mixture.jl")

    export
        SpeciesTree, profile, PhyloBDP, DuplicationLoss, DuplicationLossWGD,
        logpdf, logpdf!, Profile, PArray, set_L!, set_Ltmp!,  nrates, nwgd,
        gradient, mle, addwgd!, AbstractProfile, MixtureProfile, MPArray

        # GBM, mcmc!, DLChain, GBMRatesPrior, LogUniform, ConstantRatesPrior, NhRatesPrior, IIDRatesPrior, MixtureChain, MixtureProfile
end
